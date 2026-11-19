"""Unit and integration tests for Task A2 (`vec/memory/*`):
  1. External piece-ID tracker 100% identity parity against StrategoEnvironment & VecStratego
  2. Strict zero ground-truth rank leakage in history planes and event sequences
  3. Strict game-ID splitting with zero game overlap
  4. AAREN parallel-vs-sequential O(1) parity, causal padding invariance, forward/backward
     of all 4 models (a_baseline, b_history, c1_aaren, c2_gru), ROC-AUC parity vs sklearn,
     and end-to-end tiny CPU recorder + probe smoke test.
"""

import json
import os
import random

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from environment import StrategoEnvironment
from environment.action_encoding import encode_move
from environment.battle import BattleResolver
from environment.heuristic_setup import HeuristicSetupAgent
from environment.piece import PieceType as P
from vec.compat import placement_to_setup
from vec.engine import BOMB, FLAG, NUM_OBS_PLANES, NUM_RANKS, NUM_SQUARES, VecStratego
from vec.expert import VecExpert, VecRandom

from vec.memory.features import (
    EVENT_DIM,
    MAX_EVENTS,
    NUM_HISTORY_PLANES,
    PieceHistoryTracker,
    extract_batch_sequences,
    scatter_history_planes,
)
from vec.memory.models import (
    MODEL_KINDS,
    PieceAarenEncoder,
    PieceGruEncoder,
    build_model,
)
from vec.memory.probe import (
    binary_roc_auc,
    compute_deltas_and_decision,
    evaluate_model,
    evaluate_public_baseline,
    run_probe,
    split_by_game_id,
)
from vec.memory.recorder import (
    SETUP_HEURISTIC,
    SETUP_RANDOM,
    SETUP_STRUCTURED,
    labeled_diverse_setup_bank,
    record_dataset,
)


def _init_shadow_grid(place_p1, place_p2):
    """Independent 10x10 Python ground-truth piece-origin grid: `(owner, origin_sq_opp_frame, rank, moved)`."""
    grid = [[None for _ in range(10)] for _ in range(10)]
    for pt, (r, c) in place_p1:
        # In P-1's canonical frame, P+1's square (r, c) is (9-r, 9-c) = 99 - (r*10 + c)
        grid[r][c] = (1, 99 - (r * 10 + c), pt.value, False)
    for pt, (r, c) in place_p2:
        # In P+1's canonical frame, P-1's square (r, c) is (r, c) = r*10 + c
        grid[r][c] = (-1, r * 10 + c, pt.value, False)
    return grid


def _step_shadow_grid(grid, move, resolver: BattleResolver):
    (r1, c1), (r2, c2) = move
    att = grid[r1][c1]
    dfn = grid[r2][c2]
    assert att is not None
    grid[r1][c1] = None
    moved_att = (att[0], att[1], att[2], True)
    if dfn is None:
        grid[r2][c2] = moved_att
    else:
        res = resolver.resolve_battle(P(att[2]), P(dfn[2]))
        if res == 1:
            grid[r2][c2] = moved_att
        elif res == -1:
            grid[r2][c2] = dfn
        else:
            grid[r2][c2] = None


def _shadow_canonical_maps(grid, player: int):
    """Map canonical square 0..99 -> `(owner, origin_sq_opp_frame, rank, moved)` for own & opp."""
    own_map, opp_map = {}, {}
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            if cell is None:
                continue
            cr, cc = (r, c) if player == 1 else (9 - r, 9 - c)
            sq = cr * 10 + cc
            if cell[0] == player:
                own_map[sq] = cell
            else:
                opp_map[sq] = cell
    return own_map, opp_map


# ------------------------------------------------------------------ 1. Piece-ID identity parity
def test_tracker_piece_identity_100pct_parity_vs_stratego_environment(cpu):
    """Replay 6 full high-attack games on StrategoEnvironment + an independent ground-truth piece-origin
    grid and on VecStratego + PieceHistoryTracker. Every single ply must match 100% on piece origin
    identity, rank provenance, move flags, and occupancy across battles and 180-degree flips."""
    num_games = 6
    rng = random.Random(42)
    random.seed(42)
    resolver = BattleResolver()
    setups = {1: HeuristicSetupAgent(player_id=1), -1: HeuristicSetupAgent(player_id=-1)}

    vec = VecStratego(num_games, cpu, max_plies=220, no_battle_plies=80)
    tracker = PieceHistoryTracker(num_games, cpu, max_events=16)
    envs, states, shadows = [], [], []
    mover_setups, other_setups, starts = [], [], []
    for g in range(num_games):
        env = StrategoEnvironment(cpu, max_turns=220, no_battle_draw_plies=80)
        place = {
            pl: setups[pl].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(pl))
            for pl in (1, -1)
        }
        start = 1 if g % 2 == 0 else -1
        states.append(env.reset(place[1], place[-1], starting_player=start))
        envs.append(env)
        shadows.append(_init_shadow_grid(place[1], place[-1]))
        mover_setups.append(placement_to_setup(place[start], start))
        other_setups.append(placement_to_setup(place[-start], -start))
        starts.append(start)

    vec.reset(torch.arange(num_games), torch.stack(mover_setups), torch.stack(other_setups), torch.tensor(starts))
    tracker.reset_from_env(vec)

    # Record initial mapping: (g, side_idx, pid) -> (owner, origin_sq_opp_frame, true_rank)
    pid_truth = {}
    for g in range(num_games):
        player = int(vec.side[g])
        cur_s = 0 if player == 1 else 1
        opp_s = 1 - cur_s
        own_map, opp_map = _shadow_canonical_maps(shadows[g], player)
        for sq, (owner, orig_sq, rank, _) in own_map.items():
            pid = int(tracker.own_pid[g, sq])
            assert pid >= 0
            pid_truth[(g, cur_s, pid)] = (owner, orig_sq, rank)
            assert int(tracker.start_sq[g, cur_s, pid]) == orig_sq
        for sq, (owner, orig_sq, rank, _) in opp_map.items():
            pid = int(tracker.opp_pid[g, sq])
            assert pid >= 0
            pid_truth[(g, opp_s, pid)] = (owner, orig_sq, rank)
            assert int(tracker.start_sq[g, opp_s, pid]) == orig_sq

    live = [True] * num_games
    total_plies = 0
    while any(live):
        actions = torch.zeros(num_games, dtype=torch.long)
        for g in range(num_games):
            if not live[g]:
                continue
            env = envs[g]
            player = states[g].current_player
            assert int(vec.side[g]) == player
            cur_s = 0 if player == 1 else 1
            opp_s = 1 - cur_s

            # Exact occupancy parity
            assert torch.equal(tracker.own_pid[g] >= 0, vec.own[g] > 0)
            assert torch.equal(tracker.opp_pid[g] >= 0, vec.opp[g] > 0)

            own_map, opp_map = _shadow_canonical_maps(shadows[g], player)
            assert set(own_map.keys()) == set((tracker.own_pid[g] >= 0).nonzero().flatten().tolist())
            assert set(opp_map.keys()) == set((tracker.opp_pid[g] >= 0).nonzero().flatten().tolist())

            for sq, (owner, orig_sq, rank, moved) in own_map.items():
                pid = int(tracker.own_pid[g, sq])
                exp_owner, exp_orig, exp_rank = pid_truth[(g, cur_s, pid)]
                assert (owner, orig_sq, rank) == (exp_owner, exp_orig, exp_rank)
                assert rank == int(vec.own[g, sq])
                assert int(tracker.start_sq[g, cur_s, pid]) == orig_sq
                assert (int(tracker.move_count[g, cur_s, pid]) > 0) == moved == bool(vec.own_mov[g, sq])

            for sq, (owner, orig_sq, rank, moved) in opp_map.items():
                pid = int(tracker.opp_pid[g, sq])
                exp_owner, exp_orig, exp_rank = pid_truth[(g, opp_s, pid)]
                assert (owner, orig_sq, rank) == (exp_owner, exp_orig, exp_rank)
                assert rank == int(vec.opp[g, sq])
                assert int(tracker.start_sq[g, opp_s, pid]) == orig_sq
                assert (int(tracker.move_count[g, opp_s, pid]) > 0) == moved == bool(vec.opp_mov[g, sq])

            moves = env.get_valid_moves()
            attacks = [m for m in moves if int(states[g].board[m[1][0], m[1][1]]) != 0]
            move = rng.choice(attacks) if attacks and rng.random() < 0.65 else rng.choice(moves)
            actions[g] = encode_move(move, player)
            _step_shadow_grid(shadows[g], move, resolver)
            states[g], _, done, _ = env.step(move)
            total_plies += 1
            if done:
                live[g] = False

        tracker.step(vec, actions)

    assert total_plies > 200


def test_tracker_mid_batch_resets_and_done_masking(cpu):
    """Partial resets mid-batch must reset ONLY finished slots while preserving live slots and
    never mutating done games that stay idle."""
    torch.manual_seed(7)
    bank, _ = labeled_diverse_setup_bank(32, seed=7)
    N = 6
    vec = VecStratego(N, cpu, max_plies=40, no_battle_plies=25)
    tracker = PieceHistoryTracker(N, cpu, max_events=8)
    idx_all = torch.arange(N)
    vec.reset(idx_all, bank[:N], bank[N:2 * N], torch.tensor([1, -1, 1, -1, 1, -1]))
    tracker.reset_from_env(vec)

    # Track ground-truth rank per (g, s, pid)
    truth_rank = torch.zeros((N, 2, 40), dtype=torch.long)

    def record_truth(rows):
        for g in rows.tolist():
            cur_s = 0 if int(vec.side[g]) == 1 else 1
            opp_s = 1 - cur_s
            for sq in range(100):
                if vec.own[g, sq] > 0:
                    truth_rank[g, cur_s, int(tracker.own_pid[g, sq])] = int(vec.own[g, sq])
                if vec.opp[g, sq] > 0:
                    truth_rank[g, opp_s, int(tracker.opp_pid[g, sq])] = int(vec.opp[g, sq])

    record_truth(idx_all)
    pol = VecRandom()

    for step_i in range(60):
        # Force game 0 to finish early on step 15 and remain done for 5 steps before reset
        if step_i == 15:
            vec.done[0] = True
            saved_own_0 = tracker.own_pid[0].clone()
            saved_mc_0 = tracker.move_count[0].clone()

        actions = pol.act(vec)
        done_now, _ = tracker.step(vec, actions)

        if 15 <= step_i < 20:
            assert torch.equal(tracker.own_pid[0], saved_own_0)
            assert torch.equal(tracker.move_count[0], saved_mc_0)

        # Partial reset of any games done (except game 0 during its idle check window)
        reset_mask = vec.done.clone()
        if 15 <= step_i < 20:
            reset_mask[0] = False
        if reset_mask.any():
            r_idx = reset_mask.nonzero().flatten()
            m = r_idx.numel()
            vec.reset(r_idx, bank[:m], bank[m:2 * m], torch.ones(m, dtype=torch.long))
            tracker.reset_from_env(vec, r_idx)
            record_truth(r_idx)
            assert int(tracker.move_count[r_idx].sum()) == 0
            assert int(tracker.event_len[r_idx].sum()) == 0

        # Verify 100% rank provenance on ALL live games
        for g in range(N):
            if vec.done[g]:
                continue
            cur_s = 0 if int(vec.side[g]) == 1 else 1
            opp_s = 1 - cur_s
            assert torch.equal(tracker.own_pid[g] >= 0, vec.own[g] > 0)
            assert torch.equal(tracker.opp_pid[g] >= 0, vec.opp[g] > 0)
            for sq in (vec.own[g] > 0).nonzero().flatten().tolist():
                pid = int(tracker.own_pid[g, sq])
                assert int(truth_rank[g, cur_s, pid]) == int(vec.own[g, sq])
            for sq in (vec.opp[g] > 0).nonzero().flatten().tolist():
                pid = int(tracker.opp_pid[g, sq])
                assert int(truth_rank[g, opp_s, pid]) == int(vec.opp[g, sq])


def test_tracker_revealed_approach_retreat_battles_and_ring_overflow(cpu):
    """Scripted board verifying revealed-rank approach/retreat, battle outcomes, Scout long moves,
    and history truncation beyond `max_events`."""
    vec = VecStratego(1, cpu, max_plies=200, no_battle_plies=200)
    vec.own.zero_()
    vec.opp.zero_()
    for f in (vec.own_rev, vec.opp_rev, vec.own_mov, vec.opp_mov, vec.own_long, vec.opp_long):
        f.zero_()
    vec.done[0] = False
    vec.side[0] = 1
    vec.ply[0] = 0

    # Player +1 (starts moving): Scout at (6, 0), Marshal at (8, 5) (revealed to opp!), Flag at (9, 9)
    vec.own[0, 60] = P.SCOUT.value
    vec.own[0, 85] = P.MARSHAL.value
    vec.own_rev[0, 85] = True  # Player +1's Marshal is revealed!
    vec.own[0, 99] = P.FLAG.value
    # Player -1 (in +1's canonical frame): Spy at (5, 5), Captain at (3, 0), Flag at (0, 0)
    vec.opp[0, 55] = P.SPY.value
    vec.opp[0, 30] = P.CAPTAIN.value
    vec.opp[0, 0] = P.FLAG.value
    vec.legal = vec._legal_mask(vec.own, vec.opp, vec.own_hist)

    tracker = PieceHistoryTracker(1, cpu, max_events=3)
    tracker.reset_from_env(vec)

    def move_action(src_sq: int, dst_sq: int) -> torch.Tensor:
        a = int(vec.t.pair_to_action[src_sq, dst_sq])
        assert a >= 0 and bool(vec.legal[0, a]), f"Illegal move {src_sq}->{dst_sq}"
        return torch.tensor([a], dtype=torch.long)

    # Ply 0 (P+1): Scout makes a 2-square long move 60 -> 40
    tracker.step(vec, move_action(60, 40))
    # Ply 1 (P-1): P-1's Spy is at 99-55=44 in P-1's frame; P+1's revealed Marshal is at 99-85=14.
    # Spy approaches Marshal: 44 -> 34 (in P+1's frame: 55 -> 65, distance to Marshal 85 goes 3 -> 2!)
    tracker.step(vec, move_action(44, 34))

    # Now it is P+1's turn! Check history planes for P-1's Spy (now at sq 65 in P+1's frame).
    hp = tracker.history_planes(vec)[0]  # (48, 10, 10)
    assert hp[16, 6, 5].item() == pytest.approx(1.0, abs=1e-3)                 # ever_approached_rev
    assert hp[17, 6, 5].item() == pytest.approx(P.MARSHAL.value / 12.0, abs=1e-3)  # max_rev_rank_approached
    assert hp[18, 6, 5].item() == pytest.approx(P.MARSHAL.value / 12.0, abs=1e-3)  # min_rev_rank_approached
    assert hp[24 + P.MARSHAL.value - 1, 6, 5].item() == pytest.approx(0.5, abs=1e-3)  # 1 / dist_after(2)
    assert hp[19, 6, 5].item() == pytest.approx(0.0, abs=1e-3)                 # not retreated yet

    # Check live history_planes matches compact snapshot -> scatter_history_planes
    snap = tracker.extract_moved_enemy_snapshot(vec)
    ptr = torch.tensor([0, int(snap['piece_counts'][0])], dtype=torch.long)
    rebuilt = scatter_history_planes(
        snap['piece_summary'], ptr, snap['piece_sq'], vec.opp > 0, vec.opp_mov, vec.ply
    )[0]
    assert torch.allclose(hp, rebuilt, atol=2e-3)

    # Ply 2 (P+1): Scout attacks P-1's Captain at 30 -> Scout (3) loses to Captain (7)!
    tracker.step(vec, move_action(40, 30))
    # After battle, Captain survived (defender stays) -> now at 99-30=69 in P-1's frame, Scout is gone!
    assert int(vec.own[0, 69]) == P.CAPTAIN.value
    assert int(tracker.own_pid[0, 69]) >= 0
    assert int(tracker.ever_attacked[0, 1, int(tracker.own_pid[0, 69])]) == 1
    assert (tracker.opp_pid[0] >= 0).sum().item() == 2  # only Marshal and Flag left for P+1

    # Ply 3 (P-1): Spy retreats from Marshal: 34 -> 44 (in P+1's frame: 65 -> 55, dist 2 -> 3)
    tracker.step(vec, move_action(34, 44))
    hp2 = tracker.history_planes(vec)[0]
    assert hp2[19, 5, 5].item() == pytest.approx(1.0)                # ever_retreated_rev
    assert hp2[36 + P.MARSHAL.value - 1, 5, 5].item() == pytest.approx(0.5)  # 1 / dist_before(2)

    # Drive P-1's Spy through 4 more moves on dry land to exceed max_events=3 and verify ring shift + move_count
    p1_moves = [(85, 84), (84, 83), (83, 84), (84, 85)]
    p2_moves = [(44, 45), (45, 35), (35, 34), (34, 44)]
    for m1, m2 in zip(p1_moves, p2_moves):
        tracker.step(vec, move_action(*m1))
        tracker.step(vec, move_action(*m2))

    # P-1's Spy has now moved 6 times total, while max_events == 3
    spy_pid = int(tracker.opp_pid[0, 55])
    assert int(tracker.move_count[0, 1, spy_pid]) == 6
    assert int(tracker.first_move_ply[0, 1, spy_pid]) == 1
    assert int(tracker.event_len[0, 1, spy_pid]) == 3


# ------------------------------------------------------------------ 2. Zero ground-truth rank leakage
def test_zero_ground_truth_rank_leakage(cpu):
    """Adversarial leakage test: play the exact same public move sequence in two environments
    whose HIDDEN enemy ranks are completely different. Every history plane and every event vector
    must be bit-for-bit identical."""
    def make_env(enemy_ranks_map):
        env = VecStratego(1, cpu, max_plies=100, no_battle_plies=100)
        env.own.zero_()
        env.opp.zero_()
        for f in (env.own_rev, env.opp_rev, env.own_mov, env.opp_mov, env.own_long, env.opp_long):
            f.zero_()
        env.done[0] = False
        env.side[0] = 1
        # P+1 pieces (identical in both envs, some revealed!)
        env.own[0, 75] = P.MARSHAL.value
        env.own_rev[0, 75] = True
        env.own[0, 72] = P.CAPTAIN.value
        env.own_rev[0, 72] = True
        env.own[0, 88] = P.MINER.value
        env.own[0, 99] = P.FLAG.value
        # P-1 pieces (HIDDEN, different ranks in env A vs env B!)
        for sq, rank in enemy_ranks_map.items():
            env.opp[0, sq] = rank
        env.legal = env._legal_mask(env.own, env.opp, env.own_hist)
        tr = PieceHistoryTracker(1, cpu, max_events=16)
        tr.reset_from_env(env)
        return env, tr

    env_a, tr_a = make_env({
        0: P.FLAG.value,
        1: P.BOMB.value,
        2: P.BOMB.value,
        35: P.SPY.value,
        32: P.GENERAL.value,
        38: P.SCOUT.value,
    })
    env_b, tr_b = make_env({
        0: P.BOMB.value,
        1: P.FLAG.value,
        2: P.SERGEANT.value,
        35: P.MARSHAL.value,
        32: P.MINER.value,
        38: P.SCOUT.value,  # keep Scout on 38 so a long move is legal in both
    })

    # Sequence of quiet moves (no battles, so hidden ranks stay hidden!)
    # Alternating P+1 and P-1 canonical (src, dst) pairs:
    move_pairs = [
        (88, 78),  # P+1 Miner
        (99 - 35, 99 - 45),  # P-1 piece on 35 approaches revealed Marshal (75)
        (78, 68),  # P+1 Miner
        (99 - 32, 99 - 22),  # P-1 piece on 32 retreats from revealed Captain (72)
        (68, 78),  # P+1 Miner
        (99 - 38, 99 - 58),  # P-1 Scout makes 2-square long move!
        (78, 88),  # P+1 Miner
        (99 - 45, 99 - 55),  # P-1 piece approaches Marshal again
    ]
    for src, dst in move_pairs:
        a = torch.tensor([int(env_a.t.pair_to_action[src, dst])], dtype=torch.long)
        assert bool(env_a.legal[0, a]) and bool(env_b.legal[0, a])
        tr_a.step(env_a, a)
        tr_b.step(env_b, a)

    # Confirm hidden ranks really differ while tracked features are BIT-FOR-BIT identical
    assert not torch.equal(env_a.opp, env_b.opp)
    hp_a = tr_a.history_planes(env_a)
    hp_b = tr_b.history_planes(env_b)
    assert torch.equal(hp_a, hp_b), "History planes leaked hidden enemy ranks!"

    snap_a = tr_a.extract_moved_enemy_snapshot(env_a)
    snap_b = tr_b.extract_moved_enemy_snapshot(env_b)
    for key in ('piece_counts', 'piece_sq', 'piece_summary', 'event_len', 'events'):
        assert torch.equal(snap_a[key], snap_b[key]), f"Snapshot field `{key}` leaked hidden enemy ranks!"
    assert snap_a['events'].numel() > 0


# ------------------------------------------------------------------ 3. Strict game-ID split
def test_game_id_split_zero_overlap_and_stratification():
    # 100 games, 5 samples each, 4 matchup kinds
    game_ids = torch.arange(100).repeat_interleave(5)
    matchups = (torch.arange(100) % 4).repeat_interleave(5)
    tr, va, te, info = split_by_game_id(game_ids, matchups, ratios=(0.8, 0.1, 0.1), seed=123)

    tr_g = set(game_ids[tr].tolist())
    va_g = set(game_ids[va].tolist())
    te_g = set(game_ids[te].tolist())

    assert tr_g.isdisjoint(va_g)
    assert tr_g.isdisjoint(te_g)
    assert va_g.isdisjoint(te_g)
    assert len(tr_g) + len(va_g) + len(te_g) == 100
    assert tr.numel() + va.numel() + te.numel() == 500
    assert info['stratified_by_matchup'] is True

    # Determinism check
    tr2, va2, te2, _ = split_by_game_id(game_ids, matchups, ratios=(0.8, 0.1, 0.1), seed=123)
    assert torch.equal(tr, tr2) and torch.equal(va, va2) and torch.equal(te, te2)


# ------------------------------------------------------------------ 4. Models, AAREN parity & metrics
def test_aaren_parallel_matches_sequential_step_and_ignores_padding(cpu):
    """Verify PieceAarenEncoder parallel prefix scan matches O(1) sequential step across variable
    lengths AND both AAREN and GRU have zero leakage from padded tail timesteps."""
    torch.manual_seed(0)
    enc_aaren = PieceAarenEncoder(event_dim=EVENT_DIM, hidden_size=24, num_layers=2).eval()
    enc_gru = PieceGruEncoder(event_dim=EVENT_DIM, hidden_size=24, num_layers=2).eval()

    lengths = torch.tensor([1, 2, 3, 5, 11, 24], dtype=torch.long)
    P_b, L_max = lengths.numel(), int(lengths.max())
    seq = torch.randn(P_b, L_max, EVENT_DIM)

    # 1. Parallel vs step-by-step O(1) equivalence for AAREN
    with torch.no_grad():
        par_out = enc_aaren(seq, lengths)
        seq_outs = []
        for i in range(P_b):
            st = None
            out_i = None
            for t in range(int(lengths[i])):
                out_i, st = enc_aaren.step(seq[i:i + 1, t, :], st)
            seq_outs.append(out_i[0])
        seq_out = torch.stack(seq_outs, dim=0)
    assert torch.allclose(par_out, seq_out, atol=1e-5), (par_out - seq_out).abs().max().item()

    # 2. Causal padding invariance: wildly perturb all padded slots t >= lengths[i]
    seq_dirty = seq.clone()
    for i in range(P_b):
        L = int(lengths[i])
        if L < L_max:
            seq_dirty[i, L:, :] = 1e4 * torch.randn(L_max - L, EVENT_DIM)

    with torch.no_grad():
        assert torch.allclose(enc_aaren(seq, lengths), enc_aaren(seq_dirty, lengths), atol=1e-6)
        assert torch.allclose(enc_gru(seq, lengths), enc_gru(seq_dirty, lengths), atol=1e-6)


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_all_four_models_forward_backward_and_empty_pieces(cpu, kind):
    """Verify forward/backward gradient flow for a_baseline, b_history, c1_aaren, c2_gru on both
    populated batches and zero-moved-piece (ply-0) batches."""
    torch.manual_seed(5)
    model = build_model(kind, channels=16, blocks=2, embed_dim=16, seq_layers=2).to(cpu)
    B = 4
    obs = torch.rand(B, NUM_OBS_PLANES, 10, 10)
    hist = torch.rand(B, NUM_HISTORY_PLANES, 10, 10)
    p_batch = torch.tensor([0, 0, 1, 3, 3, 3], dtype=torch.long)
    p_sq = torch.tensor([12, 25, 44, 3, 17, 55], dtype=torch.long)
    lens = torch.tensor([1, 4, 2, 7, 1, 5], dtype=torch.long)
    seq = torch.randn(p_batch.numel(), int(lens.max()), EVENT_DIM)

    opp = torch.zeros(B, NUM_SQUARES, dtype=torch.long)
    opp[:, :20] = torch.randint(1, NUM_RANKS + 1, (B, 20))
    hidden = opp > 0
    target = (opp - 1).clamp_min(0)

    logits = model(obs, hist_planes=hist, piece_batch=p_batch, piece_sq=p_sq, seq=seq, lengths=lens)
    assert logits.shape == (B, NUM_RANKS, NUM_SQUARES)
    assert torch.isfinite(logits).all()

    ce = F.cross_entropy(logits, target, reduction='none')
    loss = (ce * hidden).sum() / hidden.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0

    # Zero moved pieces (ply 0 edge case)
    with torch.no_grad():
        logits0 = model(
            obs,
            hist_planes=torch.zeros_like(hist),
            piece_batch=torch.zeros(0, dtype=torch.long),
            piece_sq=torch.zeros(0, dtype=torch.long),
            seq=torch.zeros((0, 1, EVENT_DIM)),
            lengths=torch.zeros(0, dtype=torch.long),
        )
    assert logits0.shape == (B, NUM_RANKS, NUM_SQUARES)
    assert torch.isfinite(logits0).all()


def test_binary_roc_auc_matches_sklearn_and_handles_edge_cases():
    rng = np.random.default_rng(0)
    # Perfect separation
    assert binary_roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    # Reversed
    assert binary_roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    # Degenerate single class -> None
    assert binary_roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])) is None
    assert binary_roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) is None
    # Random with heavy ties: exact parity with sklearn.metrics.roc_auc_score
    y = rng.integers(0, 2, size=500)
    scores = rng.choice([0.0, 0.1, 0.25, 0.5, 0.75, 1.0], size=500)
    assert binary_roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)


def test_labeled_setup_bank_and_end_to_end_tiny_probe_smoke(cpu, tmp_path):
    """End-to-end CPU smoke test: record 12 tiny games, verify setup kinds & CSR extraction,
    train all 4 tiny models + public baseline, and verify results.json + summary.md."""
    bank, kinds = labeled_diverse_setup_bank(50, seed=1)
    assert bank.shape == (50, 40)
    assert set(kinds.tolist()) == {SETUP_HEURISTIC, SETUP_STRUCTURED, SETUP_RANDOM}

    ds = record_dataset(
        policy=VecRandom(),
        experts=[VecExpert()],
        num_games=12,
        self_play_share=0.5,
        num_envs=4,
        device=cpu,
        seed=2,
        max_plies=90,
        no_battle_plies=50,
        bank_size=40,
        buckets=((4, 35, 2), (40, 75, 2)),
        log_every_games=0,
    )
    assert ds['meta']['games_finished'] == 12
    assert ds['obs'].size(0) >= 12
    assert ds['piece_ptr'][-1].item() == ds['piece_sq'].size(0)
    assert ds['event_ptr'][-1].item() == ds['events'].size(0)

    # Verify CSR batch extraction
    b_idx = torch.arange(min(4, ds['obs'].size(0)))
    pb, psq, seq, lens = extract_batch_sequences(
        ds['piece_ptr'], ds['piece_sq'], ds['event_ptr'], ds['event_len'], ds['events'], b_idx, device=cpu
    )
    assert pb.numel() == psq.numel() == seq.size(0) == lens.numel()
    if lens.numel() > 0:
        assert int(lens.min()) >= 1 and int(lens.max()) <= MAX_EVENTS

    # Run full probe pipeline for 1 epoch with tiny models
    out_dir = str(tmp_path / 'memory_probe_smoke')
    payload = run_probe(
        dataset=ds,
        out_dir=out_dir,
        device=cpu,
        epochs=1,
        batch_size=8,
        channels=16,
        blocks=1,
        embed_dim=16,
        seq_layers=1,
        seed=0,
        use_amp=False,
        save_checkpoints=False,
        verbose=False,
    )

    assert os.path.exists(os.path.join(out_dir, 'results.json'))
    assert os.path.exists(os.path.join(out_dir, 'summary.md'))

    with open(os.path.join(out_dir, 'results.json'), 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    for model_key in ('public_baseline', 'a_baseline', 'b_history', 'c1_aaren', 'c2_gru'):
        assert model_key in loaded['test_results']
        r = loaded['test_results'][model_key]
        assert 0.0 <= r['accuracy']['overall'] <= 1.0
        for auc_key in ('bomb_unmoved', 'marshal_or_general_moved', 'marshal_moved', 'general_moved'):
            assert auc_key in r['aucs']

    assert loaded['decision']['recommendation']
    for k in ('b_history', 'c1_aaren', 'c2_gru'):
        assert k in loaded['decision']['deltas_vs_a']
