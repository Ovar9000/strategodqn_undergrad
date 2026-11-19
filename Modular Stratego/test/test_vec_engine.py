"""The batched engine (vec/engine.py) against the original environment: same legal moves, same
observation planes, same game endings, over games of random moves."""

import random

import torch

from environment import StrategoEnvironment
from environment.action_encoding import decode_action, encode_move
from environment.heuristic_setup import HeuristicSetupAgent
from environment.observation import encode_observations
from environment.piece import PieceType as P
from vec.compat import placement_to_setup
from vec.eval_original import full_observation
from vec.engine import END_NO_MOVES, NUM_SQUARES, VecStratego

END_NAMES = {1: 'flag_capture', 2: 'no_moves', 3: 'no_battle_draw', 4: 'timeout'}


def replay_games(num_games: int, device, seed: int = 0, max_plies: int = 400, no_battle: int = 60,
                 attack_bias: float = 0.0):
    """Play `num_games` games of random moves on both engines; returns (plies compared, games
    finished, outlook mismatches tolerated). Raises AssertionError on the first difference."""
    rng = random.Random(seed)
    random.seed(seed)
    cpu = torch.device('cpu')
    setups = {1: HeuristicSetupAgent(player_id=1), -1: HeuristicSetupAgent(player_id=-1)}
    vec = VecStratego(num_games, device, max_plies=max_plies, no_battle_plies=no_battle)
    envs, states = [], []
    mover_setups, other_setups, starts = [], [], []
    for g in range(num_games):
        env = StrategoEnvironment(cpu, max_turns=max_plies, no_battle_draw_plies=no_battle)
        place = {p: setups[p].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(p))
                 for p in (1, -1)}
        start = 1 if g % 2 == 0 else -1
        states.append(env.reset(place[1], place[-1], starting_player=start))
        envs.append(env)
        mover_setups.append(placement_to_setup(place[start], start))
        other_setups.append(placement_to_setup(place[-start], -start))
        starts.append(start)
    vec.reset(torch.arange(num_games, device=vec.device), torch.stack(mover_setups), torch.stack(other_setups),
              torch.tensor(starts))

    plies = finished = tolerated = 0
    live = [True] * num_games
    while any(live):
        obs, _ = vec.observe()
        obs = obs.cpu()
        legal = vec.legal.cpu()
        actions = torch.zeros(num_games, dtype=torch.long)
        for g in range(num_games):
            if not live[g]:
                continue
            env, state = envs[g], states[g]
            player = state.current_player
            assert int(vec.side[g]) == player
            moves = env.get_valid_moves()
            want = sorted(encode_move(m, player) for m in moves)
            got = legal[g].nonzero().flatten().tolist()
            assert want == got, f"game {g} ply {plies}: legal moves differ"
            ref = full_observation([state], [player])[0]   # thesis planes + the extra ones, from the original state
            if not torch.allclose(ref, obs[g], atol=1e-5):
                diff = (ref - obs[g]).abs().amax((1, 2))
                bad = diff.gt(1e-5).nonzero().flatten().tolist()
                raise AssertionError(f"game {g}: observation planes {bad} differ")
            attacks = [m for m in moves if int(state.board[m[1][0], m[1][1]]) != 0]
            move = rng.choice(attacks) if attacks and rng.random() < attack_bias else rng.choice(moves)
            actions[g] = encode_move(move, player)
            assert decode_action(int(actions[g]), player) == move
            states[g], _, done, info = env.step(move)
            plies += 1
            if done:
                live[g] = False
        was_live = ~vec.done.cpu()
        done_now, result = vec.step(actions.to(vec.device))
        for g in range(num_games):
            if not was_live[g]:
                continue
            env = envs[g]
            assert bool(done_now[g]) == env.game_over, f"game {g}: done differs"
            if env.game_over:
                finished += 1
                assert int(vec.winner[g]) == (env.winner or 0), f"game {g}: winner differs"
                assert END_NAMES[int(vec.end_type[g])] == env.win_type, f"game {g}: ending differs"
    return plies, finished, tolerated


def test_random_games_match_the_original_environment(cpu):
    plies, finished, tolerated = replay_games(6, cpu, seed=1, max_plies=300, no_battle=60, attack_bias=0.3)
    assert finished == 6 and plies > 300 and tolerated == 0


def test_attacking_games_reach_flag_and_no_move_endings(cpu):
    plies, finished, tolerated = replay_games(6, cpu, seed=2, max_plies=1500, no_battle=200, attack_bias=0.9)
    assert finished == 6 and tolerated == 0


def _make_pair(cpu, p1_pieces, p2_pieces, start=1, max_plies=200, no_battle=200):
    """Build identical sparse positions in StrategoEnvironment and VecStratego."""
    env = StrategoEnvironment(cpu, max_turns=max_plies, no_battle_draw_plies=no_battle)
    env.board.reset()
    env._flag_positions = {1: None, -1: None}
    env.place_pieces(1, [(pt, pos) for pos, pt in p1_pieces.items()])
    env.place_pieces(-1, [(pt, pos) for pos, pt in p2_pieces.items()])
    env.current_player = start
    env._rep_moves = {1: [], -1: []}
    env._moves_cache_key = None

    vec = VecStratego(1, cpu, max_plies=max_plies, no_battle_plies=no_battle)
    vec.own.zero_()
    vec.opp.zero_()
    for f in (vec.own_rev, vec.opp_rev, vec.own_mov, vec.opp_mov, vec.own_long, vec.opp_long):
        f.zero_()
    for (r, c), pt in p1_pieces.items():
        if start == 1:
            vec.own[0, r * 10 + c] = pt.value
        else:
            vec.opp[0, (9 - r) * 10 + (9 - c)] = pt.value
    for (r, c), pt in p2_pieces.items():
        if start == 1:
            vec.opp[0, r * 10 + c] = pt.value
        else:
            vec.own[0, (9 - r) * 10 + (9 - c)] = pt.value
    vec.side[0] = start
    vec.ply[0] = 0
    vec.nb[0] = 0
    vec.last[0] = NUM_SQUARES
    vec.own_hist[0] = NUM_SQUARES
    vec.opp_hist[0] = NUM_SQUARES
    vec.done[0] = False
    vec.winner[0] = 0
    vec.end_type[0] = 0
    vec.legal[0] = vec._legal_mask(vec.own, vec.opp, vec.own_hist)[0]
    return env, vec


def _assert_legal_parity(env, vec):
    player = env.current_player
    assert int(vec.side[0]) == player
    want = sorted(encode_move(m, player) for m in env.get_valid_moves())
    got = vec.legal[0].nonzero().flatten().tolist()
    assert want == got, f"legal moves differ for player {player}"
    return set(env.get_valid_moves())


def _step_both(env, vec, move):
    player = env.current_player
    legal_set = _assert_legal_parity(env, vec)
    assert move in legal_set, f"move {move} is not legal for player {player}"
    act = encode_move(move, player)
    _, _, done, _ = env.step(move)
    done_now, _ = vec.step(torch.tensor([act], dtype=torch.long))
    assert bool(done_now[0]) == done
    if done:
        assert int(vec.winner[0]) == (env.winner or 0)
        assert END_NAMES[int(vec.end_type[0])] == env.win_type
    return done


def test_two_square_rule_masking_and_reset_parity(cpu):
    """After A->B and B->A on two consecutive turns of a side, A->B is masked out in both
    StrategoEnvironment and VecStratego for both P1 and P2, and resets after a non-reversing move."""
    for start in (1, -1):
        env, vec = _make_pair(
            cpu,
            p1_pieces={(9, 0): P.FLAG, (7, 0): P.SERGEANT, (9, 2): P.MAJOR},
            p2_pieces={(0, 9): P.FLAG, (2, 9): P.SERGEANT, (0, 7): P.MAJOR},
            start=start,
        )
        p1_ab, p1_ba, p1_ac = ((7, 0), (6, 0)), ((6, 0), (7, 0)), ((7, 0), (7, 1))
        p2_xy, p2_yx, p2_alt = ((2, 9), (3, 9)), ((3, 9), (2, 9)), ((0, 7), (1, 7))
        seq = [p1_ab, p2_xy, p1_ba, p2_yx] if start == 1 else [p2_xy, p1_ab, p2_yx, p1_ba]
        for m in seq:
            assert not _step_both(env, vec, m)

        # First mover's 3rd turn: repeating A->B (or X->Y) is forbidden in both engines
        first_forb = p1_ab if start == 1 else p2_xy
        first_alt = p1_ac if start == 1 else p2_alt
        moves = _assert_legal_parity(env, vec)
        assert first_forb not in moves
        assert not bool(vec.legal[0, encode_move(first_forb, start)])
        assert first_alt in moves
        assert not _step_both(env, vec, first_alt)

        # Second mover's 3rd turn: repeating its own 2-square move is also forbidden (tests P2 canonical coords)
        second_forb = p2_xy if start == 1 else p1_ab
        second_alt = p2_alt if start == 1 else p1_ac
        moves = _assert_legal_parity(env, vec)
        assert second_forb not in moves
        assert not bool(vec.legal[0, encode_move(second_forb, -start)])
        assert second_alt in moves
        assert not _step_both(env, vec, second_alt)

        # First mover's 4th turn: non-reversing move cleared the two-square restriction
        moves = _assert_legal_parity(env, vec)
        if start == 1:
            assert ((7, 1), (7, 0)) in moves
        else:
            assert ((2, 9), (3, 9)) in moves


def test_two_square_rule_resets_on_battle_parity(cpu):
    """Any battle clears both sides' two-square repetition trackers in both engines."""
    env, vec = _make_pair(
        cpu,
        p1_pieces={(9, 0): P.FLAG, (7, 0): P.SERGEANT, (6, 5): P.MARSHAL},
        p2_pieces={(0, 9): P.FLAG, (2, 9): P.SERGEANT, (5, 5): P.MINER},
        start=1,
    )
    assert not _step_both(env, vec, ((7, 0), (6, 0)))
    assert not _step_both(env, vec, ((2, 9), (3, 9)))
    assert not _step_both(env, vec, ((6, 0), (7, 0)))
    # P2 attacks P1 Marshal with Miner -> battle!
    assert not _step_both(env, vec, ((5, 5), (6, 5)))
    moves = _assert_legal_parity(env, vec)
    assert ((7, 0), (6, 0)) in moves


def test_two_square_rule_trapped_piece_loses_by_no_moves_parity(cpu):
    """If a player's only legal move is the forbidden two-square repetition, they lose by no_moves."""
    env, vec = _make_pair(
        cpu,
        p1_pieces={
            (9, 0): P.FLAG,
            (8, 0): P.BOMB,
            (7, 1): P.BOMB,
            (8, 2): P.BOMB,
            (9, 2): P.BOMB,
            (9, 1): P.SERGEANT,
        },
        p2_pieces={(0, 9): P.FLAG, (2, 9): P.SCOUT},
        start=1,
    )
    assert not _step_both(env, vec, ((9, 1), (8, 1)))
    assert not _step_both(env, vec, ((2, 9), (2, 8)))
    assert not _step_both(env, vec, ((8, 1), (9, 1)))
    # After P2's quiet reply, P1's only move ((9,1)->(8,1)) is forbidden -> P2 wins by no_moves.
    done = _step_both(env, vec, ((2, 8), (2, 7)))
    assert done
    assert env.winner == -1 and env.win_type == 'no_moves'
    assert int(vec.winner[0]) == -1 and int(vec.end_type[0]) == END_NO_MOVES


def test_two_square_rule_scout_long_move_only_masks_exact_target(cpu):
    """A Scout that reversed A<->B is blocked from A->B, but can still stop short or slide past B."""
    env, vec = _make_pair(
        cpu,
        p1_pieces={(9, 0): P.FLAG, (7, 0): P.SCOUT},
        p2_pieces={(0, 9): P.FLAG, (2, 9): P.SCOUT},
        start=1,
    )
    assert not _step_both(env, vec, ((7, 0), (7, 5)))
    assert not _step_both(env, vec, ((2, 9), (2, 8)))
    assert not _step_both(env, vec, ((7, 5), (7, 0)))
    assert not _step_both(env, vec, ((2, 8), (2, 7)))
    moves = _assert_legal_parity(env, vec)
    assert ((7, 0), (7, 5)) not in moves
    assert ((7, 0), (7, 4)) in moves
    assert ((7, 0), (7, 6)) in moves


def test_is_position_repetitive_compares_same_player_moves(cpu):
    """_is_position_repetitive compares i with i-2 (same player), not i-1 (alternating players)."""
    env = StrategoEnvironment(cpu)
    # Two same-player reversals in 6 plies (P1 reverses at i=2, P2 reverses at i=3) -> True
    env.move_history = [
        ((7, 0), (6, 0)), ((2, 9), (3, 9)),
        ((6, 0), (7, 0)), ((3, 9), (2, 9)),
        ((7, 0), (7, 1)), ((2, 9), (2, 8)),
    ]
    assert env._is_position_repetitive() is True

    # Only one same-player reversal in 6 plies -> False
    env.move_history = [
        ((7, 0), (6, 0)), ((2, 9), (3, 9)),
        ((6, 0), (7, 0)), ((3, 9), (3, 8)),
        ((7, 0), (7, 1)), ((3, 8), (3, 7)),
    ]
    assert env._is_position_repetitive() is False

    # Alternating players touching reverse squares (i vs i-1), no same-player reversals -> False
    env.move_history = [
        ((5, 0), (4, 0)), ((4, 0), (5, 0)),
        ((6, 1), (5, 1)), ((5, 1), (6, 1)),
        ((7, 2), (6, 2)), ((6, 2), (7, 2)),
    ]
    assert env._is_position_repetitive() is False
