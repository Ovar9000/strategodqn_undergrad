"""
Batched game recorder for the A2 supervised memory probe.

Plays `ppo_4` self-play and `ppo_4` vs experts (`expert`, `expert2`, `expert3`) on
`VecStratego` using explicitly labeled diverse setups (`heuristic`, `structured`, `random`),
tracking persistent piece identities and public event histories OUTSIDE `vec/engine.py`.

Usage:
    python -m vec.memory.recorder --net ../runs/ppo_4/net_final.pt \
        --expert-styles ../runs/ppo_2/expert_styles.json \
        --games 20000 --out ../runs/memory_probe/dataset.pt
"""

import argparse
import gc
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from vec.compat import heuristic_setup_bank
from vec.engine import NUM_OBS_PLANES, VecStratego
from vec.expert import Style, VecExpert
from vec.net import NetPolicy, load
from vec.setups import random_setup, structured_setup

from vec.memory.features import (
    EVENT_DIM,
    MATCHUP_KIND_NAMES,
    MAX_EVENTS,
    NUM_HISTORY_PLANES,
    SETUP_KIND_NAMES,
    PieceHistoryTracker,
)

SETUP_HEURISTIC, SETUP_STRUCTURED, SETUP_RANDOM = 0, 1, 2
DEFAULT_BUCKETS = (
    (0, 39, 2),      # early (ply < 40)
    (40, 99, 2),     # mid   (40 <= ply < 100)
    (100, 199, 2),   # late  (100 <= ply < 200)
    (200, 349, 1),   # deep late (ply >= 200)
)


def quantize_obs(obs: torch.Tensor) -> torch.Tensor:
    return (obs * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)


def dequantize_obs(obs_u8: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return obs_u8.to(dtype).div_(255.0)


def labeled_diverse_setup_bank(
    count: int,
    seed: int = 0,
    shares: Tuple[float, float, float] = (0.4, 0.45, 0.15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate `(bank, kinds)` with `(count, 40)` setups and `(count,)` setup-kind labels.

    Labels: 0 = heuristic, 1 = structured, 2 = random. Matches `vec.setups.diverse_setup_bank`
    composition and shuffling while preserving exact setup-kind provenance.
    """
    rng = random.Random(seed)
    n_h = int(count * shares[0])
    n_s = int(count * shares[1])
    n_r = count - n_h - n_s

    parts_bank = []
    parts_kind = []
    if n_h > 0:
        parts_bank.append(heuristic_setup_bank(n_h, seed=seed))
        parts_kind.append(torch.full((n_h,), SETUP_HEURISTIC, dtype=torch.long))
    if n_s > 0:
        parts_bank.append(torch.stack([structured_setup(rng) for _ in range(n_s)]))
        parts_kind.append(torch.full((n_s,), SETUP_STRUCTURED, dtype=torch.long))
    if n_r > 0:
        parts_bank.append(torch.stack([random_setup(rng) for _ in range(n_r)]))
        parts_kind.append(torch.full((n_r,), SETUP_RANDOM, dtype=torch.long))

    bank = torch.cat(parts_bank, dim=0)
    kinds = torch.cat(parts_kind, dim=0)
    assert bool(((bank > 0).sum(1) == 40).all())
    perm = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return bank[perm], kinds[perm]


def load_experts(styles_path: Optional[str] = None) -> List[VecExpert]:
    """Build `[expert, expert2, expert3, ...]` matching `vec/ppo.py`."""
    experts = [VecExpert(name='expert')]
    if styles_path:
        with open(styles_path, 'r', encoding='utf-8') as f:
            styles = json.load(f)
        for i, d in enumerate(styles):
            experts.append(VecExpert(style=Style(**d), name=f'expert{i + 2}'))
    return experts


def _sample_target_plies(
    rng: random.Random,
    buckets: Sequence[Tuple[int, int, int]],
    parity: Optional[int],
    max_plies: int,
    width: int,
) -> List[int]:
    chosen: List[int] = []
    for lo, hi, k in buckets:
        hi_eff = min(hi, max_plies - 1)
        if lo > hi_eff or k <= 0:
            continue
        if parity is None:
            cands = list(range(lo, hi_eff + 1))
        else:
            first = lo if (lo % 2 == parity) else lo + 1
            cands = list(range(first, hi_eff + 1, 2))
        if not cands:
            continue
        if len(cands) <= k:
            chosen.extend(cands)
        else:
            chosen.extend(rng.sample(cands, k))
    while len(chosen) < width:
        chosen.append(-1)
    return chosen[:width]


@torch.no_grad()
def record_dataset(
    policy,
    experts: Sequence,
    num_games: int = 20000,
    self_play_share: float = 0.5,
    num_envs: int = 512,
    device=None,
    seed: int = 0,
    max_plies: int = 1500,
    no_battle_plies: int = 200,
    bank_size: Optional[int] = None,
    buckets: Sequence[Tuple[int, int, int]] = DEFAULT_BUCKETS,
    max_events: int = MAX_EVENTS,
    log_every_games: int = 2000,
    net_path: str = '',
    expert_styles_path: str = '',
) -> Dict[str, object]:
    """Record a compact game-indexed dataset of Stratego plies with tracked enemy piece histories."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    torch.manual_seed(seed)
    rng = random.Random(seed + 1337)

    if num_games <= 0:
        raise ValueError("num_games must be positive")
    n_envs = min(num_envs, num_games)
    b_size = max(2000, num_games) if bank_size is None else bank_size
    bank, bank_kinds = labeled_diverse_setup_bank(b_size, seed=seed)
    bank = bank.to(dev)
    bank_kinds = bank_kinds.to(dev)

    # Plan deterministic matchup assignment per game_id: 0 = self-play, 1..len(experts) = vs expert_k
    n_self = int(round(num_games * self_play_share)) if len(experts) > 0 else num_games
    n_exp = num_games - n_self
    game_matchups: List[int] = [0] * n_self
    if n_exp > 0:
        for i in range(n_exp):
            game_matchups.append(1 + (i % len(experts)))
    rng.shuffle(game_matchups)

    env = VecStratego(n_envs, dev, max_plies=max_plies, no_battle_plies=no_battle_plies)
    tracker = PieceHistoryTracker(n_envs, dev, max_events=max_events)
    target_width = max(sum(k for _, _, k in buckets), 1)

    slot_active = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    slot_game_id = torch.full((n_envs,), -1, dtype=torch.long, device=dev)
    slot_matchup = torch.zeros(n_envs, dtype=torch.long, device=dev)
    slot_policy_side = torch.ones(n_envs, dtype=torch.long, device=dev)
    slot_setup_kind = torch.zeros((n_envs, 2), dtype=torch.long, device=dev)
    slot_targets = torch.full((n_envs, target_width), -1, dtype=torch.long, device=dev)

    next_game_id = 0
    games_finished = 0
    next_log_milestone = log_every_games
    t0 = time.time()

    def deal_slots(slots: torch.Tensor):
        nonlocal next_game_id
        slot_list = slots.tolist()
        assign_slots = []
        assign_gids = []
        for s in slot_list:
            if next_game_id < num_games:
                assign_slots.append(s)
                assign_gids.append(next_game_id)
                next_game_id += 1
            else:
                slot_active[s] = False
        if not assign_slots:
            return

        m = len(assign_slots)
        idx = torch.tensor(assign_slots, dtype=torch.long, device=dev)
        b_idx_0 = torch.randint(0, bank.size(0), (m,), device=dev)
        b_idx_1 = torch.randint(0, bank.size(0), (m,), device=dev)
        start_side = torch.tensor([rng.choice((-1, 1)) for _ in range(m)], dtype=torch.long, device=dev)
        pol_side = torch.tensor([rng.choice((-1, 1)) for _ in range(m)], dtype=torch.long, device=dev)
        matchups = torch.tensor([game_matchups[g] for g in assign_gids], dtype=torch.long, device=dev)

        s0, s1 = bank[b_idx_0], bank[b_idx_1]
        k0, k1 = bank_kinds[b_idx_0], bank_kinds[b_idx_1]
        p1_starts = (start_side == 1)[:, None]
        mover_setup = torch.where(p1_starts, s0, s1)
        other_setup = torch.where(p1_starts, s1, s0)

        env.reset(idx, mover_setup, other_setup, start_side)
        tracker.reset_from_env(env, idx)

        slot_active[idx] = True
        slot_game_id[idx] = torch.tensor(assign_gids, dtype=torch.long, device=dev)
        slot_matchup[idx] = matchups
        slot_policy_side[idx] = pol_side
        slot_setup_kind[idx, 0] = k0
        slot_setup_kind[idx, 1] = k1

        targets_rows = []
        for i in range(m):
            mk = int(matchups[i])
            if mk == 0:
                parity = None
            else:
                parity = 0 if int(pol_side[i]) == int(start_side[i]) else 1
            targets_rows.append(_sample_target_plies(rng, buckets, parity, max_plies, target_width))
        slot_targets[idx] = torch.tensor(targets_rows, dtype=torch.long, device=dev)

    deal_slots(torch.arange(n_envs, device=dev))

    # Chunk buffers on CPU
    c_obs: List[torch.Tensor] = []
    c_opp: List[torch.Tensor] = []
    c_opp_rev: List[torch.Tensor] = []
    c_opp_mov: List[torch.Tensor] = []
    c_opp_long: List[torch.Tensor] = []
    c_ply: List[torch.Tensor] = []
    c_setup_kind: List[torch.Tensor] = []
    c_game_id: List[torch.Tensor] = []
    c_matchup: List[torch.Tensor] = []
    c_piece_counts: List[torch.Tensor] = []
    c_piece_sq: List[torch.Tensor] = []
    c_piece_summary: List[torch.Tensor] = []
    c_event_len: List[torch.Tensor] = []
    c_events: List[torch.Tensor] = []

    actions = torch.zeros(n_envs, dtype=torch.long, device=dev)
    samples_collected = 0

    while bool(slot_active.any()):
        live = (~env.done) & slot_active
        is_pol = live & ((slot_matchup == 0) | (env.side == slot_policy_side))

        # 1. Sample positions that hit their pre-selected target plies
        hit_ply = (env.ply[:, None] == slot_targets).any(dim=1)
        has_hidden = ((env.opp > 0) & ~env.opp_rev).any(dim=1)
        to_sample = is_pol & hit_ply & has_hidden
        if to_sample.any():
            s_idx = to_sample.nonzero().flatten()
            obs_s, _ = env.observe(s_idx)
            snap = tracker.extract_moved_enemy_snapshot(env, s_idx)
            enemy_s = 1 - (env.side[s_idx] < 0).long()

            c_obs.append(quantize_obs(obs_s).cpu())
            c_opp.append(env.opp[s_idx].to(torch.uint8).cpu())
            c_opp_rev.append(env.opp_rev[s_idx].cpu())
            c_opp_mov.append(env.opp_mov[s_idx].cpu())
            c_opp_long.append(env.opp_long[s_idx].cpu())
            c_ply.append(env.ply[s_idx].to(torch.int16).cpu())
            c_setup_kind.append(slot_setup_kind[s_idx, enemy_s].to(torch.uint8).cpu())
            c_game_id.append(slot_game_id[s_idx].to(torch.int32).cpu())
            c_matchup.append(slot_matchup[s_idx].to(torch.uint8).cpu())

            c_piece_counts.append(snap['piece_counts'].cpu())
            c_piece_sq.append(snap['piece_sq'].cpu())
            c_piece_summary.append(snap['piece_summary'].cpu())
            c_event_len.append(snap['event_len'].cpu())
            c_events.append(snap['events'].cpu())
            samples_collected += int(s_idx.numel())

        # 2. Choose actions for policy and experts
        ip = is_pol.nonzero().flatten()
        if ip.numel():
            actions[ip] = policy.act(env, ip)
        for e_i, ex in enumerate(experts):
            ie = (live & (~is_pol) & (slot_matchup == (e_i + 1))).nonzero().flatten()
            if ie.numel():
                actions[ie] = ex.act(env, ie)

        # 3. Step engine + external tracker
        done_now, _ = tracker.step(env, actions)

        finished_active = done_now & slot_active
        if finished_active.any():
            fin_slots = finished_active.nonzero().flatten()
            games_finished += int(fin_slots.numel())
            deal_slots(fin_slots)
            if log_every_games > 0 and games_finished >= next_log_milestone:
                dt = time.time() - t0
                print(
                    f"[recorder] games={games_finished}/{num_games} samples={samples_collected} "
                    f"elapsed={dt:.1f}s ({games_finished / max(dt, 1e-6):.1f} games/s)",
                    flush=True,
                )
                while next_log_milestone <= games_finished:
                    next_log_milestone += log_every_games

    # Concatenate one field at a time to keep peak RAM low
    def _cat_and_clear(chunks: List[torch.Tensor], empty_shape, dtype) -> torch.Tensor:
        if not chunks:
            return torch.zeros(empty_shape, dtype=dtype)
        out = torch.cat(chunks, dim=0)
        chunks.clear()
        return out

    obs = _cat_and_clear(c_obs, (0, NUM_OBS_PLANES, 10, 10), torch.uint8)
    gc.collect()
    opp = _cat_and_clear(c_opp, (0, 100), torch.uint8)
    opp_rev = _cat_and_clear(c_opp_rev, (0, 100), torch.bool)
    opp_mov = _cat_and_clear(c_opp_mov, (0, 100), torch.bool)
    opp_long = _cat_and_clear(c_opp_long, (0, 100), torch.bool)
    ply = _cat_and_clear(c_ply, (0,), torch.int16)
    setup_kind = _cat_and_clear(c_setup_kind, (0,), torch.uint8)
    game_id = _cat_and_clear(c_game_id, (0,), torch.int32)
    matchup_kind = _cat_and_clear(c_matchup, (0,), torch.uint8)

    piece_counts = _cat_and_clear(c_piece_counts, (0,), torch.long)
    M = int(obs.size(0))
    piece_ptr = torch.zeros(M + 1, dtype=torch.long)
    if M > 0:
        piece_ptr[1:] = piece_counts.cumsum(0)
    del piece_counts

    piece_sq = _cat_and_clear(c_piece_sq, (0,), torch.uint8)
    piece_summary = _cat_and_clear(c_piece_summary, (0, NUM_HISTORY_PLANES), torch.float16)
    event_len = _cat_and_clear(c_event_len, (0,), torch.uint8)
    P = int(event_len.size(0))
    event_ptr = torch.zeros(P + 1, dtype=torch.long)
    if P > 0:
        event_ptr[1:] = event_len.long().cumsum(0)
    events = _cat_and_clear(c_events, (0, EVENT_DIM), torch.float16)
    gc.collect()

    elapsed = time.time() - t0
    ply_l = ply.long()
    meta = dict(
        num_games=int(num_games),
        games_finished=int(games_finished),
        unique_games_sampled=int(torch.unique(game_id).numel()) if M > 0 else 0,
        num_samples=M,
        num_moved_pieces=P,
        num_events=int(events.size(0)),
        self_play_games=int(n_self),
        vs_expert_games=int(n_exp),
        num_experts=len(experts),
        seed=int(seed),
        max_events=int(max_events),
        event_dim=EVENT_DIM,
        num_history_planes=NUM_HISTORY_PLANES,
        net_path=net_path,
        expert_styles_path=expert_styles_path,
        elapsed_sec=round(elapsed, 2),
        setup_kind_samples={
            name: int((setup_kind == i).sum()) for i, name in enumerate(SETUP_KIND_NAMES)
        },
        phase_samples=dict(
            early=int((ply_l < 40).sum()),
            mid=int(((ply_l >= 40) & (ply_l < 100)).sum()),
            late=int((ply_l >= 100).sum()),
        ),
        matchup_samples={
            MATCHUP_KIND_NAMES[i] if i < len(MATCHUP_KIND_NAMES) else f'matchup_{i}': int((matchup_kind == i).sum())
            for i in range(1 + len(experts))
        },
    )

    return dict(
        obs=obs,
        opp=opp,
        opp_rev=opp_rev,
        opp_mov=opp_mov,
        opp_long=opp_long,
        ply=ply,
        setup_kind=setup_kind,
        game_id=game_id,
        matchup_kind=matchup_kind,
        piece_ptr=piece_ptr,
        piece_sq=piece_sq,
        piece_summary=piece_summary,
        event_ptr=event_ptr,
        event_len=event_len,
        events=events,
        meta=meta,
    )


def save_dataset(dataset: Dict[str, object], path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(dataset, path)


def load_dataset(path: str) -> Dict[str, object]:
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    ap = argparse.ArgumentParser(description='Record Stratego games with external enemy piece histories.')
    ap.add_argument('--net', required=True, help='Path to PolicyValueNet checkpoint (e.g. ../runs/ppo_4/net_final.pt)')
    ap.add_argument('--expert-styles', default='', help='Path to expert_styles.json (e.g. ../runs/ppo_2/expert_styles.json)')
    ap.add_argument('--out', required=True, help='Output dataset path (.pt)')
    ap.add_argument('--games', type=int, default=20000)
    ap.add_argument('--self-play-share', type=float, default=0.5)
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--log-every', type=int, default=2000)
    args = ap.parse_args()

    dev = torch.device(args.device)
    net = load(args.net, dev)
    policy = NetPolicy(net, temperature=args.temperature, name='ppo_4')
    experts = load_experts(args.expert_styles or None)

    ds = record_dataset(
        policy=policy,
        experts=experts,
        num_games=args.games,
        self_play_share=args.self_play_share,
        num_envs=args.envs,
        device=dev,
        seed=args.seed,
        log_every_games=args.log_every,
        net_path=args.net,
        expert_styles_path=args.expert_styles,
    )
    save_dataset(ds, args.out)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[recorder] saved {args.out} ({size_mb:.1f} MB): {json.dumps(ds['meta'])}", flush=True)


if __name__ == '__main__':
    main()
