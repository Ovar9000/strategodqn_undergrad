"""
Record Stratego games with full per-ply policy top-5 probabilities, value estimates,
belief predictions on hidden enemy pieces, battle reveals, and sure-loss blunder flags.

Saves one JSON file per game in a unified schema shared with:
  - Modular Stratego/web/replay.html  (interactive visual replay & blunder inspector)
  - Modular Stratego/web/play.html    ("Download game" exports the exact same JSON schema)

Usage examples (from Modular Stratego/):
    # Record 3 games of ppo_4 vs expert, filtering for R10 non-Scout sure-loss blunders
    python -m vec.record_games --net ../runs/ppo_4/net_final.pt --opponent expert \
        --games 2 --filter blunder --pool 64 --out ../runs/replays --seed 10

    # Record a curated suite of ppo_4 games (blunders, wins, losses, expert3, self-play)
    python -m vec.record_games --net ../runs/ppo_4/net_final.pt --suite --out ../runs/replays

Replay JSON Schema (schema_version = 1):
    All board squares are absolute 0..99 (row-major sq = r*10 + c, r=0..9, c=0..9),
    with Red (+1, Player 1) starting on rows 6..9 (squares 60..99) and Blue (-1, Player 2)
    starting on rows 0..3 (squares 0..39). Piece ranks are 1..12 (Flag=1, Spy=2, Scout=3,
    Miner=4, Sergeant=5, Lieutenant=6, Captain=7, Major=8, Colonel=9, General=10,
    Marshal=11, Bomb=12).

    {
      "schema_version": 1,
      "meta": {
        "source": "record_games" | "play_html",
        "created_at": ISO-8601 UTC string,
        "model": str,
        "opponent": str,
        "red": str,
        "blue": str,
        "net_side": 1 | -1 | 0 (0 = self-play / both sides are the net),
        "starting_side": 1 | -1,
        "temperature": float | null,
        "seed": int | null,
        "max_plies": int,
        "no_battle_plies": int,
        "winner": 1 | -1 | 0 | null,
        "end_type": "flag" | "no_moves" | "no_battle" | "turn_limit" | "in_progress",
        "num_plies": int,
        "blunders": {
          "sure_loss_total": int,
          "sure_loss_non_scout": int,
          "sure_loss_plies": [int, ...],
          "sure_loss_all_plies": [int, ...]
        }
      },
      "setups": {
        "red":  [int x 40],   # ranks for absolute squares 60..99
        "blue": [int x 40]    # ranks for absolute squares 0..39
      },
      "plies": [
        {
          "ply": int,
          "side": 1 | -1,
          "move": [src_sq, dst_sq],
          "piece": int (1..12),
          "battle": null | {
            "attacker": int,
            "defender": int,
            "result": 1 | -1 | 0,
            "defender_was_revealed": bool,
            "attacker_was_revealed": bool,
            "reveals": [{"sq": int, "side": 1 | -1, "rank": int}, ...]
          },
          "top5": null | [{"move": [src, dst], "prob": float}, ...],
          "played_prob": null | float,
          "value": null | float,       # mover's POV in [-1, 1]
          "belief": null | {
            "<abs_sq>": {
              "top_rank": int (1..12),
              "top_prob": float,
              "probs": [float x 12],   # ranks 1..12
              "true_rank": int (1..12)
            }
          },
          "sure_loss": bool,
          "sure_loss_non_scout": bool
        }
      ]
    }
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vec.arena import END_NAMES
from vec.engine import BOMB, FLAG, LAKES, NUM_SQUARES, SCOUT, VecStratego, tables
from vec.expert import Style, VecExpert, VecRandom
from vec.net import NetPolicy, PolicyValueNet, load as load_net, masked_logits
from vec.setups import diverse_setup_bank

SCHEMA_VERSION = 1

DEFAULT_EXPERT_STYLES: List[Dict[str, float]] = [
    {'pursuit': 2.0, 'repel': 0.5, 'probe': 1.5},
    {
        'pursuit': 1.721, 'repel': 0.5369, 'probe': 0.6253, 'attack_chance': 0.3724,
        'clock': 0.4912, 'advance': 0.0331, 'pull': 0.7336, 'power': 1.7568,
    },
]

LAKE_SQUARES = frozenset(r * 10 + c for r, c in LAKES)


def _model_label(path: str) -> str:
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    parent = os.path.basename(os.path.dirname(norm))
    if parent and base.startswith('net'):
        return parent
    return os.path.splitext(base)[0] or path


def _load_expert_styles(styles_path: Optional[str]) -> List[Dict[str, float]]:
    if styles_path and os.path.exists(styles_path):
        with open(styles_path, encoding='utf-8') as f:
            return json.load(f)
    default_path = os.path.join('..', 'runs', 'ppo_2', 'expert_styles.json')
    if os.path.exists(default_path):
        with open(default_path, encoding='utf-8') as f:
            return json.load(f)
    return list(DEFAULT_EXPERT_STYLES)


def make_opponent(
    spec: str,
    device: torch.device,
    temperature: float = 1.0,
    expert_styles: Optional[str] = None,
) -> Tuple[Any, str]:
    """Build an opponent policy and display label from a CLI spec."""
    if spec == 'expert':
        return VecExpert(), 'expert'
    if spec == 'expert2':
        styles = _load_expert_styles(expert_styles)
        return VecExpert(style=Style(**styles[0]), name='expert2'), 'expert2'
    if spec == 'expert3':
        styles = _load_expert_styles(expert_styles)
        idx = 1 if len(styles) > 1 else 0
        return VecExpert(style=Style(**styles[idx]), name='expert3'), 'expert3'
    if spec == 'random':
        return VecRandom(), 'random'
    if spec == 'self':
        return None, 'self'
    if spec.endswith('.json'):
        with open(spec, encoding='utf-8') as f:
            data = json.load(f)
        style_dict = data['champion'] if 'champion' in data else data
        return VecExpert(style=Style(**style_dict)), os.path.splitext(os.path.basename(spec))[0]
    opp_net = load_net(spec, device).eval()
    return NetPolicy(opp_net, temperature=temperature, name=_model_label(spec)), _model_label(spec)


def _canon_to_abs(sq: int, side: int) -> int:
    return sq if side == 1 else (NUM_SQUARES - 1 - sq)


def _select_sides(num_games: int, net_side_mode: str, device: torch.device) -> torch.Tensor:
    if net_side_mode == 'red':
        return torch.ones(num_games, dtype=torch.long, device=device)
    if net_side_mode == 'blue':
        return -torch.ones(num_games, dtype=torch.long, device=device)
    idx = torch.arange(num_games, device=device)
    return torch.where(idx % 2 == 0, 1, -1)


def _extract_initial_setups(
    env: VecStratego,
    start_side: torch.Tensor,
) -> List[Dict[str, List[int]]]:
    """Read absolute Red (60..99) and Blue (0..39) setups immediately after env.reset()."""
    out: List[Dict[str, List[int]]] = []
    own_cpu = env.own.cpu()
    opp_cpu = env.opp.cpu()
    ss_cpu = start_side.cpu()
    for g in range(env.n):
        if int(ss_cpu[g]) == 1:
            red = own_cpu[g, 60:100].tolist()
            blue = opp_cpu[g, 0:40].tolist()
        else:
            red = opp_cpu[g, 0:40].flip(0).tolist()
            blue = own_cpu[g, 60:100].flip(0).tolist()
        out.append({'red': [int(x) for x in red], 'blue': [int(x) for x in blue]})
    return out


@torch.no_grad()
def _search_pool(
    net: PolicyValueNet,
    opp_policy: Any,
    is_self_play: bool,
    pool_size: int,
    bank: torch.Tensor,
    device: torch.device,
    temperature: float,
    seed: int,
    max_plies: int,
    no_battle_plies: int,
    net_side_mode: str,
    amp: bool,
) -> Dict[str, Any]:
    """Fast batched rollout storing actions & blunder/result stats to pick filtered games."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    env = VecStratego(pool_size, device, max_plies=max_plies, no_battle_plies=no_battle_plies)
    t = env.t
    idx_all = torch.arange(pool_size, device=device)
    a_side = _select_sides(pool_size, net_side_mode, device)
    start_side = (torch.randint(0, 2, (pool_size,), generator=gen) * 2 - 1).to(device)
    m_idx = torch.randint(0, bank.size(0), (pool_size,), generator=gen)
    o_idx = torch.randint(0, bank.size(0), (pool_size,), generator=gen)
    mover_setup = bank[m_idx].to(device)
    other_setup = bank[o_idx].to(device)
    env.reset(idx_all, mover_setup, other_setup, start_side)

    net.eval()
    use_amp = amp and device.type == 'cuda'
    actions_seq: List[torch.Tensor] = []
    sl_non_scout = torch.zeros(pool_size, dtype=torch.long, device=device)
    sl_total = torch.zeros(pool_size, dtype=torch.long, device=device)
    actions = torch.zeros(pool_size, dtype=torch.long, device=device)

    while not bool(env.done.all()):
        live = ~env.done
        is_net = live if is_self_play else (live & (env.side == a_side))
        in_idx = is_net.nonzero().flatten()
        io_idx = (live & ~is_net).nonzero().flatten()

        if in_idx.numel():
            obs, _ = env.observe(in_idx)
            legal = env.legal[in_idx]
            with torch.autocast('cuda', enabled=use_amp):
                logits, _ = net(obs.contiguous(memory_format=torch.channels_last))
            ml = masked_logits(logits, legal)
            if temperature > 0:
                probs = torch.softmax(ml / temperature, dim=1)
                act_net = torch.multinomial(probs, 1).squeeze(1)
            else:
                act_net = ml.argmax(1)
            actions[in_idx] = act_net
        if io_idx.numel():
            actions[io_idx] = opp_policy.act(env, io_idx)

        src = t.act_from[actions]
        dst = t.act_to[actions].clamp(max=NUM_SQUARES - 1)
        a_rank = env.own[idx_all, src]
        d_rank = env.opp[idx_all, dst]
        battle = live & (d_rank > 0)
        res = t.battle[a_rank, d_rank]
        sl = battle & is_net & env.opp_rev[idx_all, dst] & (res == -1)
        sl_total += sl.long()
        sl_non_scout += (sl & (a_rank != SCOUT)).long()

        actions_seq.append(actions.cpu().clone())
        env.step(actions)

    res_net = (env.winner * a_side).cpu()
    return {
        'mover_setup': mover_setup.cpu(),
        'other_setup': other_setup.cpu(),
        'start_side': start_side.cpu(),
        'a_side': a_side.cpu(),
        'actions_seq': actions_seq,
        'sl_non_scout': sl_non_scout.cpu(),
        'sl_total': sl_total.cpu(),
        'res_net': res_net,
        'winner': env.winner.cpu(),
        'end_type': env.end_type.cpu(),
        'ply': env.ply.cpu(),
    }


def _match_filter(filter_kind: str, sl_ns: int, res_net: int) -> bool:
    if filter_kind == 'any':
        return True
    if filter_kind == 'blunder':
        return sl_ns > 0
    if filter_kind == 'blunder_loss':
        return sl_ns > 0 and res_net == -1
    if filter_kind == 'win':
        return res_net == 1
    if filter_kind == 'loss':
        return res_net == -1
    if filter_kind == 'draw':
        return res_net == 0
    raise ValueError(f'Unknown filter_kind: {filter_kind!r}')


@torch.no_grad()
def _record_batch(
    net: PolicyValueNet,
    opp_policy: Any,
    is_self_play: bool,
    mover_setup: torch.Tensor,
    other_setup: torch.Tensor,
    start_side: torch.Tensor,
    a_side: torch.Tensor,
    device: torch.device,
    temperature: float,
    seed: int,
    max_plies: int,
    no_battle_plies: int,
    model_name: str,
    model_label: str,
    opp_name: str,
    amp: bool,
    forced_actions: Optional[List[torch.Tensor]] = None,
) -> List[Dict[str, Any]]:
    """Record full per-ply policy/value/belief replays for a batch of B games."""
    B = mover_setup.size(0)
    if B == 0:
        return []

    env = VecStratego(B, device, max_plies=max_plies, no_battle_plies=no_battle_plies)
    t = env.t
    idx_all = torch.arange(B, device=device)
    a_side_dev = a_side.to(device)
    start_dev = start_side.to(device)
    env.reset(idx_all, mover_setup.to(device), other_setup.to(device), start_dev)

    setups = _extract_initial_setups(env, start_dev)
    game_plies: List[List[Dict[str, Any]]] = [[] for _ in range(B)]

    net.eval()
    use_amp = amp and device.type == 'cuda'
    actions = torch.zeros(B, dtype=torch.long, device=device)
    act_from_cpu = t.act_from.cpu()
    act_to_cpu = t.act_to.cpu()
    battle_cpu = t.battle.cpu()
    step_idx = 0

    while not bool(env.done.all()):
        live = ~env.done
        is_net = live if is_self_play else (live & (env.side == a_side_dev))
        in_idx = is_net.nonzero().flatten()
        io_idx = (live & ~is_net).nonzero().flatten()

        net_probs_cpu = None
        net_val_cpu = None
        net_belief_cpu = None
        net_pos_map: Dict[int, int] = {}

        if in_idx.numel():
            obs, _ = env.observe(in_idx)
            legal = env.legal[in_idx]
            with torch.autocast('cuda', enabled=use_amp):
                logits, value, belief_logits = net(
                    obs.contiguous(memory_format=torch.channels_last),
                    with_belief=True,
                )
            ml = masked_logits(logits, legal)
            policy_probs = torch.softmax(ml, dim=1)
            if forced_actions is None:
                if temperature > 0:
                    samp_probs = torch.softmax(ml / temperature, dim=1)
                    act_net = torch.multinomial(samp_probs, 1).squeeze(1)
                else:
                    act_net = ml.argmax(1)
                actions[in_idx] = act_net
            else:
                actions[in_idx] = forced_actions[step_idx].to(device)[in_idx]

            net_probs_cpu = policy_probs.float().cpu()
            net_val_cpu = value.float().cpu()
            net_belief_cpu = torch.softmax(belief_logits.float(), dim=1).cpu()  # (M, 12, 100)
            for local_i, g_val in enumerate(in_idx.cpu().tolist()):
                net_pos_map[g_val] = local_i

        if io_idx.numel():
            if forced_actions is None:
                actions[io_idx] = opp_policy.act(env, io_idx)
            else:
                actions[io_idx] = forced_actions[step_idx].to(device)[io_idx]

        # Snapshot pre-step state on CPU once per ply for all live games
        live_list = live.nonzero().flatten().cpu().tolist()
        act_cpu = actions.cpu()
        side_cpu = env.side.cpu()
        own_cpu = env.own.cpu()
        opp_cpu = env.opp.cpu()
        own_rev_cpu = env.own_rev.cpu()
        opp_rev_cpu = env.opp_rev.cpu()
        legal_counts_cpu = env.legal.sum(1).cpu()

        for g in live_list:
            side = int(side_cpu[g])
            act = int(act_cpu[g])
            c_src = int(act_from_cpu[act])
            c_dst = int(act_to_cpu[act])
            abs_src = _canon_to_abs(c_src, side)
            abs_dst = _canon_to_abs(c_dst, side)
            a_rank = int(own_cpu[g, c_src])
            d_rank = int(opp_cpu[g, c_dst])
            att_was_rev = bool(own_rev_cpu[g, c_src])
            def_was_rev = bool(opp_rev_cpu[g, c_dst])

            if d_rank > 0:
                res = int(battle_cpu[a_rank, d_rank])
                reveals: List[Dict[str, int]] = []
                if res == 1:
                    reveals.append({'sq': abs_dst, 'side': side, 'rank': a_rank})
                elif res == -1:
                    reveals.append({'sq': abs_dst, 'side': -side, 'rank': d_rank})
                battle_info: Optional[Dict[str, Any]] = {
                    'attacker': a_rank,
                    'defender': d_rank,
                    'result': res,
                    'defender_was_revealed': def_was_rev,
                    'attacker_was_revealed': att_was_rev,
                    'reveals': reveals,
                }
                sure_loss = bool(def_was_rev and res == -1)
                sure_loss_non_scout = bool(sure_loss and a_rank != SCOUT)
            else:
                battle_info = None
                sure_loss = False
                sure_loss_non_scout = False

            if g in net_pos_map:
                li = net_pos_map[g]
                p_row = net_probs_cpu[li]
                k = min(5, int(legal_counts_cpu[g]))
                top_p, top_a = torch.topk(p_row, k)
                top5 = [
                    {
                        'move': [
                            _canon_to_abs(int(act_from_cpu[int(ai)]), side),
                            _canon_to_abs(int(act_to_cpu[int(ai)]), side),
                        ],
                        'prob': round(float(pi), 4),
                    }
                    for pi, ai in zip(top_p.tolist(), top_a.tolist())
                ]
                played_prob = round(float(p_row[act]), 4)
                val_est = round(float(net_val_cpu[li]), 4)

                # Belief on all currently hidden enemy squares from mover's view
                hidden_mask = (opp_cpu[g] > 0) & (~opp_rev_cpu[g])
                hidden_sqs = hidden_mask.nonzero().flatten().tolist()
                belief: Dict[str, Dict[str, Any]] = {}
                b_mat = net_belief_cpu[li]  # (12, 100)
                for c_sq in hidden_sqs:
                    abs_sq = _canon_to_abs(c_sq, side)
                    pv = b_mat[:, c_sq]
                    top_idx = int(pv.argmax())
                    belief[str(abs_sq)] = {
                        'top_rank': top_idx + 1,
                        'top_prob': round(float(pv[top_idx]), 4),
                        'probs': [round(float(x), 3) for x in pv.tolist()],
                        'true_rank': int(opp_cpu[g, c_sq]),
                    }
            else:
                top5 = None
                played_prob = None
                val_est = None
                belief = None

            game_plies[g].append({
                'ply': len(game_plies[g]),
                'side': side,
                'move': [abs_src, abs_dst],
                'piece': a_rank,
                'battle': battle_info,
                'top5': top5,
                'played_prob': played_prob,
                'value': val_est,
                'belief': belief,
                'sure_loss': sure_loss,
                'sure_loss_non_scout': sure_loss_non_scout,
            })

        env.step(actions)
        step_idx += 1

    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    winner_cpu = env.winner.cpu()
    end_cpu = env.end_type.cpu()
    a_side_cpu = a_side.cpu()
    start_cpu = start_side.cpu()

    replays: List[Dict[str, Any]] = []
    for g in range(B):
        net_side_val = 0 if is_self_play else int(a_side_cpu[g])
        red_name = model_label if net_side_val in (0, 1) else opp_name
        blue_name = model_label if net_side_val in (0, -1) else opp_name
        plies = game_plies[g]
        net_plies = [p for p in plies if net_side_val == 0 or p['side'] == net_side_val]
        sl_all_plies = [p['ply'] for p in net_plies if p['sure_loss']]
        sl_ns_plies = [p['ply'] for p in net_plies if p['sure_loss_non_scout']]
        replays.append({
            'schema_version': SCHEMA_VERSION,
            'meta': {
                'source': 'record_games',
                'created_at': now_iso,
                'model': model_name,
                'opponent': opp_name,
                'red': red_name,
                'blue': blue_name,
                'net_side': net_side_val,
                'starting_side': int(start_cpu[g]),
                'temperature': float(temperature),
                'seed': int(seed),
                'max_plies': int(max_plies),
                'no_battle_plies': int(no_battle_plies),
                'winner': int(winner_cpu[g]),
                'end_type': END_NAMES.get(int(end_cpu[g]), 'unknown'),
                'num_plies': len(plies),
                'blunders': {
                    'sure_loss_total': len(sl_all_plies),
                    'sure_loss_non_scout': len(sl_ns_plies),
                    'sure_loss_plies': sl_ns_plies,
                    'sure_loss_all_plies': sl_all_plies,
                },
            },
            'setups': setups[g],
            'plies': plies,
        })
    return replays


@torch.no_grad()
def record_games(
    net: PolicyValueNet,
    opponent: str,
    num_games: int,
    device: torch.device,
    temperature: float = 1.0,
    seed: int = 0,
    max_plies: int = 1500,
    no_battle_plies: int = 200,
    bank: Optional[torch.Tensor] = None,
    model_name: str = 'net',
    model_label: Optional[str] = None,
    expert_styles: Optional[str] = None,
    filter_kind: str = 'any',
    search_pool: int = 0,
    net_side_mode: str = 'alternate',
    amp: bool = True,
    max_search_rounds: int = 4,
) -> List[Dict[str, Any]]:
    """Record `num_games` replays, optionally searching a larger pool to satisfy `filter_kind`."""
    opp_policy, opp_name = make_opponent(opponent, device, temperature=temperature, expert_styles=expert_styles)
    is_self_play = opponent == 'self'
    label = model_label or _model_label(model_name)

    if bank is None:
        pool_n = max(200, num_games * 8, search_pool * 2)
        bank = diverse_setup_bank(pool_n, seed=seed + 1)

    if filter_kind == 'any':
        gen = torch.Generator(device='cpu').manual_seed(seed)
        a_side = _select_sides(num_games, net_side_mode, torch.device('cpu'))
        start_side = torch.randint(0, 2, (num_games,), generator=gen) * 2 - 1
        mover_setup = bank[torch.randint(0, bank.size(0), (num_games,), generator=gen)]
        other_setup = bank[torch.randint(0, bank.size(0), (num_games,), generator=gen)]
        return _record_batch(
            net=net, opp_policy=opp_policy, is_self_play=is_self_play,
            mover_setup=mover_setup, other_setup=other_setup, start_side=start_side, a_side=a_side,
            device=device, temperature=temperature, seed=seed,
            max_plies=max_plies, no_battle_plies=no_battle_plies,
            model_name=model_name, model_label=label, opp_name=opp_name, amp=amp,
        )

    pool_size = search_pool if search_pool > 0 else max(32, num_games * 16)
    collected: List[Dict[str, Any]] = []
    for r_idx in range(max_search_rounds):
        need = num_games - len(collected)
        if need <= 0:
            break
        round_seed = seed + r_idx * 1009
        sr = _search_pool(
            net=net, opp_policy=opp_policy, is_self_play=is_self_play,
            pool_size=pool_size, bank=bank, device=device, temperature=temperature,
            seed=round_seed, max_plies=max_plies, no_battle_plies=no_battle_plies,
            net_side_mode=net_side_mode, amp=amp,
        )
        candidates = [
            g for g in range(pool_size)
            if _match_filter(filter_kind, int(sr['sl_non_scout'][g]), int(sr['res_net'][g]))
        ]
        if filter_kind in ('blunder', 'blunder_loss'):
            # Prefer games with the most non-Scout sure-loss blunders, then shorter games
            candidates.sort(key=lambda g: (-int(sr['sl_non_scout'][g]), int(sr['ply'][g])))
        else:
            candidates.sort(key=lambda g: int(sr['ply'][g]))
        chosen = candidates[:need]
        if not chosen:
            continue
        ch_t = torch.tensor(chosen, dtype=torch.long)
        forced = [step_acts[ch_t] for step_acts in sr['actions_seq']]
        replays = _record_batch(
            net=net, opp_policy=opp_policy, is_self_play=is_self_play,
            mover_setup=sr['mover_setup'][ch_t],
            other_setup=sr['other_setup'][ch_t],
            start_side=sr['start_side'][ch_t],
            a_side=sr['a_side'][ch_t],
            device=device, temperature=temperature, seed=round_seed,
            max_plies=max_plies, no_battle_plies=no_battle_plies,
            model_name=model_name, model_label=label, opp_name=opp_name,
            amp=amp, forced_actions=forced,
        )
        collected.extend(replays)
    return collected


def verify_replay(data: Dict[str, Any]) -> List[str]:
    """Validate replay JSON schema and replay every move to verify board/battle/belief consistency.

    Returns a list of error strings (empty list means completely valid).
    """
    errs: List[str] = []
    if not isinstance(data, dict):
        return ['replay root must be a dict']
    if data.get('schema_version') != SCHEMA_VERSION:
        errs.append(f"unexpected schema_version: {data.get('schema_version')!r}")
    meta = data.get('meta')
    setups = data.get('setups')
    plies = data.get('plies')
    if not isinstance(meta, dict):
        errs.append('missing meta dict')
        return errs
    if not isinstance(setups, dict) or 'red' not in setups or 'blue' not in setups:
        errs.append('missing setups.red or setups.blue')
        return errs
    if not isinstance(plies, list):
        errs.append('missing plies list')
        return errs

    red_s, blue_s = setups['red'], setups['blue']
    if len(red_s) != 40 or len(blue_s) != 40:
        errs.append(f'setup lengths must be 40/40, got {len(red_s)}/{len(blue_s)}')
        return errs

    # Reconstruct absolute 10x10 board (+rank Red, -rank Blue, -13 Lake)
    board = [0] * 100
    for sq in LAKE_SQUARES:
        board[sq] = -13
    for i, r in enumerate(blue_s):
        board[i] = -int(r)
    for i, r in enumerate(red_s):
        board[60 + i] = int(r)
    rev_to_red = [False] * 100    # Blue pieces known to Red
    rev_to_blue = [False] * 100   # Red pieces known to Blue

    t = tables(torch.device('cpu'))
    bt = t.battle
    expected_side = meta.get('starting_side')

    for idx, p in enumerate(plies):
        if p.get('ply') != idx:
            errs.append(f'ply {idx}: ply field is {p.get("ply")}')
        side = p.get('side')
        if side not in (1, -1):
            errs.append(f'ply {idx}: invalid side {side}')
            break
        if expected_side is not None and side != expected_side:
            errs.append(f'ply {idx}: expected side {expected_side}, got {side}')
        expected_side = -side

        mv = p.get('move')
        if not (isinstance(mv, list) and len(mv) == 2):
            errs.append(f'ply {idx}: invalid move {mv}')
            break
        src, dst = int(mv[0]), int(mv[1])
        if not (0 <= src < 100 and 0 <= dst < 100) or dst in LAKE_SQUARES:
            errs.append(f'ply {idx}: out-of-bounds or lake move {mv}')
            break
        mover = board[src]
        if mover == 0 or mover == -13 or (mover * side <= 0):
            errs.append(f'ply {idx}: no piece of side {side} at src={src} (board={mover})')
            break
        a_rank = abs(mover)
        if p.get('piece') != a_rank:
            errs.append(f'ply {idx}: piece={p.get("piece")} != board rank {a_rank}')
        if a_rank in (FLAG, BOMB):
            errs.append(f'ply {idx}: immobile piece {a_rank} moved')

        # Validate belief keys against currently hidden enemy pieces BEFORE move
        belief = p.get('belief')
        if belief is not None:
            for sq_s, bentry in belief.items():
                bsq = int(sq_s)
                bv = board[bsq]
                if bv == 0 or bv == -13 or (bv * side >= 0):
                    errs.append(f'ply {idx}: belief sq {bsq} has no enemy piece (board={bv})')
                    break
                is_rev = rev_to_red[bsq] if side == 1 else rev_to_blue[bsq]
                if is_rev:
                    errs.append(f'ply {idx}: belief sq {bsq} is already revealed')
                    break
                if bentry.get('true_rank') != abs(bv):
                    errs.append(f'ply {idx}: belief sq {bsq} true_rank={bentry.get("true_rank")} != {abs(bv)}')
                    break
                probs = bentry.get('probs')
                if probs is not None and len(probs) != 12:
                    errs.append(f'ply {idx}: belief sq {bsq} probs length {len(probs)} != 12')
                    break

        target = board[dst]
        binfo = p.get('battle')
        if target != 0 and target != -13:
            if target * side > 0:
                errs.append(f'ply {idx}: attacked own piece at dst={dst}')
                break
            d_rank = abs(target)
            res = int(bt[a_rank, d_rank])
            def_was_rev = rev_to_red[dst] if side == 1 else rev_to_blue[dst]
            if binfo is None:
                errs.append(f'ply {idx}: battle occurred at {dst} but battle is None')
            else:
                if binfo.get('attacker') != a_rank or binfo.get('defender') != d_rank or binfo.get('result') != res:
                    errs.append(f'ply {idx}: battle mismatch {binfo} vs ({a_rank},{d_rank},{res})')
                if bool(binfo.get('defender_was_revealed')) != def_was_rev:
                    errs.append(f'ply {idx}: defender_was_revealed={binfo.get("defender_was_revealed")} != {def_was_rev}')
            exp_sl = bool(def_was_rev and res == -1)
            exp_sl_ns = bool(exp_sl and a_rank != SCOUT)
            if bool(p.get('sure_loss')) != exp_sl:
                errs.append(f'ply {idx}: sure_loss={p.get("sure_loss")} != {exp_sl}')
            if bool(p.get('sure_loss_non_scout')) != exp_sl_ns:
                errs.append(f'ply {idx}: sure_loss_non_scout={p.get("sure_loss_non_scout")} != {exp_sl_ns}')

            if res == 1:
                board[dst] = mover
                board[src] = 0
                if side == 1:
                    rev_to_blue[dst] = True
                    rev_to_blue[src] = False
                    rev_to_red[dst] = False
                else:
                    rev_to_red[dst] = True
                    rev_to_red[src] = False
                    rev_to_blue[dst] = False
            elif res == -1:
                board[src] = 0
                rev_to_red[src] = False
                rev_to_blue[src] = False
                if side == 1:
                    rev_to_red[dst] = True
                else:
                    rev_to_blue[dst] = True
            else:
                board[src] = 0
                board[dst] = 0
                rev_to_red[src] = rev_to_red[dst] = False
                rev_to_blue[src] = rev_to_blue[dst] = False
        else:
            if binfo is not None:
                errs.append(f'ply {idx}: quiet move to {dst} has non-null battle')
            if p.get('sure_loss') or p.get('sure_loss_non_scout'):
                errs.append(f'ply {idx}: quiet move flagged as sure_loss')
            board[dst] = mover
            board[src] = 0
            if side == 1:
                rev_to_blue[dst] = rev_to_blue[src]
                rev_to_blue[src] = False
            else:
                rev_to_red[dst] = rev_to_red[src]
                rev_to_red[src] = False

        if len(errs) >= 10:
            break

    if meta.get('num_plies') != len(plies):
        errs.append(f"meta.num_plies={meta.get('num_plies')} != len(plies)={len(plies)}")
    return errs


def save_replays(replays: Sequence[Dict[str, Any]], out_dir: str, prefix: str = 'game') -> List[str]:
    """Write replay JSON files into `out_dir` and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for i, rep in enumerate(replays, 1):
        m = rep['meta']
        win_tag = 'red' if m['winner'] == 1 else ('blue' if m['winner'] == -1 else 'draw')
        bl = m['blunders']['sure_loss_non_scout']
        bl_tag = f'_blunder{bl}' if bl > 0 else ''
        fname = f'{prefix}_{i:02d}_{m["red"]}_vs_{m["blue"]}_{win_tag}_{m["end_type"]}{bl_tag}.json'
        path = os.path.join(out_dir, fname)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(rep, f, separators=(',', ':'))
        paths.append(path)
    return paths


def write_index(out_dir: str) -> str:
    """Write runs/replays/index.json summarizing all replay JSON files in `out_dir`."""
    entries = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith('.json') or fname == 'index.json':
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            if data.get('schema_version') != SCHEMA_VERSION:
                continue
            m = data['meta']
            entries.append({
                'file': fname,
                'red': m.get('red'),
                'blue': m.get('blue'),
                'model': m.get('model'),
                'opponent': m.get('opponent'),
                'winner': m.get('winner'),
                'end_type': m.get('end_type'),
                'num_plies': m.get('num_plies'),
                'sure_loss_non_scout': m.get('blunders', {}).get('sure_loss_non_scout', 0),
                'sure_loss_plies': m.get('blunders', {}).get('sure_loss_plies', []),
            })
        except Exception:
            continue
    idx_path = os.path.join(out_dir, 'index.json')
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump({'schema_version': SCHEMA_VERSION, 'games': entries}, f, indent=2)
    return idx_path


def record_curated_suite(
    net_path: str,
    out_dir: str,
    device: torch.device,
    temperature: float = 1.0,
    seed: int = 42,
    expert_styles: Optional[str] = None,
) -> List[str]:
    """Record a curated 7-game inspection suite showcasing R10 blunders, wins, losses, and self-play."""
    net = load_net(net_path, device).eval()
    label = _model_label(net_path)
    bank = diverse_setup_bank(4000, seed=seed + 1)

    plan = [
        # (prefix, opponent, num_games, filter_kind, pool, seed_offset, net_side_mode)
        ('01_r10_blunder_vs_expert', 'expert', 2, 'blunder', 64, 0, 'alternate'),
        ('02_win_vs_expert', 'expert', 1, 'win', 16, 100, 'red'),
        ('03_blunder_vs_expert3', 'expert3', 1, 'blunder', 64, 200, 'red'),
        ('04_loss_vs_expert3', 'expert3', 1, 'loss', 32, 300, 'blue'),
        ('05_win_vs_expert3', 'expert3', 1, 'win', 16, 400, 'red'),
        ('06_selfplay', 'self', 1, 'any', 0, 500, 'red'),
    ]
    all_paths: List[str] = []
    for prefix, opp, n, fkind, pool, off, side_mode in plan:
        reps = record_games(
            net=net, opponent=opp, num_games=n, device=device,
            temperature=temperature, seed=seed + off, bank=bank,
            model_name=net_path, model_label=label, expert_styles=expert_styles,
            filter_kind=fkind, search_pool=pool, net_side_mode=side_mode,
        )
        for rep in reps:
            errs = verify_replay(rep)
            if errs:
                raise RuntimeError(f'Recorded replay failed verification ({prefix}): {errs}')
        paths = save_replays(reps, out_dir, prefix=prefix)
        for p, rep in zip(paths, reps):
            m = rep['meta']
            print(
                f"[record] {os.path.basename(p)}: {m['red']} vs {m['blue']} -> "
                f"winner={m['winner']} ({m['end_type']}, {m['num_plies']} plies, "
                f"R10_blunders={m['blunders']['sure_loss_non_scout']})",
                flush=True,
            )
        all_paths.extend(paths)
    idx_path = write_index(out_dir)
    print(f'[record] wrote {len(all_paths)} replays + {idx_path}', flush=True)
    return all_paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Record Stratego replays with policy/value/belief traces.')
    ap.add_argument('--net', required=True, help='Path to PolicyValueNet checkpoint (.pt)')
    ap.add_argument('--opponent', default='expert',
                    help='expert, expert2, expert3, random, self, or path to .pt / style .json')
    ap.add_argument('--games', type=int, default=5, help='Number of games to record')
    ap.add_argument('--out', default=os.path.join('..', 'runs', 'replays'), help='Output directory')
    ap.add_argument('--prefix', default='game', help='Filename prefix')
    ap.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (0 = greedy)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--filter', dest='filter_kind', default='any',
                    choices=('any', 'blunder', 'blunder_loss', 'win', 'loss', 'draw'),
                    help='Filter games from a batched search pool')
    ap.add_argument('--pool', type=int, default=0, help='Search pool size when --filter != any')
    ap.add_argument('--net-side', default='alternate', choices=('alternate', 'red', 'blue'))
    ap.add_argument('--max-plies', type=int, default=1500)
    ap.add_argument('--no-battle-plies', type=int, default=200)
    ap.add_argument('--expert-styles', default=None)
    ap.add_argument('--device', default=None, help='cuda or cpu (default: auto)')
    ap.add_argument('--suite', action='store_true',
                    help='Record the curated 7-game inspection suite (blunders, expert/expert3 wins & losses, self-play)')
    args = ap.parse_args(argv)

    dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.suite:
        record_curated_suite(
            net_path=args.net, out_dir=args.out, device=dev,
            temperature=args.temperature, seed=args.seed, expert_styles=args.expert_styles,
        )
        return

    net = load_net(args.net, dev).eval()
    reps = record_games(
        net=net, opponent=args.opponent, num_games=args.games, device=dev,
        temperature=args.temperature, seed=args.seed,
        max_plies=args.max_plies, no_battle_plies=args.no_battle_plies,
        model_name=args.net, expert_styles=args.expert_styles,
        filter_kind=args.filter_kind, search_pool=args.pool, net_side_mode=args.net_side,
    )
    for rep in reps:
        errs = verify_replay(rep)
        if errs:
            raise RuntimeError(f'Recorded replay failed verification: {errs}')
    paths = save_replays(reps, args.out, prefix=args.prefix)
    idx_path = write_index(args.out)
    for p, rep in zip(paths, reps):
        m = rep['meta']
        print(
            f"[record] {os.path.basename(p)}: {m['red']} vs {m['blue']} -> "
            f"winner={m['winner']} ({m['end_type']}, {m['num_plies']} plies, "
            f"R10_blunders={m['blunders']['sure_loss_non_scout']})",
            flush=True,
        )
    print(f'[record] saved {len(paths)} games -> {args.out} (index: {idx_path})', flush=True)


if __name__ == '__main__':
    main()
