"""Matches between two batched policies on the batched engine."""

from typing import Dict, Optional

import torch

from vec.engine import VecStratego

END_NAMES = {1: 'flag', 2: 'no_moves', 3: 'no_battle', 4: 'turn_limit'}


def deal(env: VecStratego, idx: torch.Tensor, bank_mover: torch.Tensor, bank_other: Optional[torch.Tensor] = None,
         start_side: Optional[torch.Tensor] = None):
    """New games in rows `idx` with setups drawn from the bank(s); random starting player."""
    m = idx.numel()
    if m == 0:
        return
    bank_other = bank_mover if bank_other is None else bank_other
    dev = env.device
    a = bank_mover[torch.randint(0, bank_mover.size(0), (m,), device=bank_mover.device)]
    b = bank_other[torch.randint(0, bank_other.size(0), (m,), device=bank_other.device)]
    if start_side is None:
        start_side = torch.randint(0, 2, (m,), device=dev) * 2 - 1
    env.reset(idx, a, b, start_side)


@torch.no_grad()
def play_match(policy_a, policy_b, num_games: int, bank: torch.Tensor, device, max_plies: int = 1500,
               no_battle_plies: int = 200, bank_b: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """`num_games` games, policy A against policy B, all at once. A is Player 1 in the even games
    and Player 2 in the odd ones; the starting player is random. Returns A's wins / losses /
    draws, mean plies, and how its wins, losses and the draws ended."""
    env = VecStratego(num_games, device, max_plies=max_plies, no_battle_plies=no_battle_plies)
    all_idx = torch.arange(num_games, device=env.device)
    a_side = torch.where(all_idx % 2 == 0, 1, -1)
    start = torch.randint(0, 2, (num_games,), device=env.device) * 2 - 1
    # the mover's setup comes from its own bank
    bank_b = bank if bank_b is None else bank_b
    pick = lambda bk: bk[torch.randint(0, bk.size(0), (num_games,), device=bk.device)].to(env.device)
    sa, sb = pick(bank), pick(bank_b)
    a_starts = (start == a_side)[:, None]
    env.reset(all_idx, torch.where(a_starts, sa, sb), torch.where(a_starts, sb, sa), start)
    actions = torch.zeros(num_games, dtype=torch.long, device=env.device)
    while not bool(env.done.all()):
        live = ~env.done
        ia = (live & (env.side == a_side)).nonzero().flatten()
        ib = (live & (env.side != a_side)).nonzero().flatten()
        if ia.numel():
            actions[ia] = policy_a.act(env, ia)
        if ib.numel():
            actions[ib] = policy_b.act(env, ib)
        env.step(actions)
    res = env.winner * a_side
    out = {'games': num_games, 'wins': int((res == 1).sum()), 'losses': int((res == -1).sum()),
           'draws': int((res == 0).sum()), 'plies': float(env.ply.float().mean())}
    out['win_rate'] = out['wins'] / num_games
    for code, name in END_NAMES.items():
        e = env.end_type == code
        out[f'win_{name}'] = int((e & (res == 1)).sum())
        out[f'loss_{name}'] = int((e & (res == -1)).sum())
        out[f'draw_{name}'] = int((e & (res == 0)).sum())
    return out


def summary(r: Dict[str, float]) -> str:
    ends = ' '.join(f"{k}={v}" for k, v in r.items() if k.split('_')[0] in ('win', 'loss', 'draw') and k != 'win_rate' and v)
    return f"{r['wins']}/{r['losses']}/{r['draws']} of {r['games']} ({100 * r['win_rate']:.1f}% wins, {r['plies']:.0f} plies) [{ends}]"
