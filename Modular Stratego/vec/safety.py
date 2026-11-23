"""
Play-time safety filter (applied after the network, never during training).

Masks out strictly public-information tactical blunders:
  1. A non-Scout attacking an enemy piece that is already revealed and beats the attacker
     (battle[attacker, defender] == -1).
  2. A non-Miner attacking an already-revealed Bomb (including Scouts, since attacking a
     revealed Bomb gains no information and always loses the attacker).

If filtering would remove every legal move for a game, the filter falls back to that
game's original legal mask so the policy always has at least one legal action.

Reads ONLY public/own information: own piece ranks, revealed enemy piece ranks
(``opp`` masked by ``opp_rev``), and the legal action mask. Never inspects hidden enemy
ranks.
"""

from typing import Dict, Optional

import torch

from vec.engine import BOMB, MINER, NUM_SQUARES, SCOUT, Tables, VecStratego

_unsafe_tables: Dict[str, torch.Tensor] = {}


def unsafe_table(t: Tables) -> torch.Tensor:
    """(13, 13) bool table: True iff attacking revealed defender ``d`` with own rank ``a`` is forbidden."""
    key = str(t.device)
    tbl = _unsafe_tables.get(key)
    if tbl is not None:
        return tbl
    tbl = torch.zeros(13, 13, dtype=torch.bool, device=t.device)
    battle = t.battle
    for a in range(1, 13):
        for d in range(1, 13):
            forbid_rev = (a != SCOUT) and (int(battle[a, d]) == -1)
            forbid_bomb = (a != MINER) and (d == BOMB)
            if forbid_rev or forbid_bomb:
                tbl[a, d] = True
    _unsafe_tables[key] = tbl
    return tbl


def safe_legal_mask(
    own: torch.Tensor,
    opp: torch.Tensor,
    opp_rev: torch.Tensor,
    legal: torch.Tensor,
    t: Tables,
) -> torch.Tensor:
    """Return a copy of ``legal`` (B, 3600) with unsafe attacks on revealed enemies masked out.

    Falls back to the unmodified row of ``legal`` whenever all legal moves in a game would be
    masked out. Only revealed enemy squares (``opp_rev``) are consulted; hidden enemy ranks
    are zeroed before any lookup.
    """
    rev_opp = torch.where(opp_rev, opp, torch.zeros_like(opp))
    act_to = t.act_to.clamp(max=NUM_SQUARES - 1)
    att = own[:, t.act_from]          # (B, 3600)
    dfn = rev_opp[:, act_to]          # (B, 3600), 0 unless target is a revealed enemy
    unsafe = unsafe_table(t)[att, dfn]
    filtered = legal & ~unsafe
    has_safe = filtered.any(dim=1, keepdim=True)
    return torch.where(has_safe, filtered, legal)


def safe_mask_for_env(env: VecStratego, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convenience wrapper applying :func:`safe_legal_mask` to ``env`` (all rows or ``idx``)."""
    if idx is None:
        return safe_legal_mask(env.own, env.opp, env.opp_rev, env.legal, env.t)
    return safe_legal_mask(env.own[idx], env.opp[idx], env.opp_rev[idx], env.legal[idx], env.t)
