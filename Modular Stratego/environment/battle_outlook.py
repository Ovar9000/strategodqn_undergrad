"""
Battle-outlook planes: what the rules say about each attack, read off the public information.

For every own piece and each of the four directions of the canonical frame (up, down, left,
right), the probability that attacking the enemy piece it would hit in that direction wins, and
the probability that it loses (1 - win - lose is a tie; both are 0 when no enemy is in reach). A
Scout reaches the first piece along the line, every other piece only the adjacent square. One
more plane gives, for every own piece, the largest probability that an adjacent enemy piece
would beat it if that piece attacked (a Bomb or the Flag never attacks).

The rank of an enemy piece is known when it has been revealed. A hidden piece's rank is drawn
from the ranks of all the hidden enemy pieces, which is public: every piece that left the board
was revealed in its battle. A hidden piece that has moved is not a Bomb or the Flag, and one that
moved more than one square is a Scout. The ranks are spread over the hidden pieces in proportion
to their counts, first the Scouts that made long moves, then the other moved pieces among the
movable ranks, then the unmoved ones from what is left.

These planes put skill 3 of the thesis's skill ladder (comparing two known ranks) and a first
form of skill 4 (inference about hidden pieces) into the observation, as the action mask puts in
the rules of movement. The network still decides what each outcome is worth.
"""

from typing import Optional, Sequence

import torch

from environment.battle import BattleResolver
from environment.piece import PieceType

NUM_RANKS = 12  # plane index = PieceType value - 1
SCOUT = PieceType.SCOUT.value - 1
IMMOBILE = (PieceType.FLAG.value - 1, PieceType.BOMB.value - 1)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # action_encoding.DIRECTIONS (canonical frame)
NUM_OUTLOOK_PLANES = 2 * len(DIRECTIONS) + 1

# The 40 pieces of one side, counted by rank (environment._generate_pieces)
INITIAL_COUNTS = [0.0] * NUM_RANKS
for _t, _n in ((PieceType.FLAG, 1), (PieceType.SPY, 1), (PieceType.SCOUT, 8), (PieceType.MINER, 5),
               (PieceType.SERGEANT, 4), (PieceType.LIEUTENANT, 4), (PieceType.CAPTAIN, 4), (PieceType.MAJOR, 3),
               (PieceType.COLONEL, 2), (PieceType.GENERAL, 1), (PieceType.MARSHAL, 1), (PieceType.BOMB, 6)):
    INITIAL_COUNTS[_t.value - 1] = float(_n)


def _outcome_matrices():
    """(attacker, defender) win and lose indicators; an immobile attacker never attacks."""
    br = BattleResolver()
    win = torch.zeros(NUM_RANKS, NUM_RANKS)
    lose = torch.zeros(NUM_RANKS, NUM_RANKS)
    for a in range(NUM_RANKS):
        if a in IMMOBILE:
            continue
        for d in range(NUM_RANKS):
            r = br.resolve_battle(PieceType(a + 1), PieceType(d + 1))
            win[a, d], lose[a, d] = float(r == 1), float(r == -1)
    return win, lose


WIN, LOSE = _outcome_matrices()
_cache = {}


def _on(device):
    key = str(device)
    if key not in _cache:
        _cache[key] = (WIN.to(device), LOSE.to(device), torch.tensor(INITIAL_COUNTS, device=device))
    return _cache[key]


def hidden_rank_distributions(pool: torch.Tensor, n_long: torch.Tensor, n_moved: torch.Tensor) -> torch.Tensor:
    """(B, 3, 12) rank distributions of a hidden enemy piece that has not moved, has moved (one
    square at a time) and has made a long move, given `pool` (B, 12): the rank counts of all the
    hidden enemy pieces, and the numbers of hidden long-moved and other moved pieces (B,)."""
    B, dev = pool.size(0), pool.device
    movable = torch.ones(NUM_RANKS, device=dev)
    movable[list(IMMOBILE)] = 0.0
    c = pool.clone()
    c[:, SCOUT] = (c[:, SCOUT] - n_long).clamp_min(0.0)
    mov = c * movable
    total = mov.sum(1, keepdim=True)
    moved = torch.where(total > 0, mov / total.clamp_min(1e-9), movable / movable.sum())
    rest = (c - n_moved.view(-1, 1) * moved).clamp_min(0.0)
    total = rest.sum(1, keepdim=True)
    unmoved = torch.where(total > 0, rest / total.clamp_min(1e-9), torch.full_like(rest, 1.0 / NUM_RANKS))
    long = torch.zeros(B, NUM_RANKS, device=dev)
    long[:, SCOUT] = 1.0
    return torch.stack([unmoved, moved, long], dim=1)


def _ray_indices() -> torch.Tensor:
    """(4, 100, 9) square indices along each direction from each square, distance 1..9; 100 (a
    sentinel square, occupied and empty of targets) once the line leaves the board."""
    idx = torch.full((len(DIRECTIONS), 100, 9), 100, dtype=torch.long)
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for sq in range(100):
            r, c = divmod(sq, 10)
            for k in range(1, 10):
                rr, cc = r + dr * k, c + dc * k
                if not (0 <= rr < 10 and 0 <= cc < 10):
                    break
                idx[d, sq, k - 1] = rr * 10 + cc
    return idx


RAYS = _ray_indices()


def enemy_rank_distribution(obs: torch.Tensor, pools: Optional[Sequence] = None) -> torch.Tensor:
    """
    (B, 12, 10, 10): the rank distribution (index PieceType value - 1) of the enemy piece on each
    square of canonical-frame observation planes `obs` (B, >=32, 10, 10), 0 where there is none: a
    revealed piece's rank, or for a hidden one the distribution of its class (unmoved, moved,
    long-moved) given the rank counts of all hidden pieces.

    pools: per item, the rank counts (12,) of the viewer's hidden enemy pieces (GameState.
    hidden_pool), or None. Without one, the counts are estimated: the starting set minus the
    revealed enemy pieces on the board, scaled to the number of hidden pieces (pieces captured
    earlier are not known then).
    """
    from environment.observation import CH_ENEMY_KNOWN, CH_ENEMY_HIDDEN, CH_ENEMY_MOVED, CH_ENEMY_LONG_MOVED
    B, dev = obs.size(0), obs.device
    _, _, initial = _on(dev)
    known = obs[:, CH_ENEMY_KNOWN:CH_ENEMY_KNOWN + NUM_RANKS]
    hidden = obs[:, CH_ENEMY_HIDDEN]
    long_moved = obs[:, CH_ENEMY_LONG_MOVED] * hidden
    moved = (obs[:, CH_ENEMY_MOVED] * hidden - long_moved).clamp_min(0.0)
    unmoved = (hidden - moved - long_moved).clamp_min(0.0)

    n_hidden = hidden.sum((1, 2))
    estimate = (initial.view(1, -1) - known.sum((2, 3))).clamp_min(0.0)
    estimate = estimate * (n_hidden / estimate.sum(1).clamp_min(1e-9)).view(-1, 1)
    pool = estimate
    if pools is not None and any(p is not None for p in pools):
        pool = torch.stack([torch.as_tensor(p, dtype=torch.float32, device=dev) if p is not None else estimate[i]
                            for i, p in enumerate(pools)])
    dist = hidden_rank_distributions(pool, long_moved.sum((1, 2)), moved.sum((1, 2)))  # (B, 3, 12)
    return (known + unmoved.unsqueeze(1) * dist[:, 0].view(B, -1, 1, 1)
            + moved.unsqueeze(1) * dist[:, 1].view(B, -1, 1, 1)
            + long_moved.unsqueeze(1) * dist[:, 2].view(B, -1, 1, 1))


def outlook_planes(obs: torch.Tensor, pools: Optional[Sequence] = None) -> torch.Tensor:
    """
    (B, 9, 10, 10) battle-outlook planes from canonical-frame observation planes `obs` (B, >=32,
    10, 10): P(win) up/down/left/right, P(lose) up/down/left/right, threat. `pools` as in
    enemy_rank_distribution.
    """
    from environment.observation import CH_OWN, CH_ENEMY_KNOWN, CH_ENEMY_HIDDEN, CH_LAKE
    B, dev = obs.size(0), obs.device
    win, lose, _ = _on(dev)
    own = obs[:, CH_OWN:CH_OWN + NUM_RANKS]
    known = obs[:, CH_ENEMY_KNOWN:CH_ENEMY_KNOWN + NUM_RANKS]
    hidden = obs[:, CH_ENEMY_HIDDEN]
    target = enemy_rank_distribution(obs, pools)  # (B, 12, H, W)
    occupied = ((own.sum(1) + known.sum(1) + hidden + obs[:, CH_LAKE]) > 0).float()
    is_scout = own[:, SCOUT].reshape(B, 1, 100)  # (B, 1, squares)

    rays = RAYS.to(dev) if RAYS.device != dev else RAYS
    occ = torch.cat([occupied.reshape(B, 100), torch.ones(B, 1, device=dev, dtype=occupied.dtype)], 1)  # (B, 101)
    tgt = torch.cat([target.reshape(B, NUM_RANKS, 100), torch.zeros(B, NUM_RANKS, 1, device=dev, dtype=target.dtype)], 2)
    hits = occ[:, rays]                                                  # (B, 4, 100, 9)
    first = hits * (hits.cumsum(-1) == 1)                                # the first occupied square
    hit_sq = (first.long() * rays).sum(-1)                               # (B, 4, 100), 100 if none
    hit_sq = torch.where(first.bool().any(-1), hit_sq, torch.full_like(hit_sq, 100))
    flat = lambda sq: tgt.gather(2, sq.reshape(B, 1, -1).expand(-1, NUM_RANKS, -1)).reshape(B, NUM_RANKS, len(DIRECTIONS), 100)
    ray = flat(hit_sq)                                                   # (B, 12, 4, 100): a Scout's target
    adjacent = flat(rays[:, :, 0].expand(B, -1, -1))                     # (B, 12, 4, 100)
    reach = is_scout.unsqueeze(1) * ray + (1 - is_scout.unsqueeze(1)) * adjacent
    own_f = own.reshape(B, NUM_RANKS, 100)
    out = torch.zeros(B, NUM_OUTLOOK_PLANES, 100, device=dev, dtype=obs.dtype)
    out[:, 0:4] = torch.einsum('bas,ae,beds->bds', own_f, win, reach)
    out[:, 4:8] = torch.einsum('bas,ae,beds->bds', own_f, lose, reach)
    # The adjacent enemy piece attacking this one: it wins when it is the attacker in `win`
    out[:, 8] = torch.einsum('beds,ea,bas->bds', adjacent, win, own_f).amax(1)
    return out.reshape(B, NUM_OUTLOOK_PLANES, obs.size(2), obs.size(3))
