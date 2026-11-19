"""
Setup banks for the batched engine.

The thesis's heuristic setup always puts the Flag in a back corner with Bombs next to it. An
agent trained only on those setups can learn "the Flag is in a corner", which a human's setup
will not honour. diverse_setup_bank mixes three sources:

    heuristic   the thesis's setup agent (flag in a back corner)
    structured  flag anywhere in the back two rows with 2-3 Bombs around it, the other pieces
                placed by row preferences (Scouts forward, Miners back, ...)
    random      a uniformly random arrangement

A setup is (40,) rank values for squares 60..99 of the side's own canonical frame: index 0-9 is
the front row, 30-39 the back row.
"""

import random
from typing import Tuple

import torch

from environment.piece import PieceType as P
from vec.compat import all_pieces, heuristic_setup_bank

# preference for the rows, front to back
ROW_WEIGHTS = {
    P.MARSHAL: (3, 3, 2, 1), P.GENERAL: (3, 3, 2, 1), P.COLONEL: (2, 3, 2, 1), P.MAJOR: (2, 3, 2, 1),
    P.CAPTAIN: (3, 2, 2, 1), P.LIEUTENANT: (3, 2, 2, 1), P.SERGEANT: (2, 2, 2, 1), P.MINER: (1, 1, 3, 3),
    P.SCOUT: (4, 2, 1, 1), P.SPY: (1, 3, 3, 1), P.BOMB: (1, 2, 3, 2),
}
ORDER = (P.MARSHAL, P.GENERAL, P.SPY, P.COLONEL, P.MAJOR, P.MINER, P.SCOUT, P.CAPTAIN, P.LIEUTENANT, P.SERGEANT)
COUNTS = {P.MARSHAL: 1, P.GENERAL: 1, P.SPY: 1, P.COLONEL: 2, P.MAJOR: 3, P.MINER: 5, P.SCOUT: 8,
          P.CAPTAIN: 4, P.LIEUTENANT: 4, P.SERGEANT: 4}


def structured_setup(rng: random.Random) -> torch.Tensor:
    setup = [0] * 40
    free = set(range(40))

    def put(i, piece):
        setup[i] = piece.value
        free.discard(i)

    def pick(piece):
        cand = sorted(free)
        return rng.choices(cand, weights=[ROW_WEIGHTS[piece][i // 10] for i in cand])[0]

    flag = rng.choice(range(30, 40)) if rng.random() < 0.8 else rng.choice(range(20, 30))
    put(flag, P.FLAG)
    r, c = divmod(flag, 10)
    around = [rr * 10 + cc for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c))
              if 0 <= rr < 4 and 0 <= cc < 10]
    rng.shuffle(around)
    near = around[:rng.choice((2, 3, 3))]
    for i in near:
        put(i, P.BOMB)
    for _ in range(6 - len(near)):
        put(pick(P.BOMB), P.BOMB)
    for piece in ORDER:
        for _ in range(COUNTS[piece]):
            put(pick(piece), piece)
    assert not free
    return torch.tensor(setup, dtype=torch.long)


def random_setup(rng: random.Random) -> torch.Tensor:
    pieces = [p.value for p in all_pieces()]
    rng.shuffle(pieces)
    return torch.tensor(pieces, dtype=torch.long)


def diverse_setup_bank(count: int, seed: int = 0, shares: Tuple[float, float, float] = (0.4, 0.45, 0.15)) -> torch.Tensor:
    """(count, 40) setups: `shares` of heuristic, structured and random ones, shuffled."""
    rng = random.Random(seed)
    n_h = int(count * shares[0])
    n_s = int(count * shares[1])
    parts = [heuristic_setup_bank(n_h, seed=seed)]
    parts.append(torch.stack([structured_setup(rng) for _ in range(n_s)]))
    parts.append(torch.stack([random_setup(rng) for _ in range(count - n_h - n_s)]))
    bank = torch.cat(parts)
    assert bool(((bank > 0).sum(1) == 40).all())
    return bank[torch.randperm(count, generator=torch.Generator().manual_seed(seed))]
