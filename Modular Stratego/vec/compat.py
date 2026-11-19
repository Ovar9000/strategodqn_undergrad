"""Bridges between the batched engine (vec/engine.py) and the original environment."""

import random
from typing import List, Tuple

import torch

from environment.heuristic_setup import HeuristicSetupAgent
from environment.piece import PieceType


def all_pieces() -> List[PieceType]:
    """The 40 pieces of one side (environment._generate_pieces without the shuffle)."""
    return ([PieceType.FLAG, PieceType.SPY] + [PieceType.BOMB] * 6 + [PieceType.MARSHAL, PieceType.GENERAL]
            + [PieceType.COLONEL] * 2 + [PieceType.MAJOR] * 3 + [PieceType.CAPTAIN] * 4
            + [PieceType.LIEUTENANT] * 4 + [PieceType.SERGEANT] * 4 + [PieceType.MINER] * 5 + [PieceType.SCOUT] * 8)


def placement_to_setup(placement, player: int) -> torch.Tensor:
    """(40,) rank values for squares 60..99 of `player`'s canonical frame from an environment
    placement [(PieceType, (row, col)), ...]."""
    setup = torch.zeros(40, dtype=torch.long)
    for piece, (r, c) in placement:
        if player != 1:
            r, c = 9 - r, 9 - c
        setup[(r - 6) * 10 + c] = piece.value
    assert int((setup > 0).sum()) == 40
    return setup


def setup_to_placement(setup: torch.Tensor, player: int):
    """Inverse of placement_to_setup."""
    placement = []
    for i, v in enumerate(setup.tolist()):
        r, c = 6 + i // 10, i % 10
        if player != 1:
            r, c = 9 - r, 9 - c
        placement.append((PieceType(int(v)), (r, c)))
    return placement


def heuristic_setup_bank(count: int, seed: int = 0) -> torch.Tensor:
    """(count, 40) setups drawn from the thesis's heuristic setup agent."""
    state = random.getstate()
    random.seed(seed)
    agent = HeuristicSetupAgent(player_id=1)
    positions = [(r, c) for r in range(6, 10) for c in range(10)]
    bank = torch.zeros(count, 40, dtype=torch.long)
    for i in range(count):
        pieces = all_pieces()
        random.shuffle(pieces)
        bank[i] = placement_to_setup(agent.place_pieces(pieces, list(positions)), 1)
    random.setstate(state)
    return bank
