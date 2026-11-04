"""The evaluation games count the agent's attacks on enemy pieces whose rank it can see."""
import torch

from environment.board import HIDDEN_PIECE
from environment.piece import PieceType
from environment.battle import BattleResolver
from network import RandomAgent
from network.training.evaluation import _count_revealed_attack, play_evaluation_games


def _counts():
    return {'revealed_attacks': 0, 'revealed_won': 0, 'revealed_lost': 0}


def test_revealed_attack_counted_by_battle_rule():
    view = torch.zeros(10, 10, dtype=torch.long)
    view[5, 5] = PieceType.GENERAL.value
    view[4, 5] = -PieceType.MARSHAL.value  # known enemy Marshal: the General loses
    view[6, 5] = -PieceType.SCOUT.value    # known enemy Scout: the General wins
    view[5, 4] = HIDDEN_PIECE              # unseen piece: not counted
    view[5, 6] = 0                         # empty square: not an attack
    c, br = _counts(), BattleResolver()
    for dst in ((4, 5), (6, 5), (5, 4), (5, 6)):
        _count_revealed_attack(view, ((5, 5), dst), br, c)
    assert c == {'revealed_attacks': 2, 'revealed_won': 1, 'revealed_lost': 1}


def test_evaluation_reports_revealed_attacks():
    res = play_evaluation_games(RandomAgent(), RandomAgent(), 2, torch.device('cpu'), max_turns=200)
    assert {'revealed_attacks', 'revealed_won', 'revealed_lost'} <= set(res)
    assert res['revealed_won'] + res['revealed_lost'] <= res['revealed_attacks']
