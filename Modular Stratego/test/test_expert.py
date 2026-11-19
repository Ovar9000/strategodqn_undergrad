"""environment/expert.OutlookExpert: a scripted player on public information only."""
import random

import torch

from environment import StrategoEnvironment
from environment.expert import OutlookExpert
from environment.piece import PieceType as P


def position(cpu, p1, p2, player=1):
    env = StrategoEnvironment(cpu)
    env.board.reset()
    env._flag_positions = {1: None, -1: None}
    p1 = {(9, 0): P.FLAG, **p1}
    p2 = {(0, 9): P.FLAG, **p2}
    env.place_pieces(1, [(t, pos) for pos, t in p1.items()])
    env.place_pieces(-1, [(t, pos) for pos, t in p2.items()])
    env.current_player = player
    return env


def reveal(env, square):
    env.board.reveal_pieces(square, square)


def test_it_takes_a_revealed_piece_it_beats_and_leaves_one_that_beats_it(cpu):
    env = position(cpu, {(6, 4): P.CAPTAIN, (6, 7): P.SERGEANT}, {(5, 4): P.LIEUTENANT, (5, 7): P.GENERAL})
    reveal(env, (5, 4)); reveal(env, (5, 7))
    expert = OutlookExpert(cpu, player_id=1, noise=0.0)
    move = expert.act(env.get_state().board, env.get_valid_moves(), env.get_state())
    assert move == ((6, 4), (5, 4))
    terms = {m: t for m, _, t in expert.explain(env.get_state(), env.get_valid_moves(), top=20)}
    assert terms[((6, 7), (5, 7))]['battle'] < 0  # the Sergeant would lose to the General


def test_it_plays_legal_moves_for_both_sides_through_a_game(cpu):
    random.seed(0)
    env = StrategoEnvironment(cpu)
    state = env.reset()
    experts = {1: OutlookExpert(cpu, player_id=1, seed=1), -1: OutlookExpert(cpu, player_id=-1, seed=2)}
    for _ in range(200):
        moves = env.get_valid_moves()
        if not moves:
            break
        move = experts[state.current_player].act_batch([state.board], [moves], [state])[0]
        assert move in moves
        state, _, done, _ = env.step(move)
        if done:
            break
