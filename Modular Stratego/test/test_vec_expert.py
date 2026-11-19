"""The batched outlook expert (vec/expert.py) against the original (environment/expert.py): the
same score for every legal move, on positions from games played on both engines."""

import random

import torch

from environment import StrategoEnvironment
from environment.action_encoding import encode_move
from environment.expert import OutlookExpert
from environment.heuristic_setup import HeuristicSetupAgent
from vec.compat import placement_to_setup
from vec.engine import VecStratego
from vec.expert import expert_scores


def compare_scores(num_games: int, device, seed: int = 0, plies: int = 400, every: int = 3):
    """Play `num_games` games on both engines (the original expert's moves, some random ones) and
    compare the two experts' scores every `every` plies. Returns (positions, moves compared,
    largest difference)."""
    rng = random.Random(seed)
    random.seed(seed)
    cpu = torch.device('cpu')
    setups = {1: HeuristicSetupAgent(player_id=1), -1: HeuristicSetupAgent(player_id=-1)}
    vec = VecStratego(num_games, device)
    expert = OutlookExpert(cpu, noise=0.0)
    envs, states, ms, os_, starts = [], [], [], [], []
    for g in range(num_games):
        env = StrategoEnvironment(cpu)
        place = {p: setups[p].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(p))
                 for p in (1, -1)}
        start = 1 if g % 2 == 0 else -1
        states.append(env.reset(place[1], place[-1], starting_player=start))
        envs.append(env)
        ms.append(placement_to_setup(place[start], start))
        os_.append(placement_to_setup(place[-start], -start))
        starts.append(start)
    vec.reset(torch.arange(num_games, device=vec.device), torch.stack(ms), torch.stack(os_), torch.tensor(starts))
    positions = compared = 0
    worst = 0.0
    for ply in range(plies):
        if all(e.game_over for e in envs):
            break
        scores = expert_scores(vec).cpu()
        actions = torch.zeros(num_games, dtype=torch.long)
        for g, env in enumerate(envs):
            if env.game_over:
                continue
            state = states[g]
            moves = env.get_valid_moves()
            ranked = expert.explain(state, moves, top=10 ** 6)
            if ply % every == 0:
                positions += 1
                for move, _, terms in ranked:
                    want = sum(terms.values())
                    got = float(scores[g, encode_move(move, state.current_player)])
                    worst = max(worst, abs(want - got))
                    assert abs(want - got) < 1e-3, (g, ply, move, terms, got)
                    compared += 1
            move = rng.choice(moves) if rng.random() < 0.15 else ranked[0][0]
            actions[g] = encode_move(move, state.current_player)
            states[g], _, _, _ = env.step(move)
        vec.step(actions.to(vec.device))
    return positions, compared, worst


def test_scores_match_the_original_expert(cpu):
    positions, compared, worst = compare_scores(3, cpu, seed=3, plies=240, every=4)
    assert positions > 100 and compared > 1000 and worst < 1e-3
