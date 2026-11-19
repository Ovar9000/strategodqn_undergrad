"""
Evaluate a policy-value network in the ORIGINAL environment, against the thesis's opponents
(random, heuristic, smart heuristic, outlook expert), with the thesis's evaluation routine
(network/training/evaluation.play_evaluation_games), so the numbers compare with Appendix C.

    python -m vec.eval_original --net ../runs/ppo_1/net_final.pt --games 200
"""

import argparse
import json

import torch

from environment.action_encoding import canonical_coords, decode_action, legal_action_indices
from environment.curriculum import HeuristicOpponent, SmartHeuristicOpponent, TrueRandomOpponent
from environment.expert import OutlookExpert
from environment.observation import encode_observations
from network.training.evaluation import play_evaluation_games
from vec.engine import INITIAL_COUNTS, NUM_OBS_PLANES
from vec.net import load


def full_observation(states, players) -> torch.Tensor:
    """(B, 59, 10, 10): the thesis's 41 planes from environment/observation.py plus the extra
    public-information planes of vec/engine.py, computed from the original GameState."""
    base = encode_observations(list(states), players, device=torch.device('cpu'))
    B = base.size(0)
    obs = torch.zeros(B, NUM_OBS_PLANES, 10, 10)
    obs[:, :41] = base
    initial = torch.tensor(INITIAL_COUNTS[1:], dtype=torch.float32)
    for i, (st, player) in enumerate(zip(states, players)):
        pool = getattr(st, 'hidden_pool', None)
        if pool is not None:
            obs[i, 41:53] = (torch.as_tensor(pool).float() / initial).view(12, 1, 1)
        own = base[i, 0:12].sum(0) > 0                                   # canonical frame
        flip = (lambda x: x) if player == 1 else (lambda x: x.flip(-2, -1))
        if getattr(st, 'moved', None) is not None:
            obs[i, 53] = (flip(torch.as_tensor(st.moved).bool().cpu()) & own).float()
            obs[i, 54] = (flip(torch.as_tensor(st.long_moved).bool().cpu()) & own).float()
        history = getattr(st, 'move_history', None) or []
        for j, plane in ((1, 55), (2, 57)):                              # enemy's last move, own last move
            if len(history) >= j and history[-j] is not None:
                (r0, c0), (r1, c1) = history[-j]
                a, b = canonical_coords(int(r0), int(c0), player), canonical_coords(int(r1), int(c1), player)
                obs[i, plane, a[0], a[1]] = 1.0
                obs[i, plane + 1, b[0], b[1]] = 1.0
    return obs


class NetAgent:
    """The network with the opponents' act_batch interface (greedy, or sampling at `temperature`)."""

    def __init__(self, net, device, temperature: float = 0.0):
        self.net, self.device, self.temperature = net.eval(), device, temperature
        self.noisy_acting = False

    @torch.no_grad()
    def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
        items = game_states if game_states and all(g is not None for g in game_states) else boards
        players = [getattr(g, 'current_player', 1) for g in items]
        obs = full_observation(list(items), players).to(self.device)
        with torch.autocast('cuda', enabled=obs.is_cuda):
            logits, _ = self.net(obs.contiguous(memory_format=torch.channels_last))
        logits = logits.float().cpu()
        out = []
        for i, moves in enumerate(valid_moves_list):
            if not moves:
                out.append(None)
                continue
            idx = torch.tensor(legal_action_indices(moves, players[i]))
            lg = logits[i, idx]
            j = int(torch.multinomial(torch.softmax(lg / self.temperature, 0), 1)) if self.temperature > 0 else int(lg.argmax())
            out.append(decode_action(int(idx[j]), players[i]))
        return out


def evaluate(net, device, games: int = 200, opponents=('random', 'heuristic', 'smart_heuristic', 'expert'),
             temperature: float = 0.0):
    cpu = torch.device('cpu')
    make = {'random': lambda: TrueRandomOpponent(cpu, player_id=-1),
            'heuristic': lambda: HeuristicOpponent(device=cpu, player_id=-1),
            'smart_heuristic': lambda: SmartHeuristicOpponent(device=cpu, player_id=-1),
            'expert': lambda: OutlookExpert(cpu, player_id=-1)}
    agent = NetAgent(net, device, temperature)
    return {name: play_evaluation_games(agent, make[name](), games, cpu) for name in opponents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--opponents', default='random,heuristic,smart_heuristic,expert')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = evaluate(load(args.net, dev), dev, args.games, tuple(args.opponents.split(',')), args.temperature)
    for name, r in res.items():
        lost = r['revealed_lost'] / max(1, r['revealed_attacks'])
        print(f"vs {name}: {r['wins']}/{r['losses']}/{r['draws']} of {r['games']} "
              f"({100 * r['win_rate']:.1f}% wins), revealed attacks lost {100 * lost:.0f}% of {r['revealed_attacks']}", flush=True)
    if args.json:
        json.dump(res, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
