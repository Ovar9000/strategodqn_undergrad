"""
OutlookExpert: a one-ply scripted player that reads only the public information, through the
same observation planes the network sees (environment/observation.py) and the rank distribution
of every enemy piece (battle_outlook.enemy_rank_distribution).

Each legal move gets a score in units of piece value (PIECE_VALUE), and the best one is played
(ties broken at random):

- An attack: the expected value of the battle over the target's rank distribution: + the
  target's value when it wins (the Flag ends the game), - the attacker's when it loses, the
  difference for a tie; then, when the attacker would stand on the target square, the risk of
  being taken there.
- Every move: the risk the moved piece runs on its new square (the strongest adjacent enemy's
  chance to beat it, times its value, times ATTACK_CHANCE) against the risk it ran where it was.
- Advance: cheap pieces move forward; pieces move toward the hidden enemy pieces most likely to
  be the Flag (unmoved ones), more so as the no-battle counter runs and in the endgame.
- Own Flag: an enemy piece near it is worth attacking more, and a guard next to it stays.

It is the demonstrator and an opponent for the network (Appendix C). Its reasons are the terms of
the score, so a move can be explained by them (explain()).
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from environment.action_encoding import canonical_coords
from environment.battle_outlook import WIN, LOSE, enemy_rank_distribution
from environment.observation import (CH_OWN, CH_ENEMY_HIDDEN, CH_ENEMY_MOVED, CH_NO_BATTLE,
                                     encode_observations)
from environment.piece import PieceType as P

NUM_RANKS = 12
# What a piece is worth to its side, by PieceType value - 1. The Spy is worth more than its rank
# (it takes the Marshal), Miners more than theirs (they clear Bombs). The Flag ends the game, but its
# probability on a hidden piece is spread evenly over the unmoved ones, so its value is kept at
# 25: a probe early on is worth little, the last few unmoved pieces a lot.
PIECE_VALUE = np.zeros(NUM_RANKS)
for _p, _v in ((P.FLAG, 25.0), (P.SPY, 3.0), (P.SCOUT, 1.2), (P.MINER, 3.0), (P.SERGEANT, 2.0),
               (P.LIEUTENANT, 2.6), (P.CAPTAIN, 3.4), (P.MAJOR, 4.6), (P.COLONEL, 6.0), (P.GENERAL, 8.0),
               (P.MARSHAL, 11.0), (P.BOMB, 1.0)):
    PIECE_VALUE[_p.value - 1] = _v
FLAG = P.FLAG.value - 1
IMMOBILE = (P.FLAG.value - 1, P.BOMB.value - 1)
WIN_NP, LOSE_NP = WIN.numpy(), LOSE.numpy()
TIE_NP = 1.0 - WIN_NP - LOSE_NP
for _a in IMMOBILE:
    TIE_NP[_a] = 0.0

ATTACK_CHANCE = 0.6     # how likely a threatening enemy piece is to attack next move
CHEAP_RANK = P.CAPTAIN.value - 1  # pieces up to the Captain advance freely
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class OutlookExpert:
    """Scripted player (see the module docstring); act_batch has the opponents' interface."""

    def __init__(self, device, player_id: int = -1, noise: float = 0.02, seed: Optional[int] = None):
        self.device = device
        self.player_id = player_id
        self.noise = noise
        self.name = "OutlookExpert"
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ interface
    def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
        items = game_states if game_states and all(g is not None for g in game_states) else boards
        players = [getattr(g, 'current_player', self.player_id) for g in items]
        obs = encode_observations(list(items), players, device=torch.device('cpu'))
        dist = enemy_rank_distribution(obs, [getattr(g, 'hidden_pool', None) for g in items]).numpy()
        obs = obs.numpy()
        return [self._choose(obs[i], dist[i], moves, players[i])[0] if moves else None
                for i, moves in enumerate(valid_moves_list)]

    def act(self, board, valid_moves, game_state=None, **kwargs):
        return self.act_batch([board], [valid_moves], [game_state] if game_state is not None else None)[0]

    def explain(self, game_state, valid_moves, top: int = 5) -> List[Tuple[tuple, float, Dict[str, float]]]:
        """The `top` best moves with their scores and score terms, best first."""
        player = game_state.current_player
        obs = encode_observations([game_state], [player], device=torch.device('cpu'))
        dist = enemy_rank_distribution(obs, [getattr(game_state, 'hidden_pool', None)]).numpy()[0]
        _, scored = self._choose(obs.numpy()[0], dist, valid_moves, player, keep_terms=True)
        return sorted(scored, key=lambda x: -x[1])[:top]

    def reset_pbs(self):
        pass

    def update_pbs_batch(self, *args, **kwargs):
        pass

    # ------------------------------------------------------------------ scoring
    def _choose(self, obs, dist, moves, player, keep_terms=False):
        own_onehot = obs[CH_OWN:CH_OWN + NUM_RANKS]                       # (12, H, W), canonical
        own_rank = np.where(own_onehot.sum(0) > 0, own_onehot.argmax(0), -1)
        enemy = dist.sum(0) > 1e-6                                          # enemy piece present
        hidden_unmoved = (obs[CH_ENEMY_HIDDEN] > 0.5) & (obs[CH_ENEMY_MOVED] < 0.5)
        p_flag = dist[FLAG] * hidden_unmoved                                # where the Flag may be
        pressure = float(obs[CH_NO_BATTLE, 0, 0])                           # share of the no-battle limit used
        n_enemy = int(enemy.sum())
        endgame = n_enemy <= 15
        flag_sq = np.argwhere(own_rank == FLAG)
        own_flag = tuple(flag_sq[0]) if len(flag_sq) else None
        flag_targets = np.argwhere(p_flag > 0)
        flag_weights = p_flag[p_flag > 0]

        def risk(r, c, rank, ignore=None):
            """Expected loss of own piece `rank` standing on (r, c): the strongest adjacent enemy."""
            worst = 0.0
            for dr, dc in NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and enemy[nr, nc] and (nr, nc) != ignore:
                    worst = max(worst, float(dist[:, nr, nc] @ WIN_NP[:, rank]))
            return worst * PIECE_VALUE[rank] * ATTACK_CHANCE

        def flag_pull(r, c):
            if not len(flag_targets):
                return 0.0
            d = np.abs(flag_targets - np.array([r, c])).sum(1)
            return float((flag_weights / (1.0 + d)).sum())

        best, best_score, scored = None, -1e18, []
        for move in moves:
            (fr, fc), (tr, tc) = move
            r0, c0 = canonical_coords(fr, fc, player)
            r1, c1 = canonical_coords(tr, tc, player)
            a = int(own_rank[r0, c0])
            if a < 0:
                continue
            terms = {}
            if enemy[r1, c1]:
                d = dist[:, r1, c1] / max(dist[:, r1, c1].sum(), 1e-9)
                pw, pl, pt = float(d @ WIN_NP[a]), float(d @ LOSE_NP[a]), float(d @ TIE_NP[a])
                gain = float((d * WIN_NP[a]) @ PIECE_VALUE)                      # what it takes when it wins
                tie_gain = float((d * TIE_NP[a]) @ PIECE_VALUE)
                terms['battle'] = gain - pl * PIECE_VALUE[a] + tie_gain - pt * PIECE_VALUE[a]
                terms['exposed'] = -pw * risk(r1, c1, a, ignore=None) + risk(r0, c0, a)
                if own_flag is not None and abs(r1 - own_flag[0]) + abs(c1 - own_flag[1]) <= 3:
                    terms['defend flag'] = 1.5 * pw
                terms['no-battle clock'] = 0.8 * pressure
                if hidden_unmoved[r1, c1] and endgame:
                    terms['flag hunt'] = 0.3 * pw
            else:
                terms['safety'] = risk(r0, c0, a) - risk(r1, c1, a, ignore=(r0, c0))
                if a <= CHEAP_RANK or endgame:
                    terms['advance'] = 0.06 * (r0 - r1)
                pull = (2.0 if endgame else 0.6) * (1.0 + pressure)
                terms['toward flag'] = pull * (flag_pull(r1, c1) - flag_pull(r0, c0))
                if own_flag is not None and abs(r0 - own_flag[0]) + abs(c0 - own_flag[1]) == 1:
                    near = any(enemy[x, y] for x in range(max(0, own_flag[0] - 3), min(10, own_flag[0] + 4))
                               for y in range(max(0, own_flag[1] - 3), min(10, own_flag[1] + 4)))
                    if near:
                        terms['guard flag'] = -1.0
            score = sum(terms.values()) + self.noise * self.rng.random()
            if keep_terms:
                scored.append((move, score, terms))
            if score > best_score:
                best, best_score = move, score
        if best is None:
            best = moves[0]
        return best, scored
