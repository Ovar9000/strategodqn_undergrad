"""
The outlook expert (environment/expert.py) for a batch of games on the batched engine: the same
score for every legal move, computed for all games and all 3600 actions at once.

    scores = expert_scores(env)            # (N, 3600), -inf on illegal actions
    actions = choose(scores, noise=0.02)   # the expert's move for every game

test/test_vec_expert.py compares the scores with the original's, move by move.
"""

from typing import Optional

import torch

from environment.expert import PIECE_VALUE, ATTACK_CHANCE, CHEAP_RANK
from vec.engine import VecStratego, Tables, FLAG, BOMB, NUM_RANKS, NUM_SQUARES, NUM_ACTIONS

_cache = {}


def _consts(t: Tables):
    key = str(t.device)
    if key not in _cache:
        pv = torch.tensor(PIECE_VALUE, dtype=torch.float32, device=t.device)       # (12,) by rank index
        win, lose = t.win, t.lose                                                  # attacker x defender
        tie = 1.0 - win - lose
        tie[FLAG - 1] = 0.0
        tie[BOMB - 1] = 0.0
        row, col = t.row, t.col
        cheb = torch.maximum((row[:, None] - row[None, :]).abs(), (col[:, None] - col[None, :]).abs())
        _cache[key] = dict(pv=pv, win=win, lose=lose, tie=tie,
                           win_pv=win * pv[None, :], tie_pv=tie * pv[None, :],
                           pull_kernel=1.0 / (1.0 + t.manhattan), near=(cheb <= 3).float(),
                           row_from=row[t.act_from], row_to=row[t.act_to.clamp(max=NUM_SQUARES - 1)])
    return _cache[key]


class Style:
    """Proactive terms on top of the outlook expert's score (all 0 = the original expert).

    pursuit: quiet moves are drawn towards enemy pieces the moved piece can attack with positive
             expected value (weight per unit of expected gain, falling with distance squared).
    repel:   and pushed away from enemy pieces likely to beat it, beyond the adjacent square.
    probe:   attacking a hidden piece with a cheap piece (up to the Lieutenant) earns this much
             for the information, scaled by how uncertain the target's rank is.

    The remaining fields are the original expert's own constants, tunable: attack_chance (how
    likely a threatening piece is to attack, 0.6), clock (weight of the no-battle clock on an
    attack, 0.8), advance (0.06 per row), pull (scale of the pull towards the Flag, 1.0),
    power (exponent of the distance fall-off of pursuit / repel, 2.0).
    """

    FIELDS = ('pursuit', 'repel', 'probe', 'attack_chance', 'clock', 'advance', 'pull', 'power')

    def __init__(self, pursuit: float = 0.0, repel: float = 0.0, probe: float = 0.0, attack_chance: float = ATTACK_CHANCE,
                 clock: float = 0.8, advance: float = 0.06, pull: float = 1.0, power: float = 2.0):
        self.pursuit, self.repel, self.probe = pursuit, repel, probe
        self.attack_chance, self.clock, self.advance, self.pull, self.power = attack_chance, clock, advance, pull, power

    def active(self):
        return bool(self.pursuit or self.repel or self.probe)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}


DEFAULT_STYLE = Style()


def expert_scores(env: VecStratego, idx: Optional[torch.Tensor] = None, dist: Optional[torch.Tensor] = None,
                  detail: bool = False, style: Optional[Style] = None):
    """(B, 3600) score of every action for the side to move (-inf where illegal). `dist` is the
    (B, 12, 100) rank distribution from env.observe() if the caller already has it."""
    t = env.t
    c = _consts(t)
    style = style or DEFAULT_STYLE
    g = (lambda x: x) if idx is None else (lambda x: x[idx])
    own, opp, opp_rev, opp_mov, opp_long = g(env.own), g(env.opp), g(env.opp_rev), g(env.opp_mov), g(env.opp_long)
    legal, nb = g(env.legal), g(env.nb)
    B = own.size(0)
    if dist is None:
        from vec.engine import rank_distribution
        dist = rank_distribution(opp, opp_rev, opp_mov, opp_long, t)
    pv = c['pv']
    enemy = opp > 0
    hidden_unmoved = enemy & ~opp_rev & ~opp_mov
    p_flag = dist[:, FLAG - 1] * hidden_unmoved                                    # (B, 100)
    pressure = (nb.float() / env.no_battle_plies).clamp(max=1.0) if env.no_battle_plies else torch.zeros(B, device=own.device)
    endgame = enemy.sum(1) <= 15                                                   # (B,)
    own_flag = (own == FLAG).float().argmax(1)                                     # (B,)

    dist_t = dist.transpose(1, 2)                                                  # (B, 100, 12) by enemy rank
    # threat[b, x, r]: chance that the enemy piece on x beats an own piece of rank r by attacking it
    threat = dist_t @ c['win']                                                     # sum_e dist[e, x] win[e, r]
    threat = torch.cat([threat, torch.zeros(B, 1, NUM_RANKS, device=own.device)], 1)
    risk_sq = threat[:, t.neigh].amax(2)                                           # (B, 100, 12) strongest neighbour
    dn = dist_t / dist_t.sum(2, keepdim=True).clamp_min(1e-9)
    pw_sq = dn @ c['win'].t()                                                      # [x, a]: attacker a wins on x
    pl_sq = dn @ c['lose'].t()
    pt_sq = dn @ c['tie'].t()
    gain_sq = dn @ c['win_pv'].t()
    tieg_sq = dn @ c['tie_pv'].t()

    src = t.act_from.expand(B, -1)                                                 # (B, 3600)
    dst = t.act_to.clamp(max=NUM_SQUARES - 1).expand(B, -1)
    a = (own.gather(1, src) - 1).clamp_min(0)                                      # mover's rank index
    pva = pv[a]
    at = lambda table, sq: table.reshape(B, -1).gather(1, sq * NUM_RANKS + a)
    risk_s = at(risk_sq, src) * pva * style.attack_chance
    risk_d = at(risk_sq, dst) * pva * style.attack_chance
    attack = enemy.gather(1, dst)

    pw, pl, pt = at(pw_sq, dst), at(pl_sq, dst), at(pt_sq, dst)
    flag_dist = t.manhattan[own_flag]                                              # (B, 100) from own flag
    s_battle = at(gain_sq, dst) - pl * pva + at(tieg_sq, dst) - pt * pva
    s_exposed = -pw * risk_d + risk_s
    s_defend = 1.5 * pw * (flag_dist.gather(1, dst) <= 3)
    s_clock = style.clock * pressure[:, None].expand(-1, NUM_ACTIONS)
    s_hunt = 0.3 * pw * (hidden_unmoved.gather(1, dst) & endgame[:, None])
    attack_score = s_battle + s_exposed + s_defend + s_clock + s_hunt

    s_safety = risk_s - risk_d
    s_advance = style.advance * (c['row_from'] - c['row_to'])[None, :] * ((a <= CHEAP_RANK) | endgame[:, None])
    pull_map = p_flag @ c['pull_kernel']                                           # (B, 100)
    pull = style.pull * torch.where(endgame, 2.0, 0.6) * (1.0 + pressure)
    s_toward = pull[:, None] * (pull_map.gather(1, dst) - pull_map.gather(1, src))
    near = (enemy.float() * c['near'][own_flag]).sum(1) > 0                        # enemy within 3 of own flag
    s_guard = -1.0 * ((flag_dist.gather(1, src) == 1) & near[:, None])
    quiet_score = s_safety + s_advance + s_toward + s_guard

    if style.active():
        near2 = (1.0 / (1.0 + t.manhattan) ** style.power) * (t.manhattan > 0)
        ev_sq = (gain_sq - pl_sq * pv[None, None, :] + tieg_sq - pt_sq * pv[None, None, :])   # (B, 100, 12) [x, a]
        ev_sq = ev_sq * enemy[:, :, None]
        if style.pursuit:
            attract = near2 @ ev_sq.clamp_min(0.0)                            # (B, 100, 12) [sq, a]
            quiet_score = quiet_score + style.pursuit * (at(attract, dst) - at(attract, src))
        if style.repel:
            danger = near2 @ (threat[:, :NUM_SQUARES] * pv[None, None, :])
            quiet_score = quiet_score - style.repel * (at(danger, dst) - at(danger, src))
        if style.probe:
            hidden = enemy & ~opp_rev
            unsure = 1.0 - dn.amax(2)                                              # (B, 100)
            cheap = (pva <= 2.6).float()
            attack_score = attack_score + style.probe * cheap * (unsure * hidden).gather(1, dst)
    score = torch.where(attack, attack_score, quiet_score)
    score = score.masked_fill(~legal, float('-inf'))
    if detail:
        return score, dict(attack=attack, battle=s_battle, exposed=s_exposed, defend=s_defend, clock=s_clock,
                           hunt=s_hunt, safety=s_safety, advance=s_advance, toward=s_toward, guard=s_guard, pw=pw, pl=pl)
    return score


def choose(scores: torch.Tensor, noise: float = 0.02, temperature: float = 0.0) -> torch.Tensor:
    """The move for each game: the best score after adding noise * U(0, 1) (the original's tie
    break), or with temperature > 0 a sample from softmax(score / temperature)."""
    if temperature > 0:
        probs = torch.softmax(scores / temperature, dim=1)
        return torch.multinomial(probs, 1).squeeze(1)
    return (scores + noise * torch.rand_like(scores)).argmax(1)


class VecExpert:
    """Policy object with the arena's interface: act(env, idx) -> actions for games `idx`."""
    name = 'expert'

    def __init__(self, noise: float = 0.02, temperature: float = 0.0, epsilon: float = 0.0,
                 style: Optional[Style] = None, name: str = 'expert'):
        self.noise, self.temperature, self.epsilon, self.style, self.name = noise, temperature, epsilon, style, name

    @torch.no_grad()
    def act(self, env: VecStratego, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = expert_scores(env, idx, style=self.style)
        actions = choose(scores, self.noise, self.temperature)
        if self.epsilon > 0:
            legal = torch.isfinite(scores)
            rand = torch.multinomial(legal.float() + 1e-9 * (~legal.any(1, keepdim=True)), 1).squeeze(1)
            actions = torch.where(torch.rand(actions.size(0), device=actions.device) < self.epsilon, rand, actions)
        return actions


class VecRandom:
    name = 'random'

    @torch.no_grad()
    def act(self, env: VecStratego, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        legal = env.legal if idx is None else env.legal[idx]
        return torch.multinomial(legal.float() + 1e-9 * (~legal.any(1, keepdim=True)), 1).squeeze(1)
