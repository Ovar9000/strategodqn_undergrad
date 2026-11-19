"""
Tune the scripted expert's style parameters by playing it against itself.

A hill climb: perturb the champion's parameters, play the candidate against the champion, keep it
when it wins clearly more games than it loses. Each accepted champion is also played against
the original outlook expert and the random player, as fixed yardsticks.

    python -m vec.tune_expert --out ../runs/expert_tune_1 --minutes 40
"""

import argparse
import json
import math
import os
import random
import time

import torch

from vec.arena import play_match, summary
from vec.expert import Style, VecExpert, VecRandom
from vec.setups import diverse_setup_bank

START = dict(pursuit=2.0, repel=0.5, probe=1.5, attack_chance=0.6, clock=0.8, advance=0.06, pull=1.0, power=2.0)
BOUNDS = dict(pursuit=(0.0, 20.0), repel=(0.0, 10.0), probe=(0.0, 10.0), attack_chance=(0.1, 1.0), clock=(0.0, 5.0),
              advance=(0.0, 0.5), pull=(0.1, 5.0), power=(0.5, 4.0))


def perturb(params, rng):
    new = dict(params)
    for k in rng.sample(sorted(params), rng.choice((1, 1, 2, 3))):
        lo, hi = BOUNDS[k]
        v = params[k] * math.exp(rng.gauss(0.0, 0.4)) if params[k] > 1e-6 else abs(rng.gauss(0.0, 0.5))
        new[k] = round(min(max(v, lo), hi), 4)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--minutes', type=float, default=40)
    ap.add_argument('--games', type=int, default=800)
    ap.add_argument('--margin', type=float, default=0.04, help='(wins - losses) / games needed to replace the champion')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    bank = diverse_setup_bank(20000, seed=args.seed).to(dev)
    path = os.path.join(args.out, 'champion.json')
    state = json.load(open(path)) if os.path.exists(path) else {'champion': START, 'history': []}
    champion = state['champion']
    log = open(os.path.join(args.out, 'log.txt'), 'a')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    def yardsticks(params, tag):
        e = VecExpert(style=Style(**params))
        r1 = play_match(e, VecExpert(), args.games, bank, dev)
        r2 = play_match(e, VecRandom(), args.games // 2, bank, dev)
        say(f"[tune] {tag} {params}\n       vs original expert: {summary(r1)}\n       vs random: {summary(r2)}")
        return r1, r2

    if not state['history']:
        r1, r2 = yardsticks(champion, 'start')
        state['history'].append({'params': champion, 'vs_expert': r1, 'vs_random': r2})
    start = time.time()
    trial = 0
    while time.time() - start < args.minutes * 60:
        trial += 1
        cand = perturb(champion, rng)
        r = play_match(VecExpert(style=Style(**cand)), VecExpert(style=Style(**champion)), args.games, bank, dev)
        net = (r['wins'] - r['losses']) / r['games']
        changed = {k: cand[k] for k in cand if cand[k] != champion[k]}
        say(f"[tune] trial {trial} {changed}: {r['wins']}/{r['losses']}/{r['draws']} net {net:+.3f}"
            f"{'  ACCEPT' if net > args.margin else ''}")
        if net > args.margin:
            champion = cand
            r1, r2 = yardsticks(champion, f'champion after trial {trial}')
            state['champion'] = champion
            state['history'].append({'params': champion, 'vs_expert': r1, 'vs_random': r2, 'trial': trial})
            json.dump(state, open(path, 'w'), indent=1)
    json.dump(state, open(path, 'w'), indent=1)
    say(f"[tune] done: {champion}")


if __name__ == '__main__':
    main()
