"""
Reviewer check of a "net A is stronger than net B" claim, on seeds the claimant never saw.

    python -m vec.verify A.pt B.pt [--games 1000] [--seeds 41 42] [--primary sampled_raw]
                         [--modes sampled_raw greedy_raw sampled_safe greedy_safe] [--out res.json]

By default it draws two fresh setup-bank seeds from the OS (printed and saved, so the run can be
repeated) instead of the seeds 7 and 8 that every earlier report used. For each mode it plays
`--games` colour- and first-move-balanced games per seed (vec.ladder.play_match_balanced) and
reports A's score (wins + draws/2) per seed and pooled, with a z-score whose standard error
accounts for draws.

The PRIMARY mode is fixed before the games are played (default: sampled, no safety filter). Only
it decides the verdict; the other modes are shown for context, with Holm-adjusted p-values, so
"the best of four comparisons" cannot be quoted as the result. The script also flags seeds that
disagree by more than 2.5 standard errors, and checks the web-model replacement rule
(>= 53% on every seed in the primary mode).
"""
import argparse
import json
import math
import os
import secrets
import time
from typing import Dict, List, Sequence

MODES = {  # name -> (temperature, safety filter on both sides)
    'sampled_raw': (1.0, False), 'greedy_raw': (0.0, False),
    'sampled_safe': (1.0, True), 'greedy_safe': (0.0, True),
}
USED_SEEDS = (7, 8)  # the seeds earlier reports were tuned and judged on


def score_stats(wins: int, losses: int, draws: int) -> Dict[str, float]:
    """Score rate, its standard error and z against 50%. Each game scores 1, 1/2 or 0, so the
    per-game variance is E[x^2] - s^2 = (w + d/4)/n - s^2 (smaller than s(1-s) when there are draws)."""
    n = wins + losses + draws
    if n <= 0:
        raise ValueError('no games')
    s = (wins + 0.5 * draws) / n
    var = max((wins + 0.25 * draws) / n - s * s, 1e-12)
    se = math.sqrt(var / n)
    return {'games': n, 'wins': wins, 'losses': losses, 'draws': draws,
            'score': s, 'se': se, 'z': (s - 0.5) / se}


def p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=pvalues.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def seed_disagreement(per_seed: Sequence[Dict[str, float]]) -> float:
    """Largest pairwise gap between seeds, in standard errors of the difference."""
    worst = 0.0
    for i in range(len(per_seed)):
        for j in range(i + 1, len(per_seed)):
            a, b = per_seed[i], per_seed[j]
            worst = max(worst, abs(a['score'] - b['score']) / math.hypot(a['se'], b['se']))
    return worst


def summarize(raw: Dict[str, List[Dict[str, int]]], primary: str, z_crit: float = 1.96,
              replace_min: float = 0.53) -> Dict[str, object]:
    """raw: mode -> list of per-seed {'wins','losses','draws'}. Pure function (unit-tested)."""
    if primary not in raw:
        raise ValueError(f'primary mode {primary!r} was not played')
    modes = {}
    for mode, seeds in raw.items():
        per = [score_stats(r['wins'], r['losses'], r['draws']) for r in seeds]
        tot = score_stats(*(sum(r[k] for r in seeds) for k in ('wins', 'losses', 'draws')))
        modes[mode] = {'per_seed': per, 'pooled': tot, 'p': p_two_sided(tot['z']),
                       'seed_disagreement_se': seed_disagreement(per)}
    adj = holm({m: v['p'] for m, v in modes.items()})
    for m in modes:
        modes[m]['p_holm'] = adj[m]
    prim = modes[primary]
    z = prim['pooled']['z']
    verdict = 'A stronger' if z >= z_crit else ('B stronger' if z <= -z_crit else 'no clear difference')
    warnings = [f'{m}: seeds disagree by {v["seed_disagreement_se"]:.1f} SE (> 2.5)'
                for m, v in modes.items() if v['seed_disagreement_se'] > 2.5]
    return {'primary': primary, 'verdict': verdict, 'primary_z': z,
            'replacement_rule_53_every_seed': all(s['score'] >= replace_min for s in prim['per_seed']),
            'modes': modes, 'warnings': warnings}


def fresh_seeds(k: int, avoid: Sequence[int] = USED_SEEDS) -> List[int]:
    out: List[int] = []
    while len(out) < k:
        s = 1000 + secrets.randbelow(10 ** 6)
        if s not in avoid and s not in out:
            out.append(s)
    return out


def play(a_path: str, b_path: str, modes: Sequence[str], seeds: Sequence[int], games: int,
         bank_size: int, device) -> Dict[str, List[Dict[str, int]]]:
    import torch
    from vec.ladder import play_match_balanced
    from vec.net import NetPolicy, load
    from vec.setups import diverse_setup_bank

    na, nb = load(a_path, device).eval(), load(b_path, device).eval()
    raw: Dict[str, List[Dict[str, int]]] = {}
    t0 = time.time()
    for mode in modes:
        temp, safe = MODES[mode]
        raw[mode] = []
        for seed in seeds:
            torch.manual_seed(seed * 7919 + (10 if temp > 0 else 0) + (1 if safe else 0))
            bank = diverse_setup_bank(bank_size, seed=seed).to(device)
            r = play_match_balanced(NetPolicy(na, temperature=temp, safe=safe),
                                    NetPolicy(nb, temperature=temp, safe=safe), games, bank, device)
            raw[mode].append({'seed': seed, 'wins': int(r['wins']), 'losses': int(r['losses']),
                              'draws': int(r['draws'])})
            print(f'[verify] {mode:12s} seed {seed}: {r["wins"]}/{r["losses"]}/{r["draws"]} '
                  f'score {100 * r["score_rate"]:.2f}%  ({time.time() - t0:.0f}s)', flush=True)
    return raw


def report(res: Dict[str, object]) -> str:
    lines = []
    for m, v in res['modes'].items():
        tag = '  <- PRIMARY' if m == res['primary'] else ''
        seeds = ' / '.join(f'{100 * s["score"]:.2f}' for s in v['per_seed'])
        p = v['pooled']
        lines.append(f'{m:12s} seeds {seeds}  pooled {100 * p["score"]:.2f}% '
                     f'(+-{196 * p["se"]:.2f}, z {p["z"]:+.2f}, Holm p {v["p_holm"]:.3g}){tag}')
    lines.append(f'verdict ({res["primary"]}): {res["verdict"]} (z {res["primary_z"]:+.2f}); '
                 f'web-model rule >= 53% every seed: {res["replacement_rule_53_every_seed"]}')
    lines += [f'WARNING {w}' for w in res['warnings']]
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--games', type=int, default=1000, help='games per seed per mode (multiple of 4)')
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='default: 2 fresh OS-random seeds')
    ap.add_argument('--modes', nargs='+', default=list(MODES), choices=list(MODES))
    ap.add_argument('--primary', default='sampled_raw', choices=list(MODES))
    ap.add_argument('--bank-size', type=int, default=20000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args(argv)
    import torch
    if args.games % 4:
        ap.error('--games must be a multiple of 4 (colour x first-move balance)')
    modes = list(dict.fromkeys([args.primary] + args.modes))
    seeds = args.seeds or fresh_seeds(2)
    if set(seeds) & set(USED_SEEDS):
        print(f'[verify] note: seeds {sorted(set(seeds) & set(USED_SEEDS))} were used by earlier reports')
    dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[verify] A={args.a}\n[verify] B={args.b}\n[verify] seeds {seeds}, {args.games} games/seed, '
          f'primary {args.primary}', flush=True)
    raw = play(args.a, args.b, modes, seeds, args.games, args.bank_size, dev)
    res = summarize(raw, args.primary)
    res.update({'a': args.a, 'b': args.b, 'seeds': seeds, 'games_per_seed': args.games, 'raw': raw})
    print(report(res), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == '__main__':
    main()
