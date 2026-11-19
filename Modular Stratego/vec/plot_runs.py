"""
Training charts for the PPO runs (runs/ppo_*/log.txt), chained into one timeline.

    python -m vec.plot_runs                      # writes ../runs/plots/ppo_training.png
    python -m vec.plot_runs --every 120          # re-draws every 2 minutes while a run trains

Runs that started from another run's net (--init) are placed after it on the x axis (total
decisions trained since the clone); a run that resumed another run's state.pt already counts
from there. Lines are drawn from the [ppo] lines (every 5 iterations, training games) and the
[eval] lines (separate evaluation matches).
"""

import argparse
import math
import os
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from vec import health

PPO_RE = re.compile(r'^\[ppo\] it (\d+) samples ([+-]?(?:nan|inf|infinity|[\d.]+(?:[eE][+-]?\d+)?))M', re.IGNORECASE)
KV_RE = re.compile(r'\b(lr|pg|v|ent|kl|mkl|clip|ev|belief|len) ([+-]?(?:nan|inf|infinity|[\d.]+(?:[eE][+-]?\d+)?))', re.IGNORECASE)
TALLY_RE = re.compile(r'\b(self|expert|random|snap) (\d+)/(\d+)/(\d+)')
ENDS_RE = re.compile(r'ends flag (\d+) nomove (\d+) nobattle (\d+) limit (\d+)')
EVAL_RE = re.compile(r'^\[eval\] it (\d+) (greedy|sample) vs (\S+): (\d+)/(\d+)/(\d+) of (\d+)')
INIT_RE = re.compile(r"init='([^']*)'")
RESUME_RE = re.compile(r'^\[ppo\] resumed at iteration (\d+) \((\d+) samples')

# run -> the run it continued from, for runs whose log does not say (resumed state copies)
DEFAULT_PARENTS = {'ppo_2': 'ppo_1'}


def parse(path):
    """Parse one log.txt into {'train': [...], 'eval': [...], 'init': str|None, 'resumed': bool}."""
    run = {'train': [], 'eval': [], 'init': None, 'resumed': False}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if run['init'] is None:
                m = INIT_RE.search(line)
                if m:
                    run['init'] = m.group(1)
            if RESUME_RE.match(line):
                run['resumed'] = True
            m = PPO_RE.match(line)
            if m:
                row = {'it': int(m.group(1)), 'samples': float(m.group(2))}
                head, _, tail = line.partition('|')
                for k, v in KV_RE.findall(line):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
                for k, w, l, d in TALLY_RE.findall(line):
                    row['tally_' + k] = (int(w), int(l), int(d))
                e = ENDS_RE.search(line)
                if e:
                    row['ends'] = tuple(int(x) for x in e.groups())
                run['train'].append(row)
                continue
            m = EVAL_RE.match(line)
            if m:
                it, mode, name, w, l, d, g = m.groups()
                run['eval'].append({'it': int(it), 'mode': mode, 'name': name.rstrip(':'),
                                    'w': int(w), 'l': int(l), 'd': int(d), 'g': int(g)})
    return run


def load_runs(root):
    names = sorted((n for n in os.listdir(root) if re.fullmatch(r'ppo_\d+', n)
                    and os.path.exists(os.path.join(root, n, 'log.txt'))),
                   key=lambda n: int(n.split('_')[1]))
    runs = {n: parse(os.path.join(root, n, 'log.txt')) for n in names}
    # x offset of each run on the chained axis
    end = {}
    for n in names:
        r = runs[n]
        parent = None
        if r['init']:
            p = os.path.basename(os.path.dirname(os.path.normpath(r['init'])))
            parent = p if p in runs else None
        parent = parent or DEFAULT_PARENTS.get(n)
        if r['resumed'] or parent is None:
            off = 0.0 if not r['resumed'] else runs.get(parent, {}).get('offset', 0.0)
        else:
            off = end.get(parent, 0.0)
        r['offset'] = off
        r['parent'] = parent
        last = next((row['samples'] for row in reversed(r['train']) if math.isfinite(row['samples'])), 0.0)
        end[n] = off + last
        # samples at each iteration, for placing eval points
        r['it_x'] = {row['it']: off + row['samples'] for row in r['train'] if math.isfinite(row['samples'])}
    return runs


def x_of(run, it):
    """Chained x (M decisions) of iteration `it` (nearest logged iteration)."""
    if not run['it_x']:
        return run['offset']
    its = sorted(run['it_x'])
    best = min(its, key=lambda i: abs(i - it))
    per_it = (run['it_x'][its[-1]] - run['it_x'][its[0]]) / max(its[-1] - its[0], 1)
    return run['it_x'][best] + (it - best) * per_it


def score(w, l, d):
    n = w + l + d
    return 100.0 * (w + 0.5 * d) / n if n else float('nan')


COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948', '#b07aa1', '#9c755f']


def draw(runs, out, flags_by_run=None):
    if flags_by_run is None:
        flags_by_run = health.check_runs(runs)

    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    ax = axes.flatten()
    ax4b = ax[4].twinx()
    ax7b = ax[7].twinx()
    titles = [
        'Evaluation: sampled play vs scripted experts (win %)',
        'Evaluation: vs its own earlier nets (win %, 50 = equal)',
        'Training games: score vs league snapshots and experts (%)',
        'Policy entropy (lower = more decided)',
        'Update size: KL per update and clip fraction',
        'Value head: explained variance of returns',
        'Belief head: rank accuracy on hidden enemy pieces',
        'Training games: mean length (plies) and no-battle draws per 100 games',
    ]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=11, loc='left')
        a.grid(alpha=0.3)
        a.set_xlabel('decisions trained since the clone (millions)')

    eval_styles = {('sample', 'expert'): ('-', 'o', 'original expert'),
                   ('sample', 'expert2'): ('--', 's', 'expert2 (proactive)'),
                   ('sample', 'expert3'): (':', '^', 'expert3 (tuned)'),
                   ('greedy', 'expert'): ('-.', 'x', 'original expert, greedy')}
    legend_done = set()
    min_ev = 0.0
    for k, (name, r) in enumerate(runs.items()):
        c = COLORS[k % len(COLORS)]
        tr = r['train']
        xs = [r['offset'] + row['samples'] for row in tr if math.isfinite(row['samples'])]
        if xs:
            for a in ax:
                a.axvspan(xs[0], xs[-1], color=c, alpha=0.05, lw=0)
            ax[0].text(xs[0], 101, name, color=c, fontsize=9, va='bottom')

        # 0: evals vs experts
        for (mode, opp), (ls, mk, label) in eval_styles.items():
            pts = [(x_of(r, e['it']), 100.0 * e['w'] / e['g']) for e in r['eval']
                   if e['mode'] == mode and e['name'] == opp and e['g'] > 0]
            if pts:
                lab = label if label not in legend_done else None
                legend_done.add(label)
                ax[0].plot(*zip(*pts), ls=ls, marker=mk, color=c, label=lab, ms=5)
        # 1: vs earlier nets
        for opp, mk in (('start', 'o'), ('ref', 's')):
            pts = [(x_of(r, e['it']), 100.0 * e['w'] / e['g']) for e in r['eval']
                   if e['mode'] == 'sample' and e['name'] == opp and e['g'] > 0]
            if pts:
                # 'start' is the league's first slot: the clone for ppo_1 and runs resumed from it,
                # the --init net otherwise
                if opp == 'start':
                    who = 'the clone' if (r['resumed'] or not r['parent']) else f'its start ({r["parent"]})'
                else:
                    who = f'ref ({r["parent"] or "?"})'
                ax[1].plot(*zip(*pts), marker=mk, color=c, ls='-' if opp == 'start' else '--',
                           label=f'{name} vs {who}')
        # 2: training tallies
        for key, ls, lab in (('tally_snap', '-', 'vs league snapshots'), ('tally_expert', ':', 'vs experts')):
            pts = [(r['offset'] + row['samples'], score(*row[key])) for row in tr
                   if key in row and sum(row[key]) > 0 and math.isfinite(row['samples'])]
            if pts:
                ax[2].plot(*zip(*pts), ls=ls, color=c, lw=1.2,
                           label=lab if k == len(runs) - 1 else None)
        # 3-6: training stats
        def line(a, key, **kw):
            pts = [(r['offset'] + row['samples'], row[key]) for row in tr
                   if key in row and math.isfinite(row['samples']) and math.isfinite(row[key])]
            if pts:
                a.plot(*zip(*pts), color=c, **kw)
            return pts
        line(ax[3], 'ent', label=name)
        line(ax[4], 'kl', label=f'{name} KL')
        line(ax4b, 'clip', ls=':', label=f'{name} clip')
        ev_pts = line(ax[5], 'ev', label=name)
        for _, ev_val in ev_pts:
            if ev_val < min_ev:
                min_ev = ev_val
        line(ax[6], 'belief', label=name)
        line(ax[7], 'len', label=f'{name} length')
        pts = []
        for row in tr:
            if 'ends' in row and math.isfinite(row['samples']):
                games = sum(row['ends'])
                if games:
                    pts.append((r['offset'] + row['samples'], 100.0 * row['ends'][2] / games))
        if pts:
            ax7b.plot(*zip(*pts), color=c, ls=':', lw=1)

    ax[0].set_ylim(0, 105)
    ax[1].axhline(50, color='grey', lw=1)
    ax[1].set_ylim(0, 100)
    ax[2].axhline(50, color='grey', lw=1)
    ax[2].set_ylim(0, 100)
    ax[3].axhline(health.DEFAULT_THRESHOLDS['ent_min'], color='#d62728', ls='--', lw=0.8, alpha=0.4)
    ax[4].set_ylabel('KL (solid)')
    ax4b.set_ylabel('clip fraction (dotted)')
    ax[5].axhline(health.DEFAULT_THRESHOLDS['ev_min'], color='#d62728', ls='--', lw=0.8, alpha=0.4)
    ax[5].set_ylim(min(0.0, min_ev - 0.06), 1.0)
    ax[7].set_ylabel('plies (solid)')
    ax7b.set_ylabel('no-battle draws per 100 games (dotted)')

    # Mark health-flagged points in red on the corresponding panels
    panel_axes = {
        'eval_expert': ax[0],
        'ent': ax[3],
        'kl': ax[4],
        'clip': ax4b,
        'ev': ax[5],
        'belief': ax[6],
        'len': ax[7],
        'nobattle': ax7b,
    }
    finite_by_panel = {p: [] for p in panel_axes}
    nonfinite_by_panel = {p: [] for p in panel_axes}
    total_flags = 0
    for rflags in flags_by_run.values():
        for fl in rflags:
            total_flags += 1
            if fl.panel not in panel_axes:
                continue
            if math.isfinite(fl.x) and math.isfinite(fl.value):
                finite_by_panel[fl.panel].append((fl.x, fl.value))
            elif math.isfinite(fl.x):
                nonfinite_by_panel[fl.panel].append(fl.x)

    for pname, pts in finite_by_panel.items():
        if pts:
            fxs, fys = zip(*pts)
            panel_axes[pname].scatter(
                fxs, fys, color='#ff0000', edgecolors='#550000', linewidths=0.8,
                s=58, zorder=10, label='health flag',
            )
    for pname, fxs in nonfinite_by_panel.items():
        if fxs:
            lo, hi = panel_axes[pname].get_ylim()
            y_mark = lo + 0.92 * (hi - lo)
            panel_axes[pname].scatter(
                fxs, [y_mark] * len(fxs), color='#ff0000', marker='X', edgecolors='#550000',
                linewidths=0.8, s=78, zorder=11, label='NaN/inf flag',
            )

    for a, loc in (
        (ax[0], 'lower right'), (ax[1], 'lower right'), (ax[2], 'lower right'),
        (ax[3], 'best'), (ax[4], 'upper right'), (ax[5], 'lower right'),
        (ax[6], 'lower right'), (ax[7], 'upper center'), (ax7b, 'upper right'),
    ):
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8, loc=loc)

    fig.suptitle(
        'PPO self-play: ' + ' → '.join(runs)
        + f'   ({total_flags} health flag{"s" if total_flags != 1 else ""}, drawn {time.strftime("%Y-%m-%d %H:%M")})',
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return flags_by_run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join('..', 'runs'))
    ap.add_argument('--out', default=os.path.join('..', 'runs', 'plots', 'ppo_training.png'))
    ap.add_argument('--health-out', default=None,
                    help='path for health.txt report (default: health.txt next to --out)')
    ap.add_argument('--every', type=float, default=0, help='re-draw every N seconds (0 = once)')
    ap.add_argument('--hours', type=float, default=3.0, help='with --every: stop after this long')
    args = ap.parse_args(argv)
    health_out = args.health_out or os.path.join(os.path.dirname(args.out) or '.', 'health.txt')
    t0 = time.time()
    while True:
        runs = load_runs(args.runs)
        flags = health.write_report(runs, health_out)
        draw(runs, args.out, flags_by_run=flags)
        n_flags = sum(len(v) for v in flags.values())
        print(f'[plot] {len(runs)} runs ({n_flags} flags) -> {args.out}, {health_out}', flush=True)
        if args.every <= 0 or time.time() - t0 > args.hours * 3600:
            break
        time.sleep(args.every)


if __name__ == '__main__':
    main()
