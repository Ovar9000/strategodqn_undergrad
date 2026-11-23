"""
Health checks for PPO training runs (runs/ppo_*/log.txt).

Called by vec/plot_runs.py (never inside vec/ppo.py) to flag training anomalies,
write runs/plots/health.txt, and supply flagged (x, y) coordinates to highlight
in red on runs/plots/ppo_training.png.

Can also be run standalone:
    python -m vec.health --runs ../runs --out ../runs/plots/health.txt

Checks implemented (all thresholds configurable via DEFAULT_THRESHOLDS):
  1. nan_inf          : NaN or +/-inf in any logged numeric metric (train or eval).
  2. entropy_collapse : Policy entropy falls below 0.5 (ent < 0.5).
  3. kl_spike         : KL divergence exceeds 3x the running mean of the previous
                        up to 10 positive finite KL updates (after >= 3 prior
                        positive updates, with floor kl > 1e-4 so warmup zeros do
                        not false-positive).
  4. ev_drop          : Value-head explained variance falls below 0.5 (ev < 0.5).
  5. eval_regression  : Evaluation win % vs any scripted expert (name starting
                        with 'expert', tracked per (mode, opponent) series within
                        each run) drops more than 2.5 x SE of the difference below
                        the pooled moving average of the prior up to 3 evaluations
                        in that series.
  6. nobattle_surge   : Share of finished training games ending in no-battle draws
                        (100 * nobattle / sum(ends), requiring >= 20 finished games)
                        reaches >= 5.0% OR (>= 3.0% AND > 2.5x the running mean of
                        the previous up to 10 logged points, after >= 3 prior points).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_THRESHOLDS: Dict[str, float] = {
    'ent_min': 0.5,
    'kl_mult': 3.0,
    'kl_window': 10,
    'kl_min_history': 3,
    'kl_floor': 1e-4,
    'ev_min': 0.5,
    'eval_z': 2.5,
    'eval_window': 3,
    'nobattle_abs_pct': 5.0,
    'nobattle_rel_min_pct': 3.0,
    'nobattle_rel_mult': 2.5,
    'nobattle_window': 10,
    'nobattle_min_history': 3,
    'nobattle_min_games': 20,
}

CHECK_KINDS: Tuple[str, ...] = (
    'nan_inf',
    'entropy_collapse',
    'kl_spike',
    'ev_drop',
    'eval_regression',
    'nobattle_surge',
)

TRAIN_NUMERIC_KEYS: Tuple[str, ...] = (
    'samples', 'lr', 'pg', 'v', 'ent', 'kl', 'mkl', 'clip', 'ev', 'belief', 'len',
)

METRIC_TO_PANEL: Dict[str, str] = {
    'ent': 'ent',
    'kl': 'kl',
    'mkl': 'kl',
    'clip': 'clip',
    'ev': 'ev',
    'v': 'ev',
    'belief': 'belief',
    'len': 'len',
    'pg': 'kl',
    'lr': 'kl',
    'samples': 'ent',
}


@dataclass(frozen=True)
class Flag:
    run: str
    kind: str
    it: int
    x: float
    metric: str
    value: float
    panel: str
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _merge_thresholds(thresholds: Optional[Dict[str, float]]) -> Dict[str, float]:
    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)
    return cfg


def _finite(v: object) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _x_of_row(run: dict, row: dict, fallback_x: float) -> float:
    off = float(run.get('offset', 0.0))
    s = row.get('samples')
    if _finite(s):
        return off + float(s)
    return fallback_x


def _x_of_it(run: dict, it: int) -> float:
    off = float(run.get('offset', 0.0))
    it_x = run.get('it_x')
    if not it_x:
        it_x = {
            row['it']: off + float(row['samples'])
            for row in run.get('train', [])
            if 'it' in row and _finite(row.get('samples'))
        }
    if not it_x:
        return off
    its = sorted(it_x)
    best = min(its, key=lambda i: abs(i - it))
    per_it = (it_x[its[-1]] - it_x[its[0]]) / max(its[-1] - its[0], 1)
    return it_x[best] + (it - best) * per_it


def check_run(name: str, run: dict, thresholds: Optional[Dict[str, float]] = None) -> List[Flag]:
    """Evaluate all 6 health checks on one parsed run dict (from plot_runs.parse/load_runs)."""
    cfg = _merge_thresholds(thresholds)
    flags: List[Flag] = []

    train_rows: Sequence[dict] = run.get('train', [])
    eval_rows: Sequence[dict] = run.get('eval', [])

    kl_history: List[float] = []
    nb_history: List[float] = []
    last_good_x = float(run.get('offset', 0.0))

    kl_window = int(cfg['kl_window'])
    kl_min_hist = int(cfg['kl_min_history'])
    nb_window = int(cfg['nobattle_window'])
    nb_min_hist = int(cfg['nobattle_min_history'])
    nb_min_games = int(cfg['nobattle_min_games'])

    for row in train_rows:
        it = int(row.get('it', 0))
        x = _x_of_row(run, row, last_good_x)
        if _finite(row.get('samples')):
            last_good_x = x

        # 1. NaN / inf in any train metric
        for k in TRAIN_NUMERIC_KEYS:
            if k in row and not _finite(row[k]):
                flags.append(Flag(
                    run=name,
                    kind='nan_inf',
                    it=it,
                    x=x,
                    metric=k,
                    value=float(row[k]),
                    panel=METRIC_TO_PANEL.get(k, 'ent'),
                    detail=f'{k}={row[k]} (non-finite)',
                ))

        # 2. Entropy collapse (ent < 0.5)
        if 'ent' in row and _finite(row['ent']):
            ent = float(row['ent'])
            if ent < cfg['ent_min']:
                flags.append(Flag(
                    run=name,
                    kind='entropy_collapse',
                    it=it,
                    x=x,
                    metric='ent',
                    value=ent,
                    panel='ent',
                    detail=f'ent={ent:.3f} < {cfg["ent_min"]:.2f}',
                ))

        # 3. KL spike (kl > 3 * running_mean of prior positive finite KL values)
        if 'kl' in row and _finite(row['kl']):
            kl = float(row['kl'])
            if len(kl_history) >= kl_min_hist:
                recent = kl_history[-kl_window:]
                rm = sum(recent) / len(recent)
                if rm > 0.0 and kl > cfg['kl_floor'] and kl > cfg['kl_mult'] * rm:
                    ratio = kl / rm
                    flags.append(Flag(
                        run=name,
                        kind='kl_spike',
                        it=it,
                        x=x,
                        metric='kl',
                        value=kl,
                        panel='kl',
                        detail=f'kl={kl:.4f} > {cfg["kl_mult"]:.1f}x running mean {rm:.4f} ({ratio:.2f}x)',
                    ))
            if kl > 0.0:
                kl_history.append(kl)

        # 4. Explained variance drop (ev < 0.5)
        if 'ev' in row and _finite(row['ev']):
            ev = float(row['ev'])
            if ev < cfg['ev_min']:
                flags.append(Flag(
                    run=name,
                    kind='ev_drop',
                    it=it,
                    x=x,
                    metric='ev',
                    value=ev,
                    panel='ev',
                    detail=f'ev={ev:.3f} < {cfg["ev_min"]:.2f}',
                ))

        # 6. No-battle draw surge
        ends = row.get('ends')
        if ends is not None and len(ends) >= 3:
            if any(not _finite(c) for c in ends):
                flags.append(Flag(
                    run=name,
                    kind='nan_inf',
                    it=it,
                    x=x,
                    metric='ends',
                    value=float('nan'),
                    panel='nobattle',
                    detail=f'ends={ends} (non-finite)',
                ))
            else:
                total_games = sum(int(c) for c in ends)
                if total_games >= nb_min_games:
                    nb_count = int(ends[2])
                    nb_pct = 100.0 * nb_count / total_games
                    abs_hit = nb_pct >= cfg['nobattle_abs_pct']
                    rel_hit = False
                    rel_mean = 0.0
                    if len(nb_history) >= nb_min_hist:
                        recent_nb = nb_history[-nb_window:]
                        rel_mean = sum(recent_nb) / len(recent_nb)
                        if (nb_pct >= cfg['nobattle_rel_min_pct']
                                and rel_mean > 0.0
                                and nb_pct > cfg['nobattle_rel_mult'] * rel_mean):
                            rel_hit = True
                    if abs_hit or rel_hit:
                        reasons = []
                        if abs_hit:
                            reasons.append(f'>= {cfg["nobattle_abs_pct"]:.1f}% abs')
                        if rel_hit:
                            reasons.append(f'> {cfg["nobattle_rel_mult"]:.1f}x mean {rel_mean:.2f}%')
                        flags.append(Flag(
                            run=name,
                            kind='nobattle_surge',
                            it=it,
                            x=x,
                            metric='nobattle_pct',
                            value=nb_pct,
                            panel='nobattle',
                            detail=f'nobattle={nb_count}/{total_games} ({nb_pct:.1f}%, {" & ".join(reasons)})',
                        ))
                    nb_history.append(nb_pct)

    # 5. Eval regression (drop > eval_z * SE below pooled moving average of prior evals) + NaN/inf
    eval_window = max(int(cfg['eval_window']), 1)
    eval_z = float(cfg['eval_z'])
    hist_by_series: Dict[Tuple[str, str], List[Tuple[float, float, int]]] = {}
    for e in eval_rows:
        it = int(e.get('it', 0))
        x = _x_of_it(run, it)
        mode = str(e.get('mode', ''))
        opp = str(e.get('name', ''))
        w, l, d, g = e.get('w'), e.get('l'), e.get('d'), e.get('g')
        if not (_finite(w) and _finite(l) and _finite(d) and _finite(g)) or float(g) <= 0:
            flags.append(Flag(
                run=name,
                kind='nan_inf',
                it=it,
                x=x,
                metric=f'eval:{mode}:{opp}',
                value=float('nan'),
                panel='eval_expert',
                detail=f'eval {mode} vs {opp}: invalid counts w={w} l={l} d={d} g={g}',
            ))
            continue

        if not opp.startswith('expert'):
            continue

        wf, gf = float(w), float(g)
        p_cur = wf / gf
        pct = 100.0 * p_cur
        key = (mode, opp)
        series = hist_by_series.setdefault(key, [])
        if series:
            win = series[-eval_window:]
            ref_w = sum(item[0] for item in win)
            ref_g = sum(item[1] for item in win)
            p_ref = ref_w / ref_g
            # Jeffreys-smoothed proportions avoid SE=0 at 0% or 100%
            p_ref_s = (ref_w + 0.5) / (ref_g + 1.0)
            p_cur_s = (wf + 0.5) / (gf + 1.0)
            se = math.sqrt(p_ref_s * (1.0 - p_ref_s) / ref_g + p_cur_s * (1.0 - p_cur_s) / gf)
            drop = p_ref - p_cur
            if se > 0.0 and drop > eval_z * se + 1e-12:
                z_score = drop / se
                flags.append(Flag(
                    run=name,
                    kind='eval_regression',
                    it=it,
                    x=x,
                    metric=f'eval:{mode}:{opp}',
                    value=pct,
                    panel='eval_expert',
                    detail=(f'{mode} vs {opp}: {pct:.1f}% (-{100.0 * drop:.1f} pp vs '
                            f'MA({len(win)}) {100.0 * p_ref:.1f}%, {z_score:.2f} SE > {eval_z:.1f})'),
                ))
        series.append((wf, gf, it))

    return flags


def check_runs(runs: Dict[str, dict], thresholds: Optional[Dict[str, float]] = None) -> Dict[str, List[Flag]]:
    """Run health checks across a mapping of run_name -> parsed run dict."""
    return {name: check_run(name, run, thresholds=thresholds) for name, run in runs.items()}


def format_report(
    runs: Dict[str, dict],
    flags_by_run: Dict[str, List[Flag]],
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Format a human-readable health report across all runs."""
    cfg = _merge_thresholds(thresholds)
    lines: List[str] = []
    lines.append(f"PPO Training Health Report  (generated {time.strftime('%Y-%m-%d %H:%M')})")
    lines.append("=" * 78)
    lines.append("Rules & thresholds:")
    lines.append("  - nan_inf          : any NaN or +/-inf in logged train/eval metrics")
    lines.append(f"  - entropy_collapse : ent < {cfg['ent_min']:.2f}")
    lines.append(
        f"  - kl_spike         : kl > {cfg['kl_mult']:.1f}x running mean of prior "
        f"<= {int(cfg['kl_window'])} positive finite updates (min {int(cfg['kl_min_history'])} prior, "
        f"floor {cfg['kl_floor']:.0e})"
    )
    lines.append(f"  - ev_drop          : ev < {cfg['ev_min']:.2f}")
    lines.append(
        f"  - eval_regression  : win % vs any expert drops > {cfg['eval_z']:.1f} x SE "
        f"below pooled moving average of prior <= {int(cfg['eval_window'])} evals (per mode & expert)"
    )
    lines.append(
        f"  - nobattle_surge   : no-battle draw share >= {cfg['nobattle_abs_pct']:.1f}% OR "
        f"(>= {cfg['nobattle_rel_min_pct']:.1f}% and > {cfg['nobattle_rel_mult']:.1f}x running mean of "
        f"prior <= {int(cfg['nobattle_window'])} points, min {int(cfg['nobattle_min_games'])} games)"
    )
    lines.append("")

    # Summary table
    header = (
        f"{'Run':<8} {'Rows':>5} {'Evals':>5} {'Total':>6} | "
        f"{'NaN':>4} {'Ent':>4} {'KL':>4} {'EV':>4} {'Eval':>4} {'NoBat':>5} | Status"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_flags = 0
    for name, run in runs.items():
        rflags = flags_by_run.get(name, [])
        total_flags += len(rflags)
        counts = {k: 0 for k in CHECK_KINDS}
        for fl in rflags:
            counts[fl.kind] = counts.get(fl.kind, 0) + 1
        status = 'OK' if not rflags else f'FLAGGED ({len(rflags)})'
        lines.append(
            f"{name:<8} {len(run.get('train', [])):>5} {len(run.get('eval', [])):>5} {len(rflags):>6} | "
            f"{counts['nan_inf']:>4} {counts['entropy_collapse']:>4} {counts['kl_spike']:>4} "
            f"{counts['ev_drop']:>4} {counts['eval_regression']:>4} {counts['nobattle_surge']:>5} | {status}"
        )

    lines.append("")
    lines.append(f"Total flags across {len(runs)} run(s): {total_flags}")
    lines.append("")

    # Detailed per-run section
    for name, run in runs.items():
        rflags = flags_by_run.get(name, [])
        tr = run.get('train', [])
        it_range = f"it {tr[0]['it']}..{tr[-1]['it']}" if tr else 'no train rows'
        lines.append(f"[{name}] ({it_range}, {len(rflags)} flag(s))")
        if not rflags:
            lines.append("  No issues flagged.")
        else:
            for fl in rflags:
                x_str = f"{fl.x:6.2f}M" if _finite(fl.x) else "   nanM"
                lines.append(f"  - it {fl.it:>4} ({x_str}) [{fl.kind:<16}] {fl.detail}")
        lines.append("")

    return '\n'.join(lines).rstrip() + '\n'


def write_report(
    runs: Dict[str, dict],
    out_path: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Flag]]:
    """Run health checks, write human-readable report to `out_path`, and return flags_by_run."""
    flags_by_run = check_runs(runs, thresholds=thresholds)
    text = format_report(runs, flags_by_run, thresholds=thresholds)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return flags_by_run


def main(argv: Optional[Sequence[str]] = None) -> None:
    from vec.plot_runs import load_runs  # lazy import to avoid circularity

    ap = argparse.ArgumentParser(description='Run health checks on PPO training logs.')
    ap.add_argument('--runs', default=os.path.join('..', 'runs'))
    ap.add_argument('--out', default=os.path.join('..', 'runs', 'plots', 'health.txt'))
    args = ap.parse_args(argv)
    runs = load_runs(args.runs)
    flags = write_report(runs, args.out)
    n_flags = sum(len(v) for v in flags.values())
    print(f'[health] {len(runs)} runs, {n_flags} flags -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
