"""
Round-robin Elo ladder with Bradley-Terry ratings and bootstrap 95% confidence intervals.

Plays every unordered pair of players for `games` games split evenly across fresh setup-bank
seeds (default: 400 games/pair = 200 per seed across seeds 101 and 202, sampled T=1.0),
alternating colors and starting sides symmetrically, fits a Bradley-Terry Elo model
(draws = 0.5, anchored at expert = 1500.0), estimates 95% CIs via game-level stratified
multinomial bootstrap, and writes `ladder.json` and `ladder.png`.

    python -m vec.ladder --out ../runs/ladder
    python -m vec.ladder --out ../runs/ladder --replot
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

from vec.arena import END_NAMES
from vec.engine import VecStratego
from vec.expert import Style, VecExpert, VecRandom
from vec.net import NetPolicy, load
from vec.setups import diverse_setup_bank

ELO_SCALE = 400.0 / math.log(10.0)


def _is_strongly_connected(score_mat: np.ndarray) -> bool:
    """Return True iff the directed graph of edges (i -> j where score_mat[i, j] > 0) is strongly connected."""
    k = score_mat.shape[0]
    if k <= 1:
        return True
    adj = score_mat > 0.0
    np.fill_diagonal(adj, False)

    def reachable(mat: np.ndarray) -> int:
        seen = [False] * k
        stack = [0]
        seen[0] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(mat[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        return sum(seen)

    return reachable(adj) == k and reachable(adj.T) == k


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_bradley_terry(
    wins: np.ndarray,
    losses: np.ndarray,
    draws: np.ndarray,
    anchor_idx: int = 0,
    anchor_elo: float = 1500.0,
    reg: float = 0.0,
    fallback_reg: float = 1e-4,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Fit Bradley-Terry Elo ratings (draws counted as 0.5) via damped Newton-Raphson.

    Parameters
    ----------
    wins, losses, draws : (K, K) array-like
        Entry (i, j) is player i's wins, losses, and draws against player j. Must satisfy
        wins == losses.T and draws == draws.T. Diagonal entries are ignored.
    anchor_idx : int
        Player index fixed to `anchor_elo`.
    anchor_elo : float
        Elo value of the anchor player (default 1500.0).
    reg : float
        Optional L2 ridge penalty on mean-centered log-strengths (default 0.0 = exact MLE).
    fallback_reg : float
        Small L2 penalty applied automatically ONLY when `reg == 0.0` and the directed
        positive-score graph is not strongly connected (separation / all-win or all-loss
        player), preventing infinite ratings.
    """
    w = np.asarray(wins, dtype=np.float64).copy()
    l = np.asarray(losses, dtype=np.float64).copy()
    d = np.asarray(draws, dtype=np.float64).copy()
    if w.ndim != 2 or w.shape[0] != w.shape[1] or l.shape != w.shape or d.shape != w.shape:
        raise ValueError(f"Expected square (K, K) matrices, got {w.shape}, {l.shape}, {d.shape}")
    k = w.shape[0]
    if k == 0:
        return np.empty(0, dtype=np.float64)
    if not (0 <= anchor_idx < k):
        raise ValueError(f"anchor_idx={anchor_idx} out of range for K={k}")
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(l, 0.0)
    np.fill_diagonal(d, 0.0)
    if not np.allclose(w, l.T, atol=1e-9):
        raise ValueError("wins matrix must equal losses.T")
    if not np.allclose(d, d.T, atol=1e-9):
        raise ValueError("draws matrix must be symmetric")
    if k == 1:
        return np.array([float(anchor_elo)], dtype=np.float64)

    n = w + l + d
    s = w + 0.5 * d
    if n.sum() == 0.0:
        return np.full(k, float(anchor_elo), dtype=np.float64)

    eff_reg = float(reg)
    if eff_reg == 0.0 and not _is_strongly_connected(s):
        eff_reg = float(fallback_reg)

    free = np.array([i for i in range(k) if i != anchor_idx], dtype=np.int64)
    theta = np.zeros(k, dtype=np.float64)

    def obj_and_grad(th: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        diff = th[:, None] - th[None, :]                 # theta_i - theta_j
        # log sigmoid(diff) = -logaddexp(0, -diff)
        ll = float(-np.sum(s * np.logaddexp(0.0, -diff)))
        p = _sigmoid(diff)
        np.fill_diagonal(p, 0.0)
        g = np.sum(s - n * p, axis=1)
        if eff_reg > 0.0:
            centered = th - th.mean()
            ll -= 0.5 * eff_reg * float(np.dot(centered, centered))
            g -= eff_reg * centered
        return ll, g, p

    ll, g, p = obj_and_grad(theta)
    for _ in range(max_iter):
        g_f = g[free]
        if float(np.max(np.abs(g_f))) < tol:
            break
        wt = n * p * (1.0 - p)
        np.fill_diagonal(wt, 0.0)
        neg_h = -wt
        neg_h[np.arange(k), np.arange(k)] = np.sum(wt, axis=1)
        if eff_reg > 0.0:
            neg_h -= eff_reg / k
            neg_h[np.arange(k), np.arange(k)] += eff_reg
        h_f = neg_h[np.ix_(free, free)]
        # Solve h_f @ step = g_f (ascending log-likelihood); add tiny damping only if ill-conditioned
        step = None
        for damp in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
            try:
                mat = h_f if damp == 0.0 else h_f + damp * np.eye(k - 1)
                cand = np.linalg.solve(mat, g_f)
                if np.all(np.isfinite(cand)) and float(np.dot(g_f, cand)) > 0.0:
                    step = cand
                    break
            except np.linalg.LinAlgError:
                continue
        if step is None:
            step = g_f / (np.linalg.norm(g_f) + 1e-12)
        # Cap step magnitude for stability when odds are extreme
        max_step = float(np.max(np.abs(step)))
        if max_step > 5.0:
            step = step * (5.0 / max_step)
        # Backtracking Armijo line search on the strictly concave objective
        alpha = 1.0
        slope = float(np.dot(g_f, step))
        accepted = False
        for _ls in range(40):
            th_new = theta.copy()
            th_new[free] += alpha * step
            ll_new, g_new, p_new = obj_and_grad(th_new)
            if np.isfinite(ll_new) and ll_new >= ll + 1e-4 * alpha * slope - 1e-14:
                theta, ll, g, p = th_new, ll_new, g_new, p_new
                accepted = True
                break
            alpha *= 0.5
        if not accepted or alpha * max_step < 1e-13:
            break

    return anchor_elo + ELO_SCALE * (theta - theta[anchor_idx])


def bt_expected_matrix(elo: np.ndarray) -> np.ndarray:
    """Return (K, K) matrix of Bradley-Terry expected scores P(i beats j) = sigmoid((R_i - R_j) / ELO_SCALE)."""
    elo = np.asarray(elo, dtype=np.float64)
    return _sigmoid((elo[:, None] - elo[None, :]) / ELO_SCALE)


def bootstrap_elo(
    wins: np.ndarray,
    losses: np.ndarray,
    draws: np.ndarray,
    anchor_idx: int = 0,
    anchor_elo: float = 1500.0,
    n_boot: int = 1000,
    seed: int = 0,
    per_seed: Optional[Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    reg: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Point estimate and nonparametric game-level bootstrap 95% CIs for Bradley-Terry Elo.

    For each unordered pair (i < j), resamples game outcomes (win / loss / draw) with
    replacement via Multinomial(N_ij, (W_ij/N_ij, L_ij/N_ij, D_ij/N_ij)). When `per_seed`
    is supplied, resampling is stratified by setup-bank seed and summed per bootstrap draw.
    """
    w = np.asarray(wins, dtype=np.float64)
    l = np.asarray(losses, dtype=np.float64)
    d = np.asarray(draws, dtype=np.float64)
    k = w.shape[0]
    elo = fit_bradley_terry(w, l, d, anchor_idx=anchor_idx, anchor_elo=anchor_elo, reg=reg)
    if n_boot <= 0:
        return {
            'elo': elo,
            'ci_low': elo.copy(),
            'ci_high': elo.copy(),
            'std': np.zeros(k, dtype=np.float64),
            'samples': elo[None, :].copy(),
            'delta_ci_low': elo[:, None] - elo[None, :],
            'delta_ci_high': elo[:, None] - elo[None, :],
            'prob_stronger': (elo[:, None] > elo[None, :]).astype(np.float64),
        }

    strata = (
        [(np.asarray(ws, dtype=np.float64), np.asarray(ls, dtype=np.float64), np.asarray(ds, dtype=np.float64))
         for ws, ls, ds in per_seed]
        if per_seed is not None and len(per_seed) > 0
        else [(w, l, d)]
    )
    # Precompute upper-triangle (i, j, n_int, probs) per stratum for fast resampling
    triples = []
    for ws, ls, ds in strata:
        for i in range(k):
            for j in range(i + 1, k):
                n_ij = ws[i, j] + ls[i, j] + ds[i, j]
                n_int = int(round(n_ij))
                if n_int <= 0:
                    continue
                probs = np.array([ws[i, j], ls[i, j], ds[i, j]], dtype=np.float64) / n_ij
                probs = np.clip(probs, 0.0, 1.0)
                probs /= probs.sum()
                triples.append((i, j, n_int, probs))

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, k), dtype=np.float64)
    for b in range(n_boot):
        wb = np.zeros((k, k), dtype=np.float64)
        lb = np.zeros((k, k), dtype=np.float64)
        db = np.zeros((k, k), dtype=np.float64)
        for i, j, n_int, probs in triples:
            c_w, c_l, c_d = rng.multinomial(n_int, probs)
            wb[i, j] += c_w
            lb[i, j] += c_l
            db[i, j] += c_d
            wb[j, i] += c_l
            lb[j, i] += c_w
            db[j, i] += c_d
        boot[b] = fit_bradley_terry(wb, lb, db, anchor_idx=anchor_idx, anchor_elo=anchor_elo, reg=reg)

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5], axis=0)
    std = np.std(boot, axis=0, ddof=1) if n_boot > 1 else np.zeros(k, dtype=np.float64)
    # Pairwise rating difference CIs: R_i - R_j
    diffs = boot[:, :, None] - boot[:, None, :]          # (n_boot, K, K)
    delta_ci_low, delta_ci_high = np.percentile(diffs, [2.5, 97.5], axis=0)
    prob_stronger = np.mean(diffs > 0.0, axis=0)
    return {
        'elo': elo,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'std': std,
        'samples': boot,
        'delta_ci_low': delta_ci_low,
        'delta_ci_high': delta_ci_high,
        'prob_stronger': prob_stronger,
    }


@torch.no_grad()
def play_match_balanced(
    policy_a,
    policy_b,
    num_games: int,
    bank: torch.Tensor,
    device,
    max_plies: int = 1500,
    no_battle_plies: int = 200,
    bank_b: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Like `vec.arena.play_match`, with exact 4-way balance over colors (P1/P2) and starting player.

    Even games put A on side +1 and odd on -1; pairs of games alternate who moves first (+1/-1),
    so for any multiple of 4 games A is evenly split across (P1 first, P1 second, P2 first, P2 second).
    Each game draws independent setups from `bank` (and `bank_b` if supplied).
    """
    env = VecStratego(num_games, device, max_plies=max_plies, no_battle_plies=no_battle_plies)
    all_idx = torch.arange(num_games, device=env.device)
    a_side = torch.where(all_idx % 2 == 0, 1, -1)
    start = torch.where((all_idx // 2) % 2 == 0, 1, -1)
    bank_b = bank if bank_b is None else bank_b
    pick = lambda bk: bk[torch.randint(0, bk.size(0), (num_games,), device=bk.device)].to(env.device)
    sa, sb = pick(bank), pick(bank_b)
    a_starts = (start == a_side)[:, None]
    env.reset(all_idx, torch.where(a_starts, sa, sb), torch.where(a_starts, sb, sa), start)
    actions = torch.zeros(num_games, dtype=torch.long, device=env.device)
    while not bool(env.done.all()):
        live = ~env.done
        ia = (live & (env.side == a_side)).nonzero().flatten()
        ib = (live & (env.side != a_side)).nonzero().flatten()
        if ia.numel():
            actions[ia] = policy_a.act(env, ia)
        if ib.numel():
            actions[ib] = policy_b.act(env, ib)
        env.step(actions)
    res = env.winner * a_side
    wins = int((res == 1).sum())
    losses = int((res == -1).sum())
    draws = int((res == 0).sum())
    out: Dict[str, float] = {
        'games': num_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'plies': float(env.ply.float().mean()),
        'win_rate': wins / num_games,
        'score_rate': (wins + 0.5 * draws) / num_games,
    }
    for code, name in END_NAMES.items():
        e = env.end_type == code
        out[f'win_{name}'] = int((e & (res == 1)).sum())
        out[f'loss_{name}'] = int((e & (res == -1)).sum())
        out[f'draw_{name}'] = int((e & (res == 0)).sum())
    return out


def default_roster(runs_dir: str, styles_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Build the Phase A1 default player roster (checkpoints + scripted experts + random)."""
    styles_file = styles_path or os.path.join(runs_dir, 'ppo_2', 'expert_styles.json')
    candidates = [
        {'name': 'random', 'kind': 'random', 'spec': 'random'},
        {'name': 'pv_clone_3', 'kind': 'net', 'spec': os.path.join(runs_dir, 'pv_clone_3', 'net_final.pt')},
        {'name': 'expert', 'kind': 'expert', 'spec': 'expert'},
        {'name': 'expert2', 'kind': 'expert_style', 'spec': f'{styles_file}#0'},
        {'name': 'expert3', 'kind': 'expert_style', 'spec': f'{styles_file}#1'},
        {'name': 'ppo_1_40', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_1', 'net_40.pt')},
        {'name': 'ppo_1', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_1', 'net_final.pt')},
        {'name': 'ppo_2_500', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_2', 'net_500.pt')},
        {'name': 'ppo_2', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_2', 'net_final.pt')},
        {'name': 'ppo_3', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_3', 'net_final.pt')},
        {'name': 'ppo_4_150', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_4', 'net_150.pt')},
        {'name': 'ppo_4', 'kind': 'net', 'spec': os.path.join(runs_dir, 'ppo_4', 'net_final.pt')},
    ]
    roster = []
    for item in candidates:
        if item['kind'] == 'net' and not os.path.exists(item['spec']):
            continue
        if item['kind'] == 'expert_style':
            path = item['spec'].split('#')[0]
            if not os.path.exists(path):
                continue
        roster.append(item)
    return roster


def build_policy(entry: Dict[str, str], device, temperature: float):
    """Instantiate a batched policy from a roster entry."""
    kind = entry['kind']
    spec = entry['spec']
    if kind == 'random':
        return VecRandom()
    if kind == 'expert':
        return VecExpert()
    if kind == 'expert_style':
        path, idx_str = spec.split('#')
        with open(path) as f:
            styles = json.load(f)
        return VecExpert(style=Style(**styles[int(idx_str)]), name=entry['name'])
    if kind == 'net':
        net = load(spec, device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return NetPolicy(net, temperature=temperature, name=entry['name'])
    raise ValueError(f"Unknown policy kind: {kind!r}")


def _aggregate_from_pairs(
    names: List[str],
    pairs: List[Dict],
    seeds: List[int],
    anchor: str,
    anchor_elo: float,
    n_boot: int,
    bootstrap_seed: int,
) -> Dict:
    """Build matrices, Bradley-Terry Elo + bootstrap CIs, and ranked summary from pair records."""
    k = len(names)
    idx_of = {name: i for i, name in enumerate(names)}
    if anchor not in idx_of:
        raise ValueError(f"Anchor player {anchor!r} not in roster {names}")
    anchor_idx = idx_of[anchor]

    w = np.zeros((k, k), dtype=np.float64)
    l = np.zeros((k, k), dtype=np.float64)
    d = np.zeros((k, k), dtype=np.float64)
    seed_mats = {s: (np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k))) for s in seeds}

    for rec in pairs:
        i, j = idx_of[rec['a']], idx_of[rec['b']]
        w[i, j] += rec['wins']
        l[i, j] += rec['losses']
        d[i, j] += rec['draws']
        w[j, i] += rec['losses']
        l[j, i] += rec['wins']
        d[j, i] += rec['draws']
        for ps in rec.get('per_seed', []):
            s = int(ps['seed'])
            if s in seed_mats:
                ws, ls, ds = seed_mats[s]
                ws[i, j] += ps['wins']
                ls[i, j] += ps['losses']
                ds[i, j] += ps['draws']
                ws[j, i] += ps['losses']
                ls[j, i] += ps['wins']
                ds[j, i] += ps['draws']

    per_seed_list = [seed_mats[s] for s in seeds] if all(
        (seed_mats[s][0] + seed_mats[s][1] + seed_mats[s][2]).sum() > 0 for s in seeds
    ) else None

    bt = bootstrap_elo(
        w, l, d,
        anchor_idx=anchor_idx,
        anchor_elo=anchor_elo,
        n_boot=n_boot,
        seed=bootstrap_seed,
        per_seed=per_seed_list,
    )
    elo = bt['elo']
    order_idx = np.argsort(-elo)
    order_names = [names[i] for i in order_idx]
    expected = bt_expected_matrix(elo)

    n = w + l + d
    rankings = []
    elo_map = {}
    for rank_pos, i in enumerate(order_idx, start=1):
        tot_w = int(round(w[i].sum()))
        tot_l = int(round(l[i].sum()))
        tot_d = int(round(d[i].sum()))
        tot_g = tot_w + tot_l + tot_d
        avg_score = (tot_w + 0.5 * tot_d) / tot_g if tot_g > 0 else 0.0
        row = {
            'rank': rank_pos,
            'name': names[i],
            'elo': float(elo[i]),
            'ci_low': float(bt['ci_low'][i]),
            'ci_high': float(bt['ci_high'][i]),
            'std': float(bt['std'][i]),
            'avg_score': float(avg_score),
            'total_wins': tot_w,
            'total_losses': tot_l,
            'total_draws': tot_d,
            'total_games': tot_g,
        }
        rankings.append(row)
        elo_map[names[i]] = {
            'elo': row['elo'],
            'ci_low': row['ci_low'],
            'ci_high': row['ci_high'],
            'std': row['std'],
        }

    def ordered_matrix(mat: np.ndarray, mask_diag: bool = True):
        out = []
        for a_i in order_idx:
            row = []
            for b_i in order_idx:
                if mask_diag and a_i == b_i:
                    row.append(None)
                else:
                    val = float(mat[a_i, b_i])
                    row.append(val if math.isfinite(val) else None)
            out.append(row)
        return out

    with np.errstate(divide='ignore', invalid='ignore'):
        score_rate = np.where(n > 0, (w + 0.5 * d) / n, np.nan)
        win_rate = np.where(n > 0, w / n, np.nan)
        draw_rate = np.where(n > 0, d / n, np.nan)

    # Goodness of fit: RMS residual between observed pairwise score and BT expected score
    mask = np.triu(n > 0, k=1)
    bt_rmse = float(np.sqrt(np.mean((score_rate[mask] - expected[mask]) ** 2))) if mask.any() else 0.0
    bt_max_err = float(np.max(np.abs(score_rate[mask] - expected[mask]))) if mask.any() else 0.0

    return {
        'rankings': rankings,
        'elo': elo_map,
        'bt_fit': {
            'anchor': anchor,
            'anchor_elo': float(anchor_elo),
            'n_boot': int(n_boot),
            'bootstrap_seed': int(bootstrap_seed),
            'bootstrap_method': 'stratified_pair_multinomial' if per_seed_list is not None else 'pair_multinomial',
            'pairwise_score_rmse': bt_rmse,
            'pairwise_score_max_abs_err': bt_max_err,
        },
        'matrix': {
            'order': order_names,
            'score_rate': ordered_matrix(score_rate),
            'win_rate': ordered_matrix(win_rate),
            'draw_rate': ordered_matrix(draw_rate),
            'bt_expected_score': ordered_matrix(expected),
            'delta_ci_low': ordered_matrix(bt['delta_ci_low']),
            'delta_ci_high': ordered_matrix(bt['delta_ci_high']),
            'prob_stronger': ordered_matrix(bt['prob_stronger']),
        },
    }


def run_tournament(
    roster: List[Dict[str, str]],
    policies: Dict[str, object],
    games_per_pair: int,
    seeds: Sequence[int],
    device,
    bank_size: int = 4000,
    anchor: str = 'expert',
    anchor_elo: float = 1500.0,
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
    temperature: float = 1.0,
    max_plies: int = 1500,
    no_battle_plies: int = 200,
    log_fn=print,
    progress_path: Optional[str] = None,
    existing_pairs: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Run a full round-robin across `roster` over `seeds` setup banks and fit Elo + CIs.

    If `existing_pairs` is provided, any unordered pair (a, b) already present is reused
    without replaying it (so new players can be added onto a completed ladder on the exact
    same setup banks and seeds).
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("Need at least one setup-bank seed")
    if games_per_pair % len(seeds) != 0:
        raise ValueError(f"games_per_pair ({games_per_pair}) must be divisible by len(seeds) ({len(seeds)})")
    games_per_seed = games_per_pair // len(seeds)

    banks = {}
    for s in seeds:
        banks[s] = diverse_setup_bank(bank_size, seed=s).to(device)

    names = [e['name'] for e in roster]
    k = len(names)
    total_pairs = k * (k - 1) // 2
    reused_map: Dict[Tuple[str, str], Dict] = {}
    if existing_pairs:
        for rec in existing_pairs:
            reused_map[(rec['a'], rec['b'])] = rec
            reused_map[(rec['b'], rec['a'])] = rec
    pairs: List[Dict] = []
    t0 = time.time()
    pair_idx = 0

    for i in range(k):
        for j in range(i + 1, k):
            pair_idx += 1
            na, nb = names[i], names[j]
            if (na, nb) in reused_map:
                pairs.append(reused_map[(na, nb)])
                continue
            pa, pb = policies[na], policies[nb]
            per_seed_records = []
            tot_w = tot_l = tot_d = 0
            tot_plies = 0.0
            endings = {f'{pref}_{ename}': 0 for ename in END_NAMES.values() for pref in ('win', 'loss', 'draw')}

            for s_idx, s in enumerate(seeds):
                # Deterministic per-(pair, seed) RNG for reproducible sampling and setup selection
                torch.manual_seed(s * 100003 + i * 997 + j * 31 + s_idx)
                r = play_match_balanced(
                    pa, pb, games_per_seed, banks[s], device,
                    max_plies=max_plies, no_battle_plies=no_battle_plies,
                )
                tot_w += int(r['wins'])
                tot_l += int(r['losses'])
                tot_d += int(r['draws'])
                tot_plies += float(r['plies']) * games_per_seed
                for key in endings:
                    endings[key] += int(r.get(key, 0))
                per_seed_records.append({'seed': s, **r})

            rec = {
                'a': na,
                'b': nb,
                'games': games_per_pair,
                'wins': tot_w,
                'losses': tot_l,
                'draws': tot_d,
                'win_rate': tot_w / games_per_pair,
                'score_rate': (tot_w + 0.5 * tot_d) / games_per_pair,
                'plies': tot_plies / games_per_pair,
                'endings': endings,
                'per_seed': per_seed_records,
            }
            pairs.append(rec)
            seed_str = ' | '.join(f"s{ps['seed']}:{ps['wins']}/{ps['losses']}/{ps['draws']}" for ps in per_seed_records)
            log_fn(
                f"[ladder {pair_idx:2d}/{total_pairs}] {na:12s} vs {nb:12s}: "
                f"{tot_w:3d}/{tot_l:3d}/{tot_d:3d} (score {100 * rec['score_rate']:5.1f}%, "
                f"{rec['plies']:3.0f} plies) [{seed_str}] ({time.time() - t0:.0f}s)"
            )
            if progress_path is not None:
                partial = {
                    'config': {
                        'games_per_pair': games_per_pair,
                        'seeds': seeds,
                        'games_per_seed': games_per_seed,
                        'temperature': temperature,
                        'bank_size': bank_size,
                        'anchor': anchor,
                        'anchor_elo': anchor_elo,
                        'n_boot': n_boot,
                        'bootstrap_seed': bootstrap_seed,
                        'device': str(device),
                    },
                    'players': roster,
                    'pairs': pairs,
                }
                with open(progress_path, 'w') as f:
                    json.dump(partial, f, indent=2)

    agg = _aggregate_from_pairs(names, pairs, seeds, anchor, anchor_elo, n_boot, bootstrap_seed)
    return {
        'config': {
            'games_per_pair': games_per_pair,
            'seeds': seeds,
            'games_per_seed': games_per_seed,
            'temperature': temperature,
            'bank_size': bank_size,
            'anchor': anchor,
            'anchor_elo': anchor_elo,
            'n_boot': n_boot,
            'bootstrap_seed': bootstrap_seed,
            'device': str(device),
            'elapsed_sec': round(time.time() - t0, 2),
        },
        'players': roster,
        'pairs': pairs,
        **agg,
    }


def _player_color(name: str) -> str:
    if name == 'random':
        return '#8c8c8c'
    if name.startswith('expert'):
        return '#d95f02'
    if name.startswith('pv_clone'):
        return '#7570b3'
    if name.startswith('ppo_5') or name.startswith('ppo_6'):
        return '#e7298a'
    if name == 'ppo_4':
        return '#1b9e77'
    return '#386cb0'


def plot_ladder(result: Dict, out_png: str) -> None:
    """Render a 2-panel Elo ladder chart: left = Elo + 95% CIs, right = pairwise score heatmap."""
    rankings = result['rankings']
    order = result['matrix']['order']
    score_mat = result['matrix']['score_rate']
    anchor = result['bt_fit']['anchor']
    anchor_elo = result['bt_fit']['anchor_elo']
    cfg = result['config']
    k = len(rankings)

    fig, (ax_elo, ax_hm) = plt.subplots(
        1, 2, figsize=(16.5, max(6.8, 0.55 * k + 2.0)),
        gridspec_kw={'width_ratios': [1.05, 1.35]},
    )

    # --- Left panel: Elo ratings with 95% bootstrap error bars
    y = np.arange(k)
    elos = np.array([r['elo'] for r in rankings])
    lo = np.array([r['elo'] - r['ci_low'] for r in rankings])
    hi = np.array([r['ci_high'] - r['elo'] for r in rankings])
    names = [r['name'] for r in rankings]
    colors = [_player_color(nm) for nm in names]

    ax_elo.axvline(anchor_elo, color='#666666', linestyle='--', linewidth=1.1, alpha=0.75,
                   label=f'Anchor ({anchor} = {anchor_elo:.0f})')
    for idx_r in range(k):
        ax_elo.errorbar(
            elos[idx_r], y[idx_r], xerr=[[lo[idx_r]], [hi[idx_r]]],
            fmt='o', color=colors[idx_r], ecolor=colors[idx_r],
            elinewidth=2.2, capsize=4, markersize=7.5, zorder=3,
        )
        ci_txt = (
            f"{elos[idx_r]:.0f} (anchor)" if names[idx_r] == anchor
            else f"{elos[idx_r]:.0f} [{rankings[idx_r]['ci_low']:.0f}, {rankings[idx_r]['ci_high']:.0f}]"
        )
        ax_elo.text(rankings[idx_r]['ci_high'] + 18.0, y[idx_r], ci_txt,
                    va='center', ha='left', fontsize=8.8, color='#222222')

    ax_elo.set_yticks(y)
    ax_elo.set_yticklabels(names, fontsize=10)
    ax_elo.invert_yaxis()
    x_min = min(r['ci_low'] for r in rankings) - 60.0
    x_max = max(r['ci_high'] for r in rankings) + 400.0
    ax_elo.set_xlim(x_min, x_max)
    ax_elo.set_xlabel('Bradley-Terry Elo (draws = 0.5, 95% bootstrap CI)', fontsize=10.5)
    ax_elo.set_title('A. Elo Ladder (95% Bootstrap CIs)', fontsize=12, loc='left', fontweight='bold')
    ax_elo.grid(axis='x', alpha=0.3)
    ax_elo.legend(loc='lower right', fontsize=8.8)

    # --- Right panel: Pairwise score-rate heatmap (row vs column)
    arr = np.full((k, k), np.nan, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if score_mat[i][j] is not None:
                arr[i, j] = 100.0 * score_mat[i][j]

    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('#eeeeee')
    im = ax_hm.imshow(arr, cmap=cmap, vmin=0.0, vmax=100.0, aspect='equal')
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.044, pad=0.03)
    cbar.set_label('Row score vs Column: (W + 0.5 D) / N (%)', fontsize=9.5)

    ax_hm.set_xticks(np.arange(k))
    ax_hm.set_yticks(np.arange(k))
    ax_hm.set_xticklabels(order, rotation=42, ha='right', fontsize=9.2)
    ax_hm.set_yticklabels(order, fontsize=9.2)
    ax_hm.set_title('B. Pairwise Score Matrix (%) — Row vs Column', fontsize=12, loc='left', fontweight='bold')

    for i in range(k):
        for j in range(k):
            if i == j or np.isnan(arr[i, j]):
                ax_hm.text(j, i, '—', ha='center', va='center', fontsize=8.5, color='#888888')
            else:
                val = arr[i, j]
                txt_col = 'white' if (val < 18.0 or val > 85.0) else '#111111'
                ax_hm.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7.8,
                           color=txt_col, fontweight='bold' if abs(val - 50.0) > 10.0 else 'normal')

    seeds_str = ','.join(str(s) for s in cfg.get('seeds', []))
    rmse = result['bt_fit']['pairwise_score_rmse']
    fig.suptitle(
        f"Stratego Round-Robin Elo Ladder  |  {cfg.get('games_per_pair')} games/pair "
        f"(T={cfg.get('temperature')}, seeds={seeds_str}, {result['bt_fit']['n_boot']} bootstraps, "
        f"BT RMSE={100 * rmse:.1f} pp)",
        fontsize=12.5, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Round-robin Bradley-Terry Elo ladder with bootstrap 95% CIs.')
    ap.add_argument('--out', default='../runs/ladder', help='Output directory for ladder.json and ladder.png')
    ap.add_argument('--runs-dir', default='../runs', help='Root runs directory for default roster discovery')
    ap.add_argument('--styles', default=None, help='Path to expert_styles.json (default: <runs-dir>/ppo_2/expert_styles.json)')
    ap.add_argument('--games', type=int, default=400, help='Total games per pair (split evenly across --seeds)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[101, 202], help='Setup-bank seeds (fresh, distinct from training)')
    ap.add_argument('--bank-size', type=int, default=4000, help='Number of setups generated per seed bank')
    ap.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for network policies')
    ap.add_argument('--anchor', default='expert', help='Anchor player name')
    ap.add_argument('--anchor-elo', type=float, default=1500.0, help='Anchor Elo value')
    ap.add_argument('--bootstraps', type=int, default=1000, help='Number of bootstrap resamples')
    ap.add_argument('--bootstrap-seed', type=int, default=0, help='RNG seed for bootstrap resampling')
    ap.add_argument('--max-plies', type=int, default=1500)
    ap.add_argument('--no-battle-plies', type=int, default=200)
    ap.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available else cpu)')
    ap.add_argument('--extend', default=None, help='Existing ladder.json to extend (reuses all previously played pairs)')
    ap.add_argument('--add', nargs='*', default=[], metavar='NAME=CKPT',
                    help='Extra net players to add to the roster (NAME=path/to/net.pt)')
    ap.add_argument('--replot', action='store_true', help='Re-fit Elo/CIs and redraw ladder.png from existing ladder.json')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, 'ladder.json')
    png_path = os.path.join(args.out, 'ladder.png')
    log_path = os.path.join(args.out, 'log.txt')

    if args.replot:
        with open(json_path) as f:
            prev = json.load(f)
        names = [p['name'] for p in prev['players']]
        seeds = prev['config']['seeds']
        agg = _aggregate_from_pairs(names, prev['pairs'], seeds, args.anchor, args.anchor_elo,
                                    args.bootstraps, args.bootstrap_seed)
        prev.update(agg)
        with open(json_path, 'w') as f:
            json.dump(prev, f, indent=2)
        plot_ladder(prev, png_path)
        print(f"[ladder] replotted {png_path} from {json_path}")
        return prev

    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    existing_pairs = None
    if args.extend:
        with open(args.extend) as f:
            prev = json.load(f)
        roster = list(prev['players'])
        existing_pairs = list(prev['pairs'])
        cfg_prev = prev['config']
        args.games = int(cfg_prev['games_per_pair'])
        args.seeds = [int(s) for s in cfg_prev['seeds']]
        args.bank_size = int(cfg_prev['bank_size'])
        args.temperature = float(cfg_prev['temperature'])
    else:
        roster = default_roster(args.runs_dir, args.styles)
    seen_names = {p['name'] for p in roster}
    for spec in (args.add or []):
        if '=' not in spec:
            raise ValueError(f"--add must be NAME=CKPT, got {spec!r}")
        nm, path = spec.split('=', 1)
        if nm not in seen_names:
            roster.append({'name': nm, 'kind': 'net', 'spec': path})
            seen_names.add(nm)
    with open(log_path, 'w') as log_f:
        def say(msg: str):
            print(msg, flush=True)
            log_f.write(msg + '\n')
            log_f.flush()

        say(f"[ladder] device={dev} games={args.games} seeds={args.seeds} T={args.temperature} "
            f"bootstraps={args.bootstraps} players={[p['name'] for p in roster]}")
        policies = {entry['name']: build_policy(entry, dev, args.temperature) for entry in roster}
        result = run_tournament(
            roster=roster,
            policies=policies,
            games_per_pair=args.games,
            seeds=args.seeds,
            device=dev,
            bank_size=args.bank_size,
            anchor=args.anchor,
            anchor_elo=args.anchor_elo,
            n_boot=args.bootstraps,
            bootstrap_seed=args.bootstrap_seed,
            temperature=args.temperature,
            max_plies=args.max_plies,
            no_battle_plies=args.no_battle_plies,
            log_fn=say,
            progress_path=json_path,
            existing_pairs=existing_pairs,
        )
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        plot_ladder(result, png_path)
        say("\n[ladder] Final Elo rankings:")
        for r in result['rankings']:
            say(f"  {r['rank']:2d}. {r['name']:12s}  Elo {r['elo']:7.1f}  95% CI [{r['ci_low']:7.1f}, {r['ci_high']:7.1f}]  "
                f"avg_score {100 * r['avg_score']:5.1f}%  ({r['total_wins']}/{r['total_losses']}/{r['total_draws']})")
        say(f"[ladder] BT fit pairwise score RMSE = {100 * result['bt_fit']['pairwise_score_rmse']:.2f} pp, "
            f"max abs err = {100 * result['bt_fit']['pairwise_score_max_abs_err']:.2f} pp")
        say(f"[ladder] wrote {json_path} and {png_path}")
    return result


if __name__ == '__main__':
    main()
