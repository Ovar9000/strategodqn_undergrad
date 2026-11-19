"""Unit and CPU smoke tests for vec/ladder.py (Bradley-Terry Elo + bootstrap CIs) and vec/exploit.py."""

import json
import os

import numpy as np
import pytest
import torch

from vec import exploit as exploit_mod
from vec import ladder as ladder_mod
from vec.expert import VecRandom
from vec.ladder import (
    ELO_SCALE,
    bootstrap_elo,
    bt_expected_matrix,
    fit_bradley_terry,
    plot_ladder,
    run_tournament,
)
from vec.net import NetPolicy, PolicyValueNet, load, save


def _exact_odds_matrices(true_elos, n_games: float, draw_frac: float = 0.0):
    """Build exact-odds (W, L, D) matrices matching Bradley-Terry ratings `true_elos`."""
    elos = np.asarray(true_elos, dtype=np.float64)
    k = len(elos)
    p = bt_expected_matrix(elos)
    np.fill_diagonal(p, 0.0)
    # Keep draws <= 2 * min(p, 1 - p) so W and L stay non-negative while W + 0.5*D = n_games * p
    d = np.minimum(draw_frac, 2.0 * np.minimum(p, 1.0 - p)) * n_games
    np.fill_diagonal(d, 0.0)
    w = n_games * p - 0.5 * d
    l = n_games * (1.0 - p) - 0.5 * d
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(l, 0.0)
    return w, l, d


# ------------------------------------------------------------------ (a) Bradley-Terry Elo solver
def test_bt_recovers_exact_odds_monotonicity_and_anchor():
    true_elos = np.array([1050.0, 1275.0, 1500.0, 1640.0, 1825.0, 2080.0])
    w, l, d = _exact_odds_matrices(true_elos, n_games=400.0)

    fitted = fit_bradley_terry(w, l, d, anchor_idx=2, anchor_elo=1500.0)
    assert np.allclose(fitted, true_elos, atol=1e-6)
    assert fitted[2] == pytest.approx(1500.0, abs=1e-12)
    assert np.all(np.diff(fitted) > 0.0)

    # Shifting anchor index and anchor Elo shifts all ratings identically
    fitted_alt = fit_bradley_terry(w, l, d, anchor_idx=4, anchor_elo=2000.0)
    assert fitted_alt[4] == pytest.approx(2000.0, abs=1e-12)
    expected_alt = true_elos - true_elos[4] + 2000.0
    assert np.allclose(fitted_alt, expected_alt, atol=1e-6)


def test_bt_symmetry_permutation_and_reversal():
    true_elos = np.array([1200.0, 1450.0, 1500.0, 1720.0, 1910.0])
    w, l, d = _exact_odds_matrices(true_elos, n_games=300.0, draw_frac=0.1)

    base = fit_bradley_terry(w, l, d, anchor_idx=2, anchor_elo=1500.0)

    # Permutation equivariance
    perm = np.array([3, 0, 4, 2, 1])
    w_p = w[np.ix_(perm, perm)]
    l_p = l[np.ix_(perm, perm)]
    d_p = d[np.ix_(perm, perm)]
    new_anchor = int(np.where(perm == 2)[0][0])
    elo_p = fit_bradley_terry(w_p, l_p, d_p, anchor_idx=new_anchor, anchor_elo=1500.0)
    assert np.allclose(elo_p, base[perm], atol=1e-6)

    # Swapping wins and losses negates all Elo differences around the anchor
    elo_rev = fit_bradley_terry(l, w, d, anchor_idx=2, anchor_elo=1500.0)
    assert np.allclose(elo_rev - 1500.0, -(base - 1500.0), atol=1e-6)


def test_bt_draws_count_as_half():
    # Pure draws -> everyone tied at anchor_elo
    k = 4
    n = np.full((k, k), 100.0)
    np.fill_diagonal(n, 0.0)
    z = np.zeros((k, k))
    elo_tied = fit_bradley_terry(z, z, n, anchor_idx=1, anchor_elo=1500.0)
    assert np.allclose(elo_tied, 1500.0, atol=1e-10)

    # Replacing 2k draws with k wins + k losses yields identical BT ratings
    true_elos = np.array([1300.0, 1500.0, 1620.0, 1780.0])
    w_nodraw, l_nodraw, d_nodraw = _exact_odds_matrices(true_elos, n_games=400.0, draw_frac=0.0)
    w_draw, l_draw, d_draw = _exact_odds_matrices(true_elos, n_games=400.0, draw_frac=0.25)
    assert d_draw.sum() > 0.0
    elo_a = fit_bradley_terry(w_nodraw, l_nodraw, d_nodraw, anchor_idx=1, anchor_elo=1500.0)
    elo_b = fit_bradley_terry(w_draw, l_draw, d_draw, anchor_idx=1, anchor_elo=1500.0)
    assert np.allclose(elo_a, elo_b, atol=1e-6)


def test_bt_handles_extreme_separation_and_validates_inputs():
    # Player 0 loses 100% of games, Player 3 wins 100% of games -> fallback_reg keeps Elo finite & ordered
    w = np.array([
        [0, 0, 0, 0],
        [100, 0, 40, 0],
        [100, 60, 0, 0],
        [100, 100, 100, 0],
    ], dtype=np.float64)
    l = w.T.copy()
    d = np.zeros_like(w)
    elo = fit_bradley_terry(w, l, d, anchor_idx=1, anchor_elo=1500.0)
    assert np.all(np.isfinite(elo))
    assert elo[1] == pytest.approx(1500.0, abs=1e-9)
    assert elo[0] < elo[1] < elo[2] < elo[3]

    # Asymmetric wins/losses rejected
    bad_l = l.copy()
    bad_l[0, 1] += 1.0
    with pytest.raises(ValueError, match="losses.T"):
        fit_bradley_terry(w, bad_l, d)


# ------------------------------------------------------------------ (b) Bootstrap 95% CIs
def test_bootstrap_ci_shrinks_with_sample_size_and_respects_anchor():
    true_elos = np.array([1250.0, 1500.0, 1650.0, 1850.0])

    w_small, l_small, d_small = _exact_odds_matrices(true_elos, n_games=50.0, draw_frac=0.05)
    w_large, l_large, d_large = _exact_odds_matrices(true_elos, n_games=800.0, draw_frac=0.05)

    bt_small = bootstrap_elo(w_small, l_small, d_small, anchor_idx=1, anchor_elo=1500.0, n_boot=300, seed=7)
    bt_large = bootstrap_elo(w_large, l_large, d_large, anchor_idx=1, anchor_elo=1500.0, n_boot=300, seed=7)

    # Anchor has zero-width CI
    assert bt_small['ci_low'][1] == pytest.approx(1500.0)
    assert bt_small['ci_high'][1] == pytest.approx(1500.0)
    assert bt_small['std'][1] == pytest.approx(0.0)

    # Non-anchor CIs bracket the true Elos and shrink roughly by sqrt(800/50) = 4x
    non_anchor = [0, 2, 3]
    width_small = bt_small['ci_high'][non_anchor] - bt_small['ci_low'][non_anchor]
    width_large = bt_large['ci_high'][non_anchor] - bt_large['ci_low'][non_anchor]
    assert np.all(width_small > 0.0)
    assert np.all(width_large > 0.0)
    assert np.all(width_large < width_small * 0.4)
    assert np.all(bt_large['ci_low'] <= true_elos + 1e-6)
    assert np.all(bt_large['ci_high'] >= true_elos - 1e-6)
    assert bt_large['prob_stronger'][3, 2] > 0.99
    assert bt_large['prob_stronger'][2, 0] > 0.99

    # Stratified per-seed bootstrap is reproducible and finite
    w1, l1, d1 = _exact_odds_matrices(true_elos, n_games=200.0)
    w2, l2, d2 = _exact_odds_matrices(true_elos, n_games=200.0)
    b1 = bootstrap_elo(w1 + w2, l1 + l2, d1 + d2, anchor_idx=1, anchor_elo=1500.0, n_boot=100, seed=42,
                       per_seed=[(w1, l1, d1), (w2, l2, d2)])
    b2 = bootstrap_elo(w1 + w2, l1 + l2, d1 + d2, anchor_idx=1, anchor_elo=1500.0, n_boot=100, seed=42,
                       per_seed=[(w1, l1, d1), (w2, l2, d2)])
    assert np.allclose(b1['ci_low'], b2['ci_low'])
    assert np.allclose(b1['ci_high'], b2['ci_high'])


# ------------------------------------------------------------------ (c) Mini ladder tournament & chart
def test_ladder_mini_tournament_and_plot_cpu(tmp_path):
    dev = torch.device('cpu')
    torch.manual_seed(0)
    net_a = PolicyValueNet(channels=8, blocks=1).eval()
    net_b = PolicyValueNet(channels=8, blocks=1).eval()
    roster = [
        {'name': 'random', 'kind': 'random', 'spec': 'random'},
        {'name': 'tiny_a', 'kind': 'net', 'spec': 'mem:a'},
        {'name': 'tiny_b', 'kind': 'net', 'spec': 'mem:b'},
    ]
    policies = {
        'random': VecRandom(),
        'tiny_a': NetPolicy(net_a, temperature=1.0, amp=False),
        'tiny_b': NetPolicy(net_b, temperature=1.0, amp=False),
    }
    res = run_tournament(
        roster=roster,
        policies=policies,
        games_per_pair=8,
        seeds=[11, 22],
        device=dev,
        bank_size=16,
        anchor='random',
        anchor_elo=1500.0,
        n_boot=40,
        bootstrap_seed=0,
        temperature=1.0,
        max_plies=16,
        no_battle_plies=10,
        log_fn=lambda _msg: None,
    )
    assert len(res['pairs']) == 3
    for pair in res['pairs']:
        assert pair['games'] == 8
        assert len(pair['per_seed']) == 2
        assert sum(ps['wins'] for ps in pair['per_seed']) == pair['wins']
        assert sum(ps['losses'] for ps in pair['per_seed']) == pair['losses']
        assert sum(ps['draws'] for ps in pair['per_seed']) == pair['draws']
        assert pair['wins'] + pair['losses'] + pair['draws'] == 8
    assert res['elo']['random']['elo'] == pytest.approx(1500.0)
    assert len(res['matrix']['order']) == 3

    png_path = tmp_path / 'ladder.png'
    plot_ladder(res, str(png_path))
    assert png_path.exists() and png_path.stat().st_size > 1000


# ------------------------------------------------------------------ (d) Exploiter smoke tests on CPU
def test_exploit_forbids_init_equals_target(tmp_path):
    ckpt = tmp_path / 'same.pt'
    save(PolicyValueNet(channels=8, blocks=1), str(ckpt), iteration=0)
    with pytest.raises(ValueError, match="must NOT be initialized from its target"):
        exploit_mod.main([
            '--target', str(ckpt),
            '--init', str(ckpt),
            '--out', str(tmp_path / 'out'),
            '--max-iters', '1',
            '--device', 'cpu',
        ])


def test_exploit_smoke_vs_frozen_net_preserves_target_weights(tmp_path):
    torch.manual_seed(0)
    init_path = tmp_path / 'init_clone.pt'
    target_path = tmp_path / 'frozen_target.pt'
    init_net = PolicyValueNet(channels=8, blocks=1)
    target_net = PolicyValueNet(channels=8, blocks=1)
    save(init_net, str(init_path), iteration=0)
    save(target_net, str(target_path), iteration=0)

    init_before = {k: v.detach().clone() for k, v in init_net.state_dict().items()}
    target_before = {k: v.detach().clone() for k, v in target_net.state_dict().items()}
    target_mtime = target_path.stat().st_mtime_ns

    out_dir = tmp_path / 'exploit_net'
    res = exploit_mod.main([
        '--target', str(target_path),
        '--init', str(init_path),
        '--out', str(out_dir),
        '--max-iters', '2',
        '--envs', '8',
        '--steps', '8',
        '--minibatch', '16',
        '--epochs', '1',
        '--value-warmup', '1',
        '--eval-every', '2',
        '--eval-games', '4',
        '--bank-size', '16',
        '--eval-bank-size', '8',
        '--max-plies', '12',
        '--no-battle-plies', '8',
        '--seed', '3',
        '--device', 'cpu',
    ])

    # 1. Frozen target weights in memory and on disk are 100% unchanged
    assert res['target_kind'] == 'net'
    assert res['target_net'] is not None
    for k, ref in target_before.items():
        assert torch.equal(res['target_net'].state_dict()[k].cpu(), ref), f"Target weight {k} changed!"
    assert target_path.stat().st_mtime_ns == target_mtime
    disk_target = load(str(target_path), torch.device('cpu'))
    for k, ref in target_before.items():
        assert torch.equal(disk_target.state_dict()[k], ref)

    # 2. Exploiter learner actually updated its weights during training
    learner_after = res['net'].state_dict()
    changed = any(not torch.equal(learner_after[k].cpu(), init_before[k]) for k in init_before)
    assert changed, "Exploiter learner weights did not update"

    # 3. Output artifacts exist and have valid schema
    assert (out_dir / 'net_final.pt').exists()
    assert (out_dir / 'net_best.pt').exists()
    assert (out_dir / 'state.pt').exists()
    assert (out_dir / 'history.json').exists()
    assert (out_dir / 'log.txt').exists()
    assert (out_dir / 'config.json').exists()

    history = json.loads((out_dir / 'history.json').read_text())
    assert [h['iteration'] for h in history] == [0, 2]
    for entry in history:
        for mode in ('sample', 'greedy', 'greedy_vs_sample'):
            assert mode in entry
            assert entry[mode]['games'] == 4
            assert entry[mode]['wins'] + entry[mode]['losses'] + entry[mode]['draws'] == 4
            assert 0.0 <= entry[mode]['win_rate'] <= 1.0
            assert 0.0 <= entry[mode]['score_rate'] <= 1.0

    log_txt = (out_dir / 'log.txt').read_text()
    assert 'frozen target integrity verified' in log_txt
    assert '[eval] it 0' in log_txt and '[eval] it 2' in log_txt


def test_exploit_smoke_vs_expert_and_plot_curves_cpu(tmp_path):
    torch.manual_seed(1)
    init_path = tmp_path / 'init_clone.pt'
    save(PolicyValueNet(channels=8, blocks=1), str(init_path), iteration=0)

    out_dir = tmp_path / 'exploit_expert'
    res = exploit_mod.main([
        '--target', 'expert',
        '--init', str(init_path),
        '--out', str(out_dir),
        '--max-iters', '2',
        '--envs', '8',
        '--steps', '8',
        '--minibatch', '16',
        '--epochs', '1',
        '--value-warmup', '1',
        '--eval-every', '2',
        '--eval-games', '4',
        '--bank-size', '16',
        '--eval-bank-size', '8',
        '--max-plies', '12',
        '--no-battle-plies', '8',
        '--seed', '5',
        '--device', 'cpu',
    ])

    assert res['target_kind'] == 'expert'
    assert res['target_net'] is None
    assert (out_dir / 'net_final.pt').exists()
    assert (out_dir / 'history.json').exists()
    history = json.loads((out_dir / 'history.json').read_text())
    assert [h['iteration'] for h in history] == [0, 2]
    assert 'sample' in history[-1] and 'greedy' in history[-1]

    # Plot curves mode produces valid PNG
    curves_png = tmp_path / 'exploiter_curves.png'
    exploit_mod.main([
        '--plot-curves', f'expert={out_dir / "history.json"}',
        '--out', str(curves_png),
    ])
    assert curves_png.exists() and curves_png.stat().st_size > 1000


def test_ladder_replot_and_default_roster_discovery(tmp_path):
    dev = torch.device('cpu')
    roster = [
        {'name': 'random', 'kind': 'random', 'spec': 'random'},
        {'name': 'tiny_a', 'kind': 'net', 'spec': 'mem:a'},
    ]
    policies = {
        'random': VecRandom(),
        'tiny_a': NetPolicy(PolicyValueNet(channels=8, blocks=1).eval(), temperature=1.0, amp=False),
    }
    res = run_tournament(
        roster=roster,
        policies=policies,
        games_per_pair=4,
        seeds=[1, 2],
        device=dev,
        bank_size=8,
        anchor='random',
        anchor_elo=1500.0,
        n_boot=20,
        max_plies=12,
        no_battle_plies=8,
        log_fn=lambda _msg: None,
    )
    out_dir = tmp_path / 'replot_dir'
    out_dir.mkdir()
    (out_dir / 'ladder.json').write_text(json.dumps(res))
    re = ladder_mod.main([
        '--out', str(out_dir),
        '--replot',
        '--anchor', 'random',
        '--anchor-elo', '1600.0',
        '--bootstraps', '25',
    ])
    assert re['elo']['random']['elo'] == pytest.approx(1600.0)
    assert (out_dir / 'ladder.png').exists() and (out_dir / 'ladder.png').stat().st_size > 1000

    # Verify default_roster discovers real Phase A checkpoints when present
    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'runs'))
    if os.path.exists(os.path.join(runs_dir, 'ppo_4', 'net_final.pt')):
        dr = ladder_mod.default_roster(runs_dir)
        names = {e['name'] for e in dr}
        for req in ('random', 'pv_clone_3', 'expert', 'expert2', 'expert3', 'ppo_1', 'ppo_2', 'ppo_3', 'ppo_4'):
            assert req in names
        net4 = load(os.path.join(runs_dir, 'ppo_4', 'net_final.pt'), dev)
        assert net4.config['in_planes'] == 59

