"""Reviewer tooling: vec/verify.py statistics and CLI, ONNX export provenance, the deployed web model,
and a vec.diagnose smoke test (all CPU, seconds)."""
import importlib.util
import json
import math
import os

import pytest
import torch

from vec import verify
from vec.net import PolicyValueNet, save

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_ORT = bool(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'))


# ---------------------------------------------------------------- statistics

def test_score_se_counts_draws_as_half_points():
    s = verify.score_stats(500, 400, 100)
    assert s['score'] == pytest.approx(0.55)
    # per-game variance E[x^2] - s^2 with x in {1, 1/2, 0}
    assert s['se'] == pytest.approx(math.sqrt(((500 + 25) / 1000 - 0.55 ** 2) / 1000))
    assert s['se'] < math.sqrt(0.55 * 0.45 / 1000)  # draws shrink the variance
    assert verify.score_stats(0, 0, 10)['z'] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        verify.score_stats(0, 0, 0)


def test_holm_is_monotone_and_matches_hand_values():
    adj = verify.holm({'a': 0.01, 'b': 0.04, 'c': 0.03, 'd': 0.5})
    assert adj['a'] == pytest.approx(0.04)
    assert adj['c'] == pytest.approx(0.09)
    assert adj['b'] == pytest.approx(0.09)  # max(0.08, 0.09): never below a smaller raw p's value
    assert adj['d'] == pytest.approx(0.5)


def _games(score, n=1000, draws=0):
    w = round(score * n - draws / 2)
    return {'wins': w, 'losses': n - w - draws, 'draws': draws}


def test_only_the_preregistered_primary_mode_decides():
    raw = {'sampled_raw': [_games(0.512), _games(0.495)],  # pooled ~50.4%: no difference
           'greedy_safe': [_games(0.555), _games(0.521)]}  # the "best of four"
    res = verify.summarize(raw, 'sampled_raw')
    assert res['verdict'] == 'no clear difference'
    assert res['modes']['greedy_safe']['pooled']['z'] > 2.5
    assert res['modes']['greedy_safe']['p_holm'] >= res['modes']['greedy_safe']['p']
    res2 = verify.summarize(raw, 'greedy_safe')
    assert res2['verdict'] == 'A stronger'
    assert res2['replacement_rule_53_every_seed'] is False  # 52.1% on the second seed
    with pytest.raises(ValueError):
        verify.summarize(raw, 'greedy_raw')


def test_disagreeing_seeds_raise_a_warning():
    res = verify.summarize({'sampled_raw': [_games(0.60), _games(0.50)]}, 'sampled_raw')
    assert res['modes']['sampled_raw']['seed_disagreement_se'] > 2.5
    assert res['warnings'] and 'sampled_raw' in res['warnings'][0]
    assert verify.summarize({'sampled_raw': [_games(0.54), _games(0.53)]}, 'sampled_raw')['warnings'] == []


def test_fresh_seeds_avoid_the_seeds_earlier_reports_used():
    seeds = verify.fresh_seeds(50)
    assert len(set(seeds)) == 50 and not set(seeds) & set(verify.USED_SEEDS)


# ---------------------------------------------------------------- CLI smoke (CPU, tiny nets)

def test_verify_cli_plays_balanced_games_and_writes_json(tmp_path):
    torch.manual_seed(0)
    a, b = tmp_path / 'a.pt', tmp_path / 'b.pt'
    save(PolicyValueNet(channels=8, blocks=1), str(a))
    save(PolicyValueNet(channels=8, blocks=1), str(b))
    out = tmp_path / 'res.json'
    res = verify.main([str(a), str(b), '--games', '8', '--seeds', '3', '4', '--bank-size', '64',
                       '--modes', 'sampled_raw', 'greedy_safe', '--device', 'cpu', '--out', str(out)])
    saved = json.load(open(out))
    assert saved['seeds'] == [3, 4] and saved['primary'] == 'sampled_raw'
    for mode in ('sampled_raw', 'greedy_safe'):
        for r in saved['raw'][mode]:
            assert r['wins'] + r['losses'] + r['draws'] == 8
    assert res['verdict'] in ('A stronger', 'B stronger', 'no clear difference')
    with pytest.raises(SystemExit):
        verify.main([str(a), str(b), '--games', '10', '--seeds', '3'])


# ---------------------------------------------------------------- ONNX provenance + deployed model

@pytest.mark.skipif(not HAVE_ORT, reason='onnx / onnxruntime are not installed')
def test_vec_export_records_the_exact_checkpoint_and_matches_it(tmp_path):
    import numpy as np
    import onnxruntime
    from vec import export_onnx
    torch.manual_seed(1)
    net = PolicyValueNet(channels=8, blocks=1)
    ck = tmp_path / 'n.pt'
    save(net, str(ck))
    out = export_onnx.export(str(ck), str(tmp_path / 'n.onnx'))
    meta = json.load(open(tmp_path / 'n.json'))
    assert meta['source_sha256'] == export_onnx.sha256_of(str(ck))
    assert meta['source_path'].endswith('n.pt')
    x = (torch.rand(3, meta['input_planes'], 10, 10) < 0.2).float()
    with torch.no_grad():
        lt, vt = net.eval()(x)
    lo, vo = onnxruntime.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, {'obs': x.numpy()})
    assert np.abs(lo - lt.numpy()).max() < 1e-4 and np.abs(vo - vt.numpy()).max() < 1e-5


@pytest.mark.skipif(not HAVE_ORT, reason='onnx / onnxruntime are not installed')
def test_deployed_web_model_is_the_checkpoint_its_metadata_names():
    """web/model.onnx is what people play against: it must be the checkpoint model.json names."""
    import numpy as np
    import onnxruntime
    from vec.export_onnx import sha256_of
    from vec.net import load
    meta_path = os.path.join(PROJECT_ROOT, 'web', 'model.json')
    meta = json.load(open(meta_path))
    assert 'source_path' in meta and 'source_sha256' in meta, 're-export with vec.export_onnx'
    ck = os.path.normpath(os.path.join(PROJECT_ROOT, meta['source_path']))
    if not os.path.exists(ck):
        pytest.skip(f'checkpoint {meta["source_path"]} not present (runs/ not checked out)')
    assert sha256_of(ck) == meta['source_sha256'], 'checkpoint changed since export'
    net = load(ck, torch.device('cpu')).to(memory_format=torch.contiguous_format).float().eval()
    assert net.config == meta['network']
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(4, meta['input_planes'], 10, 10, generator=g) < 0.2).float()
    with torch.no_grad():
        lt, vt = net(x)
    sess = onnxruntime.InferenceSession(os.path.join(PROJECT_ROOT, 'web', 'model.onnx'),
                                        providers=['CPUExecutionProvider'])
    lo, vo = sess.run(None, {'obs': x.numpy()})
    assert np.abs(lo - lt.numpy()).max() < 1e-3 and np.abs(vo - vt.numpy()).max() < 1e-4


# ---------------------------------------------------------------- vec.diagnose smoke

def test_diagnose_counts_every_game_and_reports():
    from vec.diagnose import diagnose, report
    from vec.expert import VecRandom
    from vec.net import NetPolicy
    from vec.setups import diverse_setup_bank
    torch.manual_seed(2)
    dev = torch.device('cpu')
    d = diagnose(NetPolicy(PolicyValueNet(channels=8, blocks=1).eval(), temperature=1.0), VecRandom(),
                 8, diverse_setup_bank(64, seed=0), dev)
    assert d['wins'] + d['losses'] + d['draws'] == d['games'] == 8
    assert d['sure_loss'][1] <= d['sure_loss'][0]
    assert 'attacks per game' in report(d)
