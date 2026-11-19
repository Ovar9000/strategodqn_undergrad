"""
Phase A3 tool tests:
  - vec/health.py & vec/plot_runs.py (all 6 anomaly flags on synthetic logs + real runs/ppo_1..4)
  - vec/record_games.py (schema, board/battle/belief consistency, sure_loss_non_scout detection)
  - web/play.html ("Download game" export schema parity with vec/record_games.py)
  - web/replay.html (static self-contained checks + Node replay-core verification on both sources)
"""

import json
import math
import os
import re
import shutil
import subprocess

import pytest
import torch

from environment.piece import PieceType
from vec import health, plot_runs, record_games
from vec.engine import BOMB, FLAG, MINER, SCOUT, SPY, VecStratego
from vec.net import PolicyValueNet
from vec.setups import diverse_setup_bank

CAPTAIN = PieceType.CAPTAIN.value
MARSHAL = PieceType.MARSHAL.value

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.normpath(os.path.join(ROOT, '..', 'runs'))
NODE = shutil.which('node')


# ---------------------------------------------------------------------------
# Helpers for synthetic PPO logs
# ---------------------------------------------------------------------------

def _ppo_line(it, samples, pg=-0.005, v=0.045, ent=1.0, kl=0.010, clip=0.08,
              ev=0.82, belief=0.38, length=250, ends=(2000, 200, 10, 0)):
    f, nm, nb, lim = ends
    return (
        f"[ppo] it {it} samples {samples}M roll 8s train 8s lr 1.00e-04 "
        f"pg {pg} v {v} ent {ent} kl {kl} mkl 0.0500 clip {clip} ev {ev} belief {belief} | "
        f"self 900/0/5 expert 400/50/5 random 50/0/0 snap 400/350/5 | "
        f"len {length} ends flag {f} nomove {nm} nobattle {nb} limit {lim}"
    )


def _eval_line(it, mode, opp, w, l, d, g=300):
    return f"[eval] it {it} {mode} vs {opp}: {w}/{l}/{d} of {g} ({100*w/g:.1f}% wins, 200 plies)"


def _write_log(tmp_path, run_name, lines):
    d = tmp_path / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(d / 'log.txt')


# ---------------------------------------------------------------------------
# 1. Health checks (synthetic logs + real runs/ppo_1..4 + plot_runs hook)
# ---------------------------------------------------------------------------

def test_health_clean_run_has_zero_flags(tmp_path):
    lines = [
        _ppo_line(1, '0.10', kl=0.0000, ev=0.75, ends=(300, 10, 0, 0)),
        _ppo_line(5, '0.50', kl=0.0100, ev=0.80, ends=(2000, 200, 20, 0)),
        _ppo_line(10, '1.00', kl=0.0110, ev=0.82, ends=(2000, 200, 22, 0)),
        _ppo_line(15, '1.50', kl=0.0095, ev=0.84, ends=(2000, 200, 18, 0)),
        _ppo_line(20, '2.00', kl=0.0120, ev=0.85, ends=(2000, 200, 25, 0)),
        _eval_line(10, 'sample', 'expert', 270, 28, 2),
        _eval_line(20, 'sample', 'expert', 275, 23, 2),
        _eval_line(10, 'sample', 'start', 200, 100, 0),
        _eval_line(20, 'sample', 'start', 170, 130, 0),  # non-expert drop must not trigger expert regression
    ]
    _write_log(tmp_path, 'ppo_1', lines)
    runs = plot_runs.load_runs(str(tmp_path))
    flags = health.check_runs(runs)['ppo_1']
    assert flags == [], flags


def test_health_flags_all_six_anomalies_and_plots(tmp_path):
    lines = [
        # Warmup it=1 (kl=0 must NOT cause kl_spike on it=5)
        _ppo_line(1, '0.10', kl='0.0000', ev=0.72, ent=1.10, ends=(300, 10, 1, 0)),
        _ppo_line(5, '0.50', kl='0.0100', ev=0.80, ent=1.05, ends=(2000, 200, 20, 0)),
        _ppo_line(10, '1.00', kl='0.0100', ev=0.81, ent=1.02, ends=(2000, 200, 20, 0)),
        _ppo_line(15, '1.50', kl='0.0100', ev=0.82, ent=0.98, ends=(2000, 200, 20, 0)),
        # Flag 1: entropy_collapse (ent < 0.5)
        _ppo_line(20, '2.00', kl='0.0100', ev=0.80, ent=0.42, ends=(2000, 200, 20, 0)),
        # Flag 2: kl_spike (kl = 0.045 > 3 * 0.010)
        _ppo_line(25, '2.50', kl='0.0450', ev=0.80, ent=0.90, ends=(2000, 200, 20, 0)),
        # Flag 3: ev_drop (ev < 0.5)
        _ppo_line(30, '3.00', kl='0.0100', ev=0.31, ent=0.90, ends=(2000, 200, 20, 0)),
        # Flag 4a: nobattle_surge (relative surge: ~3.5% > 2.5x ~0.9%)
        _ppo_line(35, '3.50', kl='0.0100', ev=0.80, ent=0.90, ends=(1900, 200, 76, 0)),
        # Flag 4b: nobattle_surge (absolute >= 5%)
        _ppo_line(40, '4.00', kl='0.0100', ev=0.80, ent=0.90, ends=(1800, 100, 200, 0)),
        # Flag 5: nan_inf (pg nan and kl inf)
        _ppo_line(45, '4.50', pg='nan', v='inf', kl='0.0100', ev=0.80, ent=0.90, ends=(2000, 200, 20, 0)),
        # Flag 6: eval_regression (> 5 pp drop vs expert2)
        _eval_line(20, 'sample', 'expert2', 264, 30, 6),  # 88.0%
        _eval_line(40, 'sample', 'expert2', 240, 54, 6),  # 80.0% (-8.0 pp)
    ]
    _write_log(tmp_path, 'ppo_1', lines)
    runs = plot_runs.load_runs(str(tmp_path))
    flags = health.check_runs(runs)['ppo_1']
    kinds = {f.kind for f in flags}
    assert kinds == set(health.CHECK_KINDS), f"Expected all 6 kinds, got {kinds}"

    # Verify specific iterations flagged
    by_kind = {}
    for fl in flags:
        by_kind.setdefault(fl.kind, []).append(fl)
    assert [f.it for f in by_kind['entropy_collapse']] == [20]
    assert [f.it for f in by_kind['kl_spike']] == [25]
    assert [f.it for f in by_kind['ev_drop']] == [30]
    assert [f.it for f in by_kind['nobattle_surge']] == [35, 40]
    assert [f.it for f in by_kind['eval_regression']] == [40]
    assert {f.metric for f in by_kind['nan_inf']} == {'pg', 'v'}
    assert all(f.it == 45 for f in by_kind['nan_inf'])

    # End-to-end report + plot generation (must handle NaN/inf without crashing)
    out_png = str(tmp_path / 'plots' / 'ppo_training.png')
    out_txt = str(tmp_path / 'plots' / 'health.txt')
    plot_runs.main(['--runs', str(tmp_path), '--out', out_png, '--health-out', out_txt])
    assert os.path.exists(out_png) and os.path.getsize(out_png) > 5000
    report = open(out_txt, encoding='utf-8').read()
    for kind in health.CHECK_KINDS:
        assert kind in report


def test_health_on_real_ppo_runs_and_ppo_isolation():
    # health.py must never be imported inside vec/ppo.py
    ppo_src = open(os.path.join(ROOT, 'vec', 'ppo.py'), encoding='utf-8').read()
    assert 'health' not in ppo_src

    if not os.path.exists(os.path.join(RUNS_DIR, 'ppo_4', 'log.txt')):
        pytest.skip('real runs/ppo_1..4 logs not present')
    runs = plot_runs.load_runs(RUNS_DIR)
    for r in ('ppo_1', 'ppo_2', 'ppo_3', 'ppo_4'):
        assert r in runs
    flags = health.check_runs(runs)
    # No NaN/inf, entropy collapse, or KL spikes in ppo_1..4
    for r in ('ppo_1', 'ppo_2', 'ppo_3', 'ppo_4'):
        kinds = {f.kind for f in flags[r]}
        assert 'nan_inf' not in kinds
        assert 'entropy_collapse' not in kinds
        assert 'kl_spike' not in kinds
    # Known real signals caught on ppo_1..4
    assert 'ev_drop' in {f.kind for f in flags['ppo_1']}
    assert 'nobattle_surge' in {f.kind for f in flags['ppo_1']}
    assert 'eval_regression' in {f.kind for f in flags['ppo_1']}
    assert 'eval_regression' in {f.kind for f in flags['ppo_2']}
    assert flags['ppo_3'] == []
    assert 'eval_regression' in {f.kind for f in flags['ppo_4']}


# ---------------------------------------------------------------------------
# 2. Game recorder (vec/record_games.py) on CPU
# ---------------------------------------------------------------------------

def test_record_games_cpu_smoke_and_schema(tmp_path):
    torch.manual_seed(0)
    dev = torch.device('cpu')
    net = PolicyValueNet(channels=16, blocks=1).to(dev).eval()
    bank = diverse_setup_bank(16, seed=7)

    for opp in ('expert', 'self'):
        reps = record_games.record_games(
            net=net, opponent=opp, num_games=1, device=dev,
            temperature=1.0, seed=3, max_plies=40, no_battle_plies=25,
            bank=bank, model_name='tiny_cpu_net',
        )
        assert len(reps) == 1
        rep = reps[0]
        errs = record_games.verify_replay(rep)
        assert errs == [], f'verify_replay errors for {opp}: {errs}'
        assert rep['schema_version'] == 1
        assert len(rep['setups']['red']) == 40
        assert len(rep['setups']['blue']) == 40
        assert rep['meta']['num_plies'] == len(rep['plies']) > 0
        assert rep['meta']['end_type'] in ('flag', 'no_moves', 'no_battle', 'turn_limit')
        net_side = rep['meta']['net_side']
        net_plies = [p for p in rep['plies'] if net_side == 0 or p['side'] == net_side]
        assert len(net_plies) > 0
        for p in net_plies:
            assert p['top5'] and 1 <= len(p['top5']) <= 5
            probs = [item['prob'] for item in p['top5']]
            assert probs == sorted(probs, reverse=True)
            assert -1.0 <= p['value'] <= 1.0
            assert isinstance(p['belief'], dict) and len(p['belief']) > 0
            sample_b = next(iter(p['belief'].values()))
            assert 1 <= sample_b['top_rank'] <= 12
            assert 1 <= sample_b['true_rank'] <= 12
            assert len(sample_b['probs']) == 12
            assert abs(sum(sample_b['probs']) - 1.0) < 0.03

        paths = record_games.save_replays(reps, str(tmp_path), prefix=f'smoke_{opp}')
        assert len(paths) == 1 and os.path.exists(paths[0])

    idx_path = record_games.write_index(str(tmp_path))
    idx_data = json.load(open(idx_path, encoding='utf-8'))
    assert len(idx_data['games']) == 2


def test_record_games_sure_loss_non_scout_detection():
    """Construct a deterministic opening sequence with:
      - Ply 0: Red Scout attacks HIDDEN Blue Marshal -> loses, reveals Marshal (NOT sure_loss)
      - Ply 2: Red Captain (non-Scout) attacks REVEALED Blue Marshal -> sure_loss=True, sure_loss_non_scout=True
      - Ply 4: Red Scout attacks REVEALED Blue Marshal -> sure_loss=True, sure_loss_non_scout=False
    """
    dev = torch.device('cpu')
    net = PolicyValueNet(channels=16, blocks=1).to(dev).eval()
    base = diverse_setup_bank(10, seed=11)
    red_setup = base[0].clone()
    blue_setup = base[1].clone()

    def put_at(setup, pos, rank):
        cur = (setup == rank).nonzero().flatten()
        # swap first occurrence that isn't already locked
        idx = int(cur[-1])
        setup[pos], setup[idx] = setup[idx].clone(), setup[pos].clone()

    # In Red's canonical frame (60..99), index 0..9 is row 6 (squares 60..69), 10..19 is row 7 (70..79)
    # Place Red Scout at 60 (row 6 col 0), Red Captain at 70 (row 7 col 0), Red Scout #2 at 61 (row 6 col 1)
    # Avoid lake columns (2, 3, 6, 7): col 0 and col 1 are clear!
    put_at(red_setup, 0, SCOUT)      # sq 60
    put_at(red_setup, 10, CAPTAIN)   # sq 70
    put_at(red_setup, 1, SCOUT)      # sq 61
    # Place Blue Marshal at Blue's canonical sq 69 -> abs sq 99 - 69 = 30 (row 3 col 0)!
    # Wait: row 6 col 0 -> row 5 col 0 -> row 4 col 0 -> row 3 col 0.
    # Blue's canonical sq 69 is index 9 in Blue's setup (since 60 + 9 = 69 -> flip -> abs 30 = row 3 col 0).
    put_at(blue_setup, 9, MARSHAL)   # abs sq 30 (row 3 col 0)
    # Give Blue a movable piece at abs sq 39 (Blue canonical sq 60 -> index 0) to make quiet moves
    put_at(blue_setup, 0, MINER)     # abs sq 39 (row 3 col 9)

    env = VecStratego(1, dev, max_plies=10, no_battle_plies=50)
    t = env.t
    p2a = t.pair_to_action

    # Build forced action sequence (canonical coordinates for each side to move):
    # Ply 0 (Red, side +1): Scout slides 60 -> 30 (attacks hidden Blue Marshal at 30 -> loses, Marshal revealed at 30)
    # Ply 1 (Blue, side -1): Blue Miner at abs 39 (canon 60) steps to abs 49 (canon 50)
    # Ply 2 (Red, side +1): Captain at 70 -> 60 (quiet)
    # Ply 3 (Blue, side -1): Blue Miner abs 49 -> 59 (canon 50 -> 40)
    # Ply 4 (Red, side +1): Captain at 60 -> 50 (quiet)
    # Ply 5 (Blue, side -1): Blue Miner abs 59 -> 49 (canon 40 -> 50)
    # Ply 6 (Red, side +1): Captain at 50 -> 40 (quiet)
    # Ply 7 (Blue, side -1): Blue Miner abs 49 -> 39 (canon 50 -> 60)
    # Ply 8 (Red, side +1): Captain at 40 -> 30 (ATTACKS REVEALED BLUE MARSHAL -> sure_loss_non_scout!)
    # Ply 9 (Blue, side -1): Blue Marshal at abs 30 (canon 69) steps to abs 31 (canon 68)? Keep Marshal at 30, move Miner 39->49 (canon 60->50)!
    # Wait: Blue's 2-square rule after 49->59->49->39: 39->49 is NOT repeating 49->39->49 (last two were 59->49, 49->39)!
    # Let's move Miner 39->38? Wait 38 is occupied. Move 39->49 (canon 60->50) is legal!
    # Ply 10 (Red, side +1): Red Scout #2 at 61 -> 31 -> wait, Scout can slide 61->41 (quiet), or move 61->60->30.
    # Let's just slide Red Scout #2 at 61 -> 60 on ply 10? Wait 61->60 is quiet, then ply 11 Blue 49->59, ply 12 Red Scout #2 slides 60->30 (attacks revealed Blue Marshal -> sure_loss=True, sure_loss_non_scout=False)!
    # Wait: let's check Blue's two-square rule on ply 9:
    #   Ply 1: 60->50
    #   Ply 3: 50->40
    #   Ply 5: 40->50  (hist: [40->50, 50->40])
    #   Ply 7: 50->60  (hist: [50->60, 40->50] -> NOT reversing!)
    #   Ply 8: BATTLE! (clears rep history for BOTH sides!)
    #   Ply 9: 60->50  (legal!)
    #   Ply 11: 50->40 (legal!)
    forced_pairs = [
        (60, 30),  # ply 0  Red  Scout -> hidden Blue Marshal (reveals Marshal!)
        (60, 50),  # ply 1  Blue Miner quiet
        (70, 60),  # ply 2  Red  Captain quiet
        (50, 40),  # ply 3  Blue Miner quiet
        (60, 50),  # ply 4  Red  Captain quiet
        (40, 50),  # ply 5  Blue Miner quiet
        (50, 40),  # ply 6  Red  Captain quiet
        (50, 60),  # ply 7  Blue Miner quiet
        (40, 30),  # ply 8  Red  Captain -> REVEALED Blue Marshal (R10 sure_loss_non_scout!)
        (60, 50),  # ply 9  Blue Miner quiet
        (61, 60),  # ply 10 Red  Scout #2 quiet
        (50, 40),  # ply 11 Blue Miner quiet
        (60, 30),  # ply 12 Red  Scout #2 -> REVEALED Blue Marshal (sure_loss=True, non_scout=False)
    ]
    forced = [torch.tensor([int(p2a[s, d])], dtype=torch.long) for s, d in forced_pairs]

    reps = record_games._record_batch(
        net=net, opp_policy=None, is_self_play=False,
        mover_setup=red_setup[None], other_setup=blue_setup[None],
        start_side=torch.tensor([1]), a_side=torch.tensor([1]),
        device=dev, temperature=1.0, seed=0,
        max_plies=len(forced), no_battle_plies=50,
        model_name='test_net', model_label='test_net', opp_name='scripted',
        amp=False, forced_actions=forced,
    )
    rep = reps[0]
    errs = record_games.verify_replay(rep)
    assert errs == [], errs

    p0, p8, p12 = rep['plies'][0], rep['plies'][8], rep['plies'][12]
    # Ply 0: Scout vs HIDDEN Marshal
    assert p0['battle']['defender'] == MARSHAL
    assert p0['battle']['defender_was_revealed'] is False
    assert p0['sure_loss'] is False and p0['sure_loss_non_scout'] is False
    # Ply 8: Captain vs REVEALED Marshal -> R10 non-Scout sure-loss!
    assert p8['piece'] == CAPTAIN
    assert p8['battle']['defender'] == MARSHAL
    assert p8['battle']['defender_was_revealed'] is True
    assert p8['sure_loss'] is True and p8['sure_loss_non_scout'] is True
    # Ply 12: Scout vs REVEALED Marshal -> Scout sure-loss (NOT non-Scout)
    assert p12['piece'] == SCOUT
    assert p12['battle']['defender_was_revealed'] is True
    assert p12['sure_loss'] is True and p12['sure_loss_non_scout'] is False

    bl = rep['meta']['blunders']
    assert bl['sure_loss_total'] == 2
    assert bl['sure_loss_non_scout'] == 1
    assert bl['sure_loss_plies'] == [8]
    assert bl['sure_loss_all_plies'] == [8, 12]


# ---------------------------------------------------------------------------
# 3. Web tools: play.html Download export & replay.html static + Node checks
# ---------------------------------------------------------------------------

def test_web_replay_and_play_static_sanity():
    play_html = open(os.path.join(ROOT, 'web', 'play.html'), encoding='utf-8').read()
    replay_html = open(os.path.join(ROOT, 'web', 'replay.html'), encoding='utf-8').read()

    # play.html has Download button + pure replay-export helper
    assert 'id="downloadBtn"' in play_html
    assert 'id="replay-export"' in play_html
    assert 'StrategoReplayExport' in play_html

    # replay.html is standalone (no external CDN <script src="http...">)
    assert not re.search(r'<script[^>]+src=["\']https?://', replay_html, re.IGNORECASE)
    for token in (
        'id="replay-core"',
        'id="fileInput"',
        'dropOverlay',
        'id="scrubber"',
        'id="btnPlay"',
        'id="btnNextBattle"',
        'id="btnNextBlunder"',
        'data-view="full"',
        'data-view="red"',
        'data-view="blue"',
        'id="top5List"',
        'id="valSvg"',
        'id="beliefBody"',
        'ArrowLeft',
        'ArrowRight',
    ):
        assert token in replay_html, f'missing {token} in web/replay.html'


@pytest.mark.skipif(NODE is None, reason='node is not installed')
def test_play_html_export_matches_schema_and_loads_in_replay_html(tmp_path):
    """Simulate a game with play.html's engine + StrategoReplayExport in Node, verify the
    exported JSON with Python's verify_replay(), and feed BOTH play.html's export AND
    record_games.py's output into replay.html's StrategoReplayCore in Node."""
    # First produce a tiny Python replay
    torch.manual_seed(1)
    net = PolicyValueNet(channels=16, blocks=1).eval()
    py_rep = record_games.record_games(
        net=net, opponent='expert', num_games=1, device=torch.device('cpu'),
        temperature=1.0, seed=5, max_plies=30, no_battle_plies=20,
        bank=diverse_setup_bank(8, seed=9), model_name='tiny_py',
    )[0]
    py_path = tmp_path / 'py_replay.json'
    py_path.write_text(json.dumps(py_rep), encoding='utf-8')
    js_out_path = tmp_path / 'play_export.json'

    node_script = tmp_path / 'check_web.js'
    node_script.write_text(
        r"""
'use strict';
const fs = require('fs');
const vm = require('vm');
const [playPath, replayPath, pyReplayPath, jsOutPath] = process.argv.slice(2);

const playHtml = fs.readFileSync(playPath, 'utf8');
const engineSrc = /<script id="engine">([\s\S]*?)<\/script>/.exec(playHtml)[1];
const exportSrc = /<script id="replay-export">([\s\S]*?)<\/script>/.exec(playHtml)[1];

const ctx = { globalThis: {} };
vm.createContext(ctx);
vm.runInContext(engineSrc, ctx);
vm.runInContext(exportSrc, ctx);
const E = ctx.globalThis.StrategoEngine;
const R = ctx.globalThis.StrategoReplayExport;

// Deterministic PRNG
let s = 1234567;
const rng = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
const p1 = E.heuristicSetup(1, rng);
const p2 = E.structuredSetup(-1, rng);
const game = new E.Game({ maxTurns: 40, noBattleLimit: 25 }).reset(p1, p2, 1);
const rec = R.createRecorder(game, { model: 'ppo_4.onnx', red: 'human', blue: 'ppo_4.onnx', netSide: -1, temperature: 0.5 });

for (let ply = 0; ply < 30 && !game.over; ply++) {
  const moves = game.legalMoves(game.current);
  const mv = E.heuristicMove(game, game.current, rng);
  const pre = R.capturePreMove(game, mv);
  const info = game.step(mv);
  let botMeta = null;
  if (pre.side === -1) {
    const top5 = moves.slice(0, 5).map((m, i) => ({
      move: [m[0][0] * 10 + m[0][1], m[1][0] * 10 + m[1][1]],
      prob: Number((0.4 / (i + 1)).toFixed(4))
    }));
    botMeta = { top5, playedProb: 0.4, value: Number((0.15 - ply * 0.01).toFixed(4)) };
  }
  R.recordPly(rec, pre, info, botMeta);
}
const exported = R.buildReplayJson(rec, game);
fs.writeFileSync(jsOutPath, JSON.stringify(exported));

// Now load replay.html's pure replay-core and verify BOTH replays parse & step identically
const replayHtml = fs.readFileSync(replayPath, 'utf8');
const coreSrc = /<script id="replay-core">([\s\S]*?)<\/script>/.exec(replayHtml)[1];
const rctx = { globalThis: {} };
vm.createContext(rctx);
vm.runInContext(coreSrc, rctx);
const C = rctx.globalThis.StrategoReplayCore;

const pyData = JSON.parse(fs.readFileSync(pyReplayPath, 'utf8'));
for (const [label, data] of [['play_html', exported], ['record_games', pyData]]) {
  const built = C.buildStates(data);
  if (built.states.length !== data.plies.length + 1) {
    throw new Error(`${label}: state count mismatch`);
  }
  if (built.values.length === 0) {
    throw new Error(`${label}: expected value curve points`);
  }
}
console.log('ok');
""",
        encoding='utf-8',
    )

    res = subprocess.run(
        [
            NODE,
            str(node_script),
            os.path.join(ROOT, 'web', 'play.html'),
            os.path.join(ROOT, 'web', 'replay.html'),
            str(py_path),
            str(js_out_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert res.stdout.strip() == 'ok'

    # Validate play.html's exported JSON with Python's verify_replay()!
    exported = json.loads(js_out_path.read_text(encoding='utf-8'))
    errs = record_games.verify_replay(exported)
    assert errs == [], f'play.html export failed verify_replay: {errs}'
    assert exported['meta']['source'] == 'play_html'
    assert exported['meta']['model'] == 'ppo_4.onnx'
    bot_plies = [p for p in exported['plies'] if p['side'] == -1]
    assert bot_plies and all(p['top5'] is not None and p['value'] is not None for p in bot_plies)
