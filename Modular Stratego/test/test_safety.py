"""
Tests for Phase B1 play-time safety filter (vec/safety.py and web/play.html) and Phase B2
exploiter extensions (--allow-same-init, --target-safe).
"""

import json
import os
import re
import shutil
import subprocess

import pytest
import torch

from environment.piece import PieceType
from vec import exploit as exploit_mod
from vec.engine import BOMB, FLAG, MINER, NUM_ACTIONS, SCOUT, SPY, VecStratego, tables
from vec.net import NetPolicy, PolicyValueNet, save
from vec.safety import safe_legal_mask, safe_mask_for_env, unsafe_table
from vec.setups import diverse_setup_bank

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAY_HTML = os.path.join(ROOT, 'web', 'play.html')
NODE = shutil.which('node')

SERGEANT = PieceType.SERGEANT.value
CAPTAIN = PieceType.CAPTAIN.value
GENERAL = PieceType.GENERAL.value
MARSHAL = PieceType.MARSHAL.value


def test_unsafe_table_rules():
    t = tables(torch.device('cpu'))
    u = unsafe_table(t)
    assert u.shape == (13, 13)
    # Row/col 0 (empty) is never unsafe
    assert not u[0].any() and not u[:, 0].any()

    # Non-Scout attacking revealed stronger piece -> forbidden
    assert bool(u[CAPTAIN, MARSHAL])
    assert bool(u[GENERAL, MARSHAL])
    assert bool(u[MINER, SERGEANT])
    assert bool(u[SPY, SCOUT])
    assert bool(u[SPY, MINER])

    # Non-Miner attacking revealed Bomb -> forbidden (INCLUDING Scout!)
    for r in range(SPY, MARSHAL + 1):
        if r == MINER:
            assert not bool(u[r, BOMB]), "Miner must be allowed to defuse revealed Bomb"
        else:
            assert bool(u[r, BOMB]), f"Rank {r} attacking revealed Bomb must be forbidden"

    # Special allowed wins & equal trades
    assert not bool(u[SPY, MARSHAL]), "Spy attacking revealed Marshal wins and must be allowed"
    assert not bool(u[MARSHAL, MARSHAL]), "Equal-rank trade must be allowed"
    assert not bool(u[CAPTAIN, CAPTAIN]), "Equal-rank trade must be allowed"
    assert not bool(u[SPY, SPY]), "Equal-rank Spy trade must be allowed"
    assert not bool(u[MARSHAL, GENERAL]), "Stronger attacking weaker must be allowed"
    for r in range(SPY, MARSHAL + 1):
        assert not bool(u[r, FLAG]), "Attacking revealed Flag must always be allowed"

    # Scout attacking revealed stronger mobile piece -> allowed by spec
    assert not bool(u[SCOUT, MARSHAL])
    assert not bool(u[SCOUT, CAPTAIN])
    assert not bool(u[SCOUT, MINER])


def test_safe_mask_never_leaks_hidden_information():
    """Hidden enemy ranks must never affect the safety mask."""
    dev = torch.device('cpu')
    env = VecStratego(2, dev)
    t = env.t
    env.own.zero_()
    env.opp.zero_()
    env.opp_rev.zero_()
    env.done.zero_()

    # Game 0: Captain at 50 adjacent to HIDDEN Bomb at 40 and empty 60
    # Game 1: Captain at 50 adjacent to HIDDEN Flag at 40 and empty 60
    env.own[:, 50] = CAPTAIN
    env.opp[0, 40] = BOMB
    env.opp[1, 40] = FLAG
    env.legal = env._legal_mask(env.own, env.opp)

    m = safe_mask_for_env(env)
    assert torch.equal(m[0], env.legal[0])
    assert torch.equal(m[1], env.legal[1])
    assert torch.equal(m[0], m[1])

    # Once revealed, Bomb attack is masked out in Game 0 while Flag attack stays legal in Game 1
    env.opp_rev[:, 40] = True
    m_rev = safe_mask_for_env(env)
    atk_act = int(t.pair_to_action[50, 40])
    quiet_act = int(t.pair_to_action[50, 60])
    assert not bool(m_rev[0, atk_act]) and bool(m_rev[0, quiet_act])
    assert bool(m_rev[1, atk_act]) and bool(m_rev[1, quiet_act])


def test_safe_mask_fallback_when_only_unsafe_moves_exist():
    """When every legal move is an unsafe attack, fall back to the original legal mask."""
    dev = torch.device('cpu')
    env = VecStratego(2, dev)
    t = env.t
    env.own.zero_()
    env.opp.zero_()
    env.opp_rev.zero_()
    env.done.zero_()

    # Game 0: Captain trapped in corner 0 by own Bombs at 1 and revealed enemy Marshal at 10 -> only move is 0->10
    env.own[0, 0] = CAPTAIN
    env.own[0, 1] = BOMB
    env.opp[0, 10] = MARSHAL
    env.opp_rev[0, 10] = True

    # Game 1: Scout trapped in corner 0 by own Bomb at 1 and revealed enemy Bomb at 10 -> only move is 0->10
    env.own[1, 0] = SCOUT
    env.own[1, 1] = BOMB
    env.opp[1, 10] = BOMB
    env.opp_rev[1, 10] = True

    env.legal = env._legal_mask(env.own, env.opp)
    assert env.legal[0].sum() == 1 and env.legal[1].sum() == 1

    filtered = safe_mask_for_env(env)
    assert torch.equal(filtered, env.legal), "Fallback must preserve legal moves when no safe move exists"


def test_net_policy_safe_flag_blocks_sure_loss_blunder():
    dev = torch.device('cpu')
    env = VecStratego(1, dev)
    t = env.t
    env.own.zero_()
    env.opp.zero_()
    env.opp_rev.zero_()
    env.done.zero_()
    env.own[0, 50] = CAPTAIN
    env.opp[0, 40] = MARSHAL
    env.opp_rev[0, 40] = True
    env.legal = env._legal_mask(env.own, env.opp)

    bad_act = int(t.pair_to_action[50, 40])
    good_act = int(t.pair_to_action[50, 60])

    class ConstantLogitNet(PolicyValueNet):
        def forward(self, obs, with_belief=False):
            logits = torch.full((obs.size(0), NUM_ACTIONS), -5.0, device=obs.device)
            logits[:, bad_act] = 10.0   # network strongly prefers the sure-loss blunder
            logits[:, good_act] = 2.0
            val = torch.zeros(obs.size(0), device=obs.device)
            return logits, val

    net = ConstantLogitNet(channels=8, blocks=1).to(dev).eval()
    raw_pol = NetPolicy(net, temperature=0.0, amp=False, safe=False)
    safe_pol = NetPolicy(net, temperature=0.0, amp=False, safe=True)

    assert int(raw_pol.act(env)[0]) == bad_act
    assert int(safe_pol.act(env)[0]) == good_act


def test_safety_never_imported_in_ppo_trainer():
    ppo_src = open(os.path.join(ROOT, 'vec', 'ppo.py'), encoding='utf-8').read()
    assert 'safety' not in ppo_src


def test_play_html_has_safe_filter_toggle_on_by_default():
    html = open(PLAY_HTML, encoding='utf-8').read()
    assert re.search(r'<input[^>]*id="safeFilter"[^>]*checked', html), "safeFilter toggle must be checked by default"
    assert 'filterSafeMoves' in html


@pytest.mark.skipif(NODE is None, reason='node not available')
def test_js_and_python_safety_filters_agree_across_games_and_edge_cases(tmp_path):
    """Compare Python safe_legal_mask against play.html's StrategoEngine.filterSafeMoves on
    (a) all 12x12 rank pairs in isUnsafeAttack, (b) constructed edge cases (revealed Bomb vs
    Scout/Miner, Spy vs Marshal, equal trade, forced-suicide fallback), and (c) real multi-ply
    games with battles and reveals for both Player 1 and Player -1."""
    dev = torch.device('cpu')
    t = tables(dev)
    u_py = unsafe_table(t).tolist()

    # Record 3 games with frequent battles so both players accumulate revealed pieces and Bombs
    torch.manual_seed(42)
    bank = diverse_setup_bank(20, seed=42)
    records = []
    for g_idx in range(3):
        env = VecStratego(1, dev, max_plies=160, no_battle_plies=120)
        s1, s2 = bank[2 * g_idx], bank[2 * g_idx + 1]
        start_side = 1 if g_idx % 2 == 0 else -1
        mover_s, other_s = (s1, s2) if start_side == 1 else (s2, s1)
        env.reset(torch.tensor([0]), mover_s[None], other_s[None], torch.tensor([start_side]))

        # Convert canonical setups (squares 60..99) to absolute (piece, [r, c]) placements for JS
        p1_setup = [[int(s1[i]), [6 + i // 10, i % 10]] for i in range(40)]
        p2_setup = [[int(s2[i]), [3 - i // 10, 9 - i % 10]] for i in range(40)]

        plies = []
        for _ in range(120):
            if bool(env.done[0]):
                break
            player = int(env.side[0])
            raw_acts = env.legal[0].nonzero().flatten().tolist()
            safe_acts = safe_mask_for_env(env)[0].nonzero().flatten().tolist()
            # Pick a move preferring attacks so reveals happen frequently
            dst_sqs = t.act_to[env.legal[0].nonzero().flatten()].clamp(max=99)
            is_atk = env.opp[0, dst_sqs] > 0
            if is_atk.any() and (len(plies) % 2 == 0):
                atk_acts = env.legal[0].nonzero().flatten()[is_atk]
                act = int(atk_acts[torch.randint(0, atk_acts.numel(), (1,)).item()])
            else:
                act = int(raw_acts[torch.randint(0, len(raw_acts), (1,)).item()])

            src_c, dst_c = int(t.act_from[act]), int(t.act_to[act])
            if player == 1:
                mv = [[src_c // 10, src_c % 10], [dst_c // 10, dst_c % 10]]
            else:
                s_abs, d_abs = 99 - src_c, 99 - dst_c
                mv = [[s_abs // 10, s_abs % 10], [d_abs // 10, d_abs % 10]]

            plies.append({
                'player': player,
                'move': mv,
                'raw_acts': raw_acts,
                'safe_acts': safe_acts,
            })
            env.step(torch.tensor([act]))
        records.append({
            'p1': p1_setup, 'p2': p2_setup, 'start': start_side,
            'max_turns': 160, 'no_battle': 120, 'plies': plies,
        })

    # Verify our recorded games actually exercised filtering (at least one ply filtered moves)
    filtered_plies = sum(
        1 for g in records for p in g['plies'] if len(p['safe_acts']) < len(p['raw_acts'])
    )
    assert filtered_plies >= 5, f"Expected >=5 plies where safety filter removed moves, got {filtered_plies}"

    payload = {'unsafe_table': u_py, 'games': records}
    data_path = tmp_path / 'safety_parity.json'
    data_path.write_text(json.dumps(payload), encoding='utf-8')

    js_script = r"""
'use strict';
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const src = /<script id="engine">([\s\S]*?)<\/script>/.exec(html)[1];
const ctx = { globalThis: {} };
vm.createContext(ctx);
vm.runInContext(src, ctx);
const E = ctx.globalThis.StrategoEngine;
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

// 1. Exhaustive 12x12 isUnsafeAttack table parity
for (let a = 1; a <= 12; a++) {
  for (let d = 1; d <= 12; d++) {
    const jsVal = Boolean(E.isUnsafeAttack(a, d));
    const pyVal = Boolean(data.unsafe_table[a][d]);
    if (jsVal !== pyVal) {
      console.error(`isUnsafeAttack(${a}, ${d}) mismatch: js=${jsVal} py=${pyVal}`);
      process.exit(1);
    }
  }
}

// 2. Fallback check in JS (trapped Captain vs revealed Marshal)
{
  const g = new E.Game().reset(E.heuristicSetup(1), E.heuristicSetup(-1), 1);
  g.actual.fill(E.EMPTY); g.vis[1].fill(E.EMPTY); g.vis[-1].fill(E.EMPTY);
  g.actual[0] = E.P.CAPTAIN; g.vis[1][0] = E.P.CAPTAIN;
  g.actual[1] = E.P.BOMB;    g.vis[1][1] = E.P.BOMB;
  g.actual[10] = -E.P.MARSHAL; g.vis[1][10] = -E.P.MARSHAL;
  const raw = g.legalMoves(1);
  const safe = E.filterSafeMoves(g, 1, raw);
  if (raw.length !== 1 || safe.length !== 1) {
    console.error(`JS fallback failed: raw=${raw.length} safe=${safe.length}`);
    process.exit(1);
  }
}

// 3. Full ply-by-ply replay parity across games
let checked = 0, filtered = 0;
data.games.forEach((g, gi) => {
  const game = new E.Game({ maxTurns: g.max_turns, noBattleLimit: g.no_battle }).reset(g.p1, g.p2, g.start);
  g.plies.forEach((p, pi) => {
    const rawMoves = game.legalMoves(p.player);
    const safeMoves = E.filterSafeMoves(game, p.player, rawMoves);
    const jsRaw = rawMoves.map((m) => E.encodeMove(m, p.player)).sort((a, b) => a - b);
    const jsSafe = safeMoves.map((m) => E.encodeMove(m, p.player)).sort((a, b) => a - b);
    if (JSON.stringify(jsRaw) !== JSON.stringify(p.raw_acts)) {
      console.error(`game ${gi} ply ${pi}: raw legal mismatch`);
      process.exit(1);
    }
    if (JSON.stringify(jsSafe) !== JSON.stringify(p.safe_acts)) {
      console.error(`game ${gi} ply ${pi}: safe mask mismatch\n  js=${jsSafe}\n  py=${p.safe_acts}`);
      process.exit(1);
    }
    if (jsSafe.length < jsRaw.length) filtered++;
    game.step(p.move);
    checked++;
  });
});
console.log(`ok parity checked=${checked} filtered=${filtered}`);
"""
    js_path = tmp_path / 'check_safety.js'
    js_path.write_text(js_script, encoding='utf-8')
    res = subprocess.run([NODE, str(js_path), PLAY_HTML, str(data_path)],
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"JS/Python safety parity failed:\n{res.stdout}\n{res.stderr}"
    assert 'ok parity' in res.stdout


def test_exploit_allow_same_init_and_target_safe_smoke(tmp_path):
    """Phase B2: exploiter started from its own target with --allow-same-init and --target-safe."""
    torch.manual_seed(0)
    ckpt = tmp_path / 'target.pt'
    net = PolicyValueNet(channels=8, blocks=1)
    save(net, str(ckpt), iteration=0)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}

    out_dir = tmp_path / 'exploit_same'
    res = exploit_mod.main([
        '--target', str(ckpt),
        '--init', str(ckpt),
        '--allow-same-init',
        '--target-safe',
        '--out', str(out_dir),
        '--max-iters', '2',
        '--envs', '8',
        '--steps', '8',
        '--minibatch', '16',
        '--epochs', '1',
        '--value-warmup', '0',
        '--eval-every', '2',
        '--eval-games', '4',
        '--bank-size', '16',
        '--eval-bank-size', '8',
        '--max-plies', '12',
        '--no-battle-plies', '8',
        '--seed', '1',
        '--device', 'cpu',
    ])
    for k, ref in before.items():
        assert torch.equal(res['target_net'].state_dict()[k].cpu(), ref)
    assert any(not torch.equal(res['net'].state_dict()[k].cpu(), before[k]) for k in before)


def test_distill_smoke_cpu(tmp_path):
    """Phase B4: CPU smoke test for vec/distill.py teacher -> wider student distillation."""
    from vec import distill as distill_mod
    torch.manual_seed(0)
    teacher_path = tmp_path / 'teacher.pt'
    save(PolicyValueNet(channels=8, blocks=1), str(teacher_path), iteration=0)
    out_dir = tmp_path / 'distill_out'
    res = distill_mod.main([
        '--teacher', str(teacher_path),
        '--out', str(out_dir),
        '--channels', '16',
        '--blocks', '2',
        '--max-steps', '3',
        '--envs', '8',
        '--warmup', '1',
        '--eval-every', '3',
        '--eval-games', '4',
        '--bank-size', '16',
        '--eval-bank-size', '8',
        '--device', 'cpu',
    ])
    assert res['step'] == 3
    assert (out_dir / 'net_final.pt').exists()
    assert (out_dir / 'net_best.pt').exists()
    assert (out_dir / 'history.json').exists()
    ck = torch.load(out_dir / 'net_final.pt', map_location='cpu', weights_only=False)
    assert ck['config']['channels'] == 16 and ck['config']['blocks'] == 2
