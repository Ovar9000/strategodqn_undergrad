"""Unit and smoke tests for vec/ppo.py (side_gae, PFSP League, masked magnet KL, cosine LR, PPO loop)."""

import json
import os

import pytest
import torch

from vec.net import PolicyValueNet, load, save
from vec.ppo import League, cosine_lr, main, masked_kl_loss, side_gae


def run(val, learner, side, done, res, v_last, last_side, gamma=1.0, lam=1.0, draw=-0.5,
        v_other=None, other_tr=None):
    f = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt).view(-1, 1)
    vo = None if v_other is None else torch.tensor([v_other], dtype=torch.float32)
    ot = None if other_tr is None else torch.tensor([other_tr], dtype=torch.float32)
    return side_gae(f(val), f(learner, torch.bool), f(side, torch.long), f(done, torch.bool), f(res),
                    torch.tensor([v_last], dtype=torch.float32), torch.tensor([last_side], dtype=torch.long),
                    gamma, lam, draw, v_other=vo, other_tr=ot).view(-1).tolist()


# ------------------------------------------------------------------ (a) side_gae
def test_self_play_win_reaches_both_sides_with_opposite_signs():
    # plies: side 0, 1, 0, 1, 0 (wins on its move); values all 0; gamma = lambda = 1
    adv = run([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], 0.0, 1)
    assert adv == [1.0, -1.0, 1.0, -1.0, 1.0]


def test_draw_costs_both_sides_and_values_are_baselines():
    adv = run([0.2, -0.1, 0.0, 0.3], [1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], 0.0, 0)
    # Monte Carlo return is -0.5 for every decision; advantage = return - value
    assert all(abs(a - b) < 1e-6 for a, b in zip(adv, [-0.7, -0.4, -0.5, -0.8]))


def test_opponent_plies_are_skipped_and_new_game_does_not_leak():
    # game 1: learner is side 0, opponent (side 1) wins on ply 1; then a new game starts.
    adv = run([0.5, 0.0, 0.1, 0.0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], 0.4, 0, lam=0.0)
    # ply 0: learner lost -> -1 - 0.5; ply 2 (new game): bootstraps from v_last for side 0 = 0.4
    assert abs(adv[0] + 1.5) < 1e-6 and adv[1] == 0.0 and abs(adv[2] - 0.3) < 1e-6 and adv[3] == 0.0


def test_truncation_draw_penalty_uses_v_other_not_negated_v_last():
    # Near-draw self-play with draw=-0.5: both sides expect ~ -0.45 to -0.48.
    # Blindly negating v_last would wrongly give side 0 a +0.45 bootstrap instead of -0.48.
    adv = run([-0.4, -0.4], [1, 1], [1, 0], [0, 0], [0, 0],
              v_last=-0.45, last_side=1, gamma=1.0, lam=0.0, draw=-0.5, v_other=-0.48, other_tr=0.0)
    assert abs(adv[0] - (-0.05)) < 1e-6   # side 1 bootstraps from v_last = -0.45
    assert abs(adv[1] - (-0.08)) < 1e-6   # side 0 bootstraps from v_other = -0.48


def test_truncation_asymmetric_opponent_bootstraps_from_v_other():
    # Learner is side 0 against a weak opponent (side 1); opponent is to move at s_T (last_side=1).
    # Learner's last decision at ply 2 must bootstrap from v_other (learner's post-opponent value),
    # completely independent of v_last (which is for the opponent side).
    adv = run([0.7, 0.0, 0.8], [1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0],
              v_last=0.95, last_side=1, gamma=1.0, lam=0.0, draw=-0.5, v_other=0.9, other_tr=0.0)
    assert abs(adv[2] - 0.1) < 1e-6       # 0.9 - 0.8, NOT -0.95 - 0.8
    assert adv[1] == 0.0
    assert abs(adv[0] - 0.1) < 1e-6       # 0.8 - 0.7


def test_truncation_lookahead_terminal_win_and_draw_for_waiting_side():
    # Batched N=2: learner is side 0, opponent is side 1 (last_side=1).
    # Env 0: opponent wins on lookahead ply T -> v_other=0, other_tr=-1.0.
    # Env 1: lookahead ply T hits a draw -> v_other=0, other_tr=-0.5.
    val = torch.tensor([[0.3, 0.2], [0.0, 0.0], [0.6, 0.4]], dtype=torch.float32)
    learner = torch.tensor([[True, True], [False, False], [True, True]])
    side = torch.tensor([[0, 0], [1, 1], [0, 0]], dtype=torch.long)
    done = torch.zeros(3, 2, dtype=torch.bool)
    res = torch.zeros(3, 2, dtype=torch.float32)
    v_last = torch.tensor([0.9, 0.9], dtype=torch.float32)
    last_side = torch.tensor([1, 1], dtype=torch.long)
    v_other = torch.tensor([0.0, 0.0], dtype=torch.float32)
    other_tr = torch.tensor([-1.0, -0.5], dtype=torch.float32)

    adv = side_gae(val, learner, side, done, res, v_last, last_side,
                   gamma=1.0, lam=1.0, draw=-0.5, v_other=v_other, other_tr=other_tr)
    # Env 0: exact MC return is -1.0 -> adv = -1.0 - val
    assert abs(float(adv[2, 0]) - (-1.6)) < 1e-6
    assert abs(float(adv[0, 0]) - (-1.3)) < 1e-6
    # Env 1: exact MC return is -0.5 -> adv = -0.5 - val
    assert abs(float(adv[2, 1]) - (-0.9)) < 1e-6
    assert abs(float(adv[0, 1]) - (-0.7)) < 1e-6
    assert float(adv[1, 0]) == 0.0 and float(adv[1, 1]) == 0.0


def test_truncation_without_v_other_never_negates_v_last():
    # When v_other is omitted, side 0's truncated final decision receives 0 advantage (never -v_last),
    # while earlier decisions of side 0 still bootstrap cleanly from val[2].
    adv = run([0.2, 0.0, 0.5], [1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0],
              v_last=-0.9, last_side=1, gamma=1.0, lam=0.5, draw=-0.5)
    assert adv[2] == 0.0
    assert adv[1] == 0.0
    assert abs(adv[0] - 0.3) < 1e-6


def test_done_on_final_ply_overrides_lookahead_values():
    # If the game ended on ply T-1, lookahead values from the reset game must not leak in.
    adv = run([0.1, 0.2], [1, 1], [1, 0], [0, 1], [0, 1],
              v_last=0.7, last_side=1, gamma=1.0, lam=1.0, draw=-0.5, v_other=0.8, other_tr=-0.9)
    assert abs(adv[1] - 0.8) < 1e-6       # +1.0 - 0.2
    assert abs(adv[0] - (-1.1)) < 1e-6    # -1.0 - 0.1


# ------------------------------------------------------------------ (b) PFSP League
def _tiny_net(tag: float) -> PolicyValueNet:
    net = PolicyValueNet(channels=8, blocks=1)
    with torch.no_grad():
        net.stem.bias.fill_(tag)
    return net


def test_pfsp_prioritizes_harder_snapshots_and_laplace_explores_new():
    league = League(max_size=4, pfsp_power=2.0, prior=1.0)
    league.add(_tiny_net(0.0))  # slot 0: easy
    league.add(_tiny_net(1.0))  # slot 1: hard
    league.add(_tiny_net(2.0))  # slot 2: brand new (0 games)
    league.record(0, wins=95, losses=5, draws=0)
    league.record(1, wins=10, losses=90, draws=0)

    probs = league.probs()
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert probs[1] > probs[2] > probs[0] > 0.0

    # Empirical sampling heavily favors the hard snapshot over the easy one
    torch.manual_seed(0)
    draws = league.sample(2000)
    assert (draws == 1).sum() > 10 * (draws == 0).sum()
    assert (draws == 2).sum() > 3 * (draws == 0).sum()

    # pfsp_power=0 reduces to uniform sampling
    league_u = League(max_size=4, pfsp_power=0.0)
    for i in range(3):
        league_u.add(_tiny_net(float(i)))
    league_u.record(0, wins=99, losses=1, draws=0)
    league_u.record(1, wins=1, losses=99, draws=0)
    assert torch.allclose(league_u.probs(), torch.full((3,), 1.0 / 3.0), atol=1e-6)


def test_pfsp_protects_anchor_and_evicts_easiest_non_anchor():
    league = League(max_size=3, pfsp_power=2.0)
    league.add(_tiny_net(0.0))  # slot 0: anchor (protected even at 100% learner win rate!)
    league.add(_tiny_net(1.0))  # slot 1: hard
    league.add(_tiny_net(2.0))  # slot 2: easy non-anchor
    league.record(0, wins=100, losses=0, draws=0)
    league.record(1, wins=20, losses=80, draws=0)
    league.record(2, wins=90, losses=10, draws=0)

    replaced = league.add(_tiny_net(3.0))
    assert replaced == 2
    assert float(league.nets[0].stem.bias[0]) == 0.0   # anchor intact
    assert float(league.nets[1].stem.bias[0]) == 1.0   # hard snapshot kept
    assert float(league.nets[2].stem.bias[0]) == 3.0   # easiest non-anchor evicted
    assert league.stats[2] == [0.0, 0.0, 0.0]          # stats reset for new snapshot


def test_pfsp_windowed_stats_adapt_to_recent_results():
    league = League(max_size=2, window=100.0)
    league.add(_tiny_net(0.0))
    league.record(0, wins=0, losses=100, draws=0)
    assert league.win_rate(0) < 0.05
    for _ in range(8):
        league.record(0, wins=50, losses=0, draws=0)
    assert league.win_rate(0) > 0.90


# ------------------------------------------------------------------ (c) Masked magnet KL
def test_masked_kl_zero_when_equal_and_ignores_illegal_logits():
    torch.manual_seed(0)
    legal = torch.tensor([[True, False, True, False, True],
                          [False, True, True, False, False]])
    base = torch.randn(2, 5)
    logits = base.clone().requires_grad_(True)
    ref_logits = base.clone()
    # Wildly perturb illegal logits in opposite directions
    with torch.no_grad():
        logits[~legal] = 1e4
        ref_logits[~legal] = -1e4

    kl = masked_kl_loss(logits, ref_logits, legal)
    assert abs(float(kl)) < 1e-6
    kl.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[~legal] == 0.0).all()
    assert logits.grad.abs().max() < 1e-6


def test_masked_kl_positive_when_shifted_and_invariant_to_illegal_perturbations():
    legal = torch.tensor([[True, False, True, True, False],
                          [True, True, False, False, False]])
    cur_a = torch.tensor([[2.0, 50.0, 0.5, -1.0, -80.0],
                          [1.0, -0.5, 99.0, -99.0, 42.0]], requires_grad=True)
    ref_a = torch.tensor([[0.2, -30.0, 1.5, 0.0, 70.0],
                          [-0.5, 1.0, -10.0, 20.0, -42.0]])

    kl_a = masked_kl_loss(cur_a, ref_a, legal)
    assert float(kl_a) > 0.1
    kl_a.backward()
    grad_a = cur_a.grad.clone()
    assert torch.isfinite(grad_a).all()
    assert (grad_a[~legal] == 0.0).all()

    # Compare against manual exact KL computed strictly over each row's legal indices
    manual = []
    for i in range(2):
        m = legal[i]
        lp = torch.log_softmax(cur_a.detach()[i, m], dim=-1)
        lq = torch.log_softmax(ref_a[i, m], dim=-1)
        manual.append((lp.exp() * (lp - lq)).sum())
    expected = torch.stack(manual).mean()
    assert abs(float(kl_a) - float(expected)) < 1e-6

    # Perturb illegal logits dramatically: both KL and legal gradients must be identical
    cur_b = cur_a.detach().clone()
    ref_b = ref_a.clone()
    cur_b[~legal] = -1e5
    ref_b[~legal] = 1e5
    cur_b.requires_grad_(True)
    kl_b = masked_kl_loss(cur_b, ref_b, legal)
    kl_b.backward()
    assert abs(float(kl_b) - float(kl_a)) < 1e-6
    assert torch.allclose(cur_b.grad, grad_a, atol=1e-6)


# ------------------------------------------------------------------ (d) Cosine LR schedule
def test_cosine_lr_schedule():
    assert abs(cosine_lr(0.0, 3e-4, 3e-5) - 3e-4) < 1e-12
    assert abs(cosine_lr(0.5, 3e-4, 3e-5) - 1.65e-4) < 1e-12
    assert abs(cosine_lr(1.0, 3e-4, 3e-5) - 3e-5) < 1e-12
    # Clamping outside [0, 1]
    assert abs(cosine_lr(-0.5, 3e-4, 3e-5) - 3e-4) < 1e-12
    assert abs(cosine_lr(1.5, 3e-4, 3e-5) - 3e-5) < 1e-12
    # Constant when lr_start == lr_end
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert abs(cosine_lr(p, 3e-4, 3e-4) - 3e-4) < 1e-12
    # Monotonically decreasing when lr_end < lr_start
    vals = [cosine_lr(i / 10.0, 3e-4, 3e-5) for i in range(11)]
    assert all(vals[i] > vals[i + 1] for i in range(10))


# ------------------------------------------------------------------ (e) End-to-end PPO smoke test
def test_ppo_smoke_end_to_end_and_resume(tmp_path, monkeypatch):
    import vec.ppo as ppo_mod

    orig_env_cls = ppo_mod.VecStratego
    monkeypatch.setattr(
        ppo_mod,
        "VecStratego",
        lambda n, dev, **kw: orig_env_cls(n, dev, max_plies=6, no_battle_plies=6),
    )

    orig_play_match = ppo_mod.play_match
    monkeypatch.setattr(
        ppo_mod,
        "play_match",
        lambda *a, **kw: orig_play_match(*a, **{**kw, "max_plies": 30, "no_battle_plies": 20}),
    )

    kl_calls = []
    orig_mkl = ppo_mod.masked_kl_loss

    def tracked_mkl(*a, **kw):
        kl_calls.append(1)
        return orig_mkl(*a, **kw)

    monkeypatch.setattr(ppo_mod, "masked_kl_loss", tracked_mkl)

    ref_path = tmp_path / "ref.pt"
    out_dir = tmp_path / "run"
    init_net = PolicyValueNet(channels=8, blocks=1)
    save(init_net, str(ref_path), iteration=0)

    common = [
        "--out", str(out_dir),
        "--envs", "16",
        "--steps", "12",
        "--minibatch", "16",
        "--epochs", "1",
        "--value-warmup", "1",
        "--lr", "3e-4",
        "--lr-end", "3e-5",
        "--p-self", "0.3",
        "--p-expert", "0.2",
        "--p-random", "0.1",
        "--p-snapshot", "0.4",
        "--pfsp-power", "2.0",
        "--snapshot-every", "1",
        "--league-size", "4",
        "--eval-every", "2",
        "--eval-games", "4",
        "--eval-ref", str(ref_path),
        "--bank-size", "16",
        "--channels", "8",
        "--blocks", "1",
        "--seed", "0",
    ]
    # Run 2 iterations with magnet KL enabled (iter 0 = value warmup, iter 1 = full PPO + magnet KL)
    main(common + ["--init", str(ref_path), "--kl-coef", "0.02", "--magnet-every", "1", "--max-iters", "2"])
    assert len(kl_calls) > 0

    assert (out_dir / "net_final.pt").exists()
    assert (out_dir / "state.pt").exists()
    assert (out_dir / "history.json").exists()
    log_text = (out_dir / "log.txt").read_text()
    assert "mkl " in log_text and "lr " in log_text
    assert "[eval] it 2 sample vs ref:" in log_text

    history = json.loads((out_dir / "history.json").read_text())
    assert len(history) == 1 and history[0]["iteration"] == 2
    assert "sample_ref" in history[0] and history[0]["sample_ref"]["games"] == 4

    ck = torch.load(out_dir / "state.pt", map_location="cpu", weights_only=False)
    assert ck["iteration"] == 2
    assert "magnet" in ck and "league_stats" in ck
    assert len(ck["league"]) == 3 and len(ck["league_stats"]) == 3
    # Games finished during rollout (max_plies=6 < steps=12), exercising record_done() & PFSP league_stats
    assert sum(sum(slot) for slot in ck["league_stats"]) > 0

    # Also test backward-compatible resume from a legacy state.pt (no 'magnet' or 'league_stats')
    # AND verify --kl-coef 0 skips the magnet forward/KL path entirely.
    legacy_ck = {k: v for k, v in ck.items() if k not in ("magnet", "league_stats")}
    torch.save(legacy_ck, out_dir / "state.pt")
    kl_calls.clear()

    main(common + ["--kl-coef", "0", "--magnet-every", "1", "--max-iters", "3"])
    assert len(kl_calls) == 0
    ck2 = torch.load(out_dir / "state.pt", map_location="cpu", weights_only=False)
    assert ck2["iteration"] == 3
    assert "magnet" in ck2 and "league_stats" in ck2
    assert sum(sum(slot) for slot in ck2["league_stats"]) > 0
    log_text2 = (out_dir / "log.txt").read_text()
    assert "resumed at iteration 2" in log_text2


def test_ppo_backward_compat_init_from_ppo2(tmp_path):
    ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "ppo_2", "net_final.pt"))
    if not os.path.exists(ckpt):
        pytest.skip("runs/ppo_2/net_final.pt not present")
    net = load(ckpt, torch.device("cpu"))
    assert net.config["in_planes"] == 59
    assert net.config["channels"] == 64 and net.config["blocks"] == 6

    out_dir = tmp_path / "ppo3_smoke"
    main([
        "--out", str(out_dir),
        "--init", ckpt,
        "--eval-ref", ckpt,
        "--max-iters", "2",
        "--envs", "8",
        "--steps", "8",
        "--minibatch", "16",
        "--epochs", "1",
        "--value-warmup", "1",
        "--kl-coef", "0.02",
        "--magnet-every", "1",
        "--eval-every", "0",
        "--snapshot-every", "0",
        "--bank-size", "16",
        "--seed", "0",
    ])
    assert (out_dir / "net_final.pt").exists()
    loaded = load(str(out_dir / "net_final.pt"), torch.device("cpu"))
    assert loaded.config == net.config
