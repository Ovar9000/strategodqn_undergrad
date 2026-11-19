"""
PPO self-play on the batched engine, with a league of opponents.

Every game has an opponent kind: the learner itself (both sides are training data), the outlook
expert, a random player, or a frozen earlier snapshot of the learner. Only the result of the
game is rewarded (+1 win, -1 loss, DRAW for a draw, to both sides). Advantages are GAE over each
side's own sequence of decisions, so a game outcome reaches a move through real returns, not
through a chain of target networks.

    python -m vec.ppo --out ../runs/ppo_1 --init ../runs/pv_clone_1/net_final.pt --hours 2

Resumes from <out>/state.pt when it exists.
"""

import argparse
import copy
import json
import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F

from vec.arena import deal, play_match, summary
from vec.compat import heuristic_setup_bank
from vec.setups import diverse_setup_bank
from vec.engine import VecStratego, NUM_ACTIONS, NUM_OBS_PLANES
from vec.expert import Style, VecExpert, VecRandom
from vec.net import NetPolicy, PolicyValueNet, belief_loss, masked_logits, load, save

KIND_SELF, KIND_EXPERT, KIND_RANDOM, KIND_SNAPSHOT = 0, 1, 2, 3   # 3 + j = snapshot j


def quantize(obs: torch.Tensor) -> torch.Tensor:
    return (obs * 255.0).round().to(torch.uint8)


def dequantize(obs_u8: torch.Tensor) -> torch.Tensor:
    return obs_u8.float().div_(255.0)


def cosine_lr(progress: float, lr_start: float, lr_end: float) -> float:
    """Cosine annealing from lr_start (progress=0) to lr_end (progress=1), clamped to [0, 1]."""
    p = min(max(float(progress), 0.0), 1.0)
    return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * p))


def masked_kl_loss(logits: torch.Tensor, ref_logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Mean KL(pi_current || pi_ref) strictly over legal actions.

    Illegal logits are masked to -1e9 before log-softmax and illegal probability/KL terms are
    zeroed out so the result and its gradients are completely invariant to illegal logits and
    free of 0 * (-inf) NaNs.
    """
    neg = -1e9
    cur_m = logits.float().masked_fill(~legal, neg)
    ref_m = ref_logits.detach().float().masked_fill(~legal, neg)
    log_p = F.log_softmax(cur_m, dim=-1)
    log_q = F.log_softmax(ref_m, dim=-1)
    p = log_p.exp().masked_fill(~legal, 0.0)
    kl_terms = (p * (log_p - log_q)).masked_fill(~legal, 0.0)
    return kl_terms.sum(dim=-1).mean()


class League:
    """Frozen copies of the learner with Prioritized Fictitious Self-Play (PFSP)."""

    def __init__(self, max_size: int, pfsp_power: float = 2.0, prior: float = 1.0, window: float = 200.0):
        self.nets = []
        self.stats = []      # per-slot [wins, losses, draws] of the learner against this snapshot
        self.max_size = max_size
        self.pfsp_power = pfsp_power
        self.prior = prior
        self.window = window

    def win_rate(self, j: int) -> float:
        """Laplace-smoothed learner score rate (win=1, draw=0.5, loss=0) against snapshot j."""
        w, l, d = self.stats[j]
        return (w + 0.5 * d + self.prior) / (w + l + d + 2.0 * self.prior)

    def probs(self, device=None) -> torch.Tensor:
        """PFSP sampling probabilities proportional to (1 - win_rate)^pfsp_power."""
        n = len(self.nets)
        if n == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        rates = torch.tensor([self.win_rate(j) for j in range(n)], dtype=torch.float32, device=device)
        weights = (1.0 - rates).clamp_min(0.0).pow(self.pfsp_power).clamp_min(1e-6)
        return weights / weights.sum()

    def sample(self, m: int, device=None) -> torch.Tensor:
        return torch.multinomial(self.probs(device), m, replacement=True)

    def record(self, j: int, wins: float, losses: float, draws: float):
        """Update exponentially-windowed learner W/L/D stats for snapshot slot j."""
        if not (0 <= j < len(self.stats)):
            return
        w, l, d = self.stats[j]
        w += float(wins)
        l += float(losses)
        d += float(draws)
        total = w + l + d
        if self.window > 0 and total > self.window:
            scale = self.window / total
            w, l, d = w * scale, l * scale, d * scale
        self.stats[j] = [w, l, d]

    def add(self, net: PolicyValueNet, stats=None) -> int:
        snap = copy.deepcopy(net).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        st = [float(x) for x in stats] if stats is not None else [0.0, 0.0, 0.0]
        if len(self.nets) < self.max_size:
            self.nets.append(snap)
            self.stats.append(st)
            return len(self.nets) - 1
        if self.max_size <= 1:
            return -1
        # Keep anchor nets[0] protected; evict the easiest (highest learner win-rate) non-anchor snapshot
        j = max(range(1, len(self.nets)), key=lambda i: (self.win_rate(i), sum(self.stats[i])))
        self.nets[j] = snap
        self.stats[j] = st
        return j

    def __len__(self):
        return len(self.nets)


def side_gae(val, learner, side, done, res, v_last, last_side, gamma, lam, draw,
             v_other: Optional[torch.Tensor] = None, other_tr: Optional[torch.Tensor] = None):
    """GAE over each side's own sequence of decisions.

    All of shape (T, N): val the learner's value at a ply (mover's view), learner whether the
    learner chose the move, side 0/1 the mover, done whether the game ended on this ply, res the
    result for the mover (+1 won, 0 draw; a mover never loses on its own move).

    v_last (N,) is the value of the position after ply T-1 for the side to move there,
    last_side (N,). For the waiting side (1 - last_side), pass v_other (N,) (its value after
    one lookahead ply by last_side, 0 where that ply ended the game) and other_tr (N,) (its
    terminal result if that lookahead ply ended the game, 0 otherwise). When neither v_other
    nor other_tr is given, the waiting side has no bootstrap at truncation and its final
    truncated decision receives zero advantage (never negating v_last, which is biased under
    non-zero draw penalties or asymmetric opponents).
    """
    T, N = val.shape
    dev = val.device
    ar = torch.arange(N, device=dev)
    adv = torch.zeros_like(val)
    nv = torch.zeros(N, 2, device=dev, dtype=val.dtype)          # value at the side's next decision
    na = torch.zeros(N, 2, device=dev, dtype=val.dtype)          # advantage at the side's next decision
    tr = torch.zeros(N, 2, device=dev, dtype=val.dtype)          # result waiting for the side's last decision
    has_boot = torch.zeros(N, 2, device=dev, dtype=torch.bool)   # whether a valid bootstrap/terminal exists
    nv[ar, last_side] = v_last.to(device=dev, dtype=val.dtype)
    has_boot[ar, last_side] = True
    if v_other is not None:
        nv[ar, 1 - last_side] = v_other.to(device=dev, dtype=val.dtype)
        has_boot[ar, 1 - last_side] = True
    if other_tr is not None:
        tr[ar, 1 - last_side] = other_tr.to(device=dev, dtype=val.dtype)
        has_boot[ar, 1 - last_side] = True
    for t in reversed(range(T)):
        s, d, z = side[t], done[t], res[t]
        if d.any():
            is_draw = z == 0
            zm = torch.where(is_draw, torch.full_like(z, draw), z)
            zo = torch.where(is_draw, torch.full_like(z, draw), -z)
            nv[d] = 0.0
            na[d] = 0.0
            has_boot[d] = True
            tr[ar, s] = torch.where(d, zm, tr[ar, s])
            tr[ar, 1 - s] = torch.where(d, zo, tr[ar, 1 - s])
        L = learner[t]
        valid = L & has_boot[ar, s]
        delta = tr[ar, s] + gamma * nv[ar, s] - val[t]
        a_t = delta + gamma * lam * na[ar, s]
        adv[t] = torch.where(valid, a_t, torch.zeros_like(a_t))
        nv[ar, s] = torch.where(L, val[t], nv[ar, s])
        na[ar, s] = torch.where(valid, a_t, torch.where(L, torch.zeros_like(a_t), na[ar, s]))
        tr[ar, s] = torch.where(L, torch.zeros_like(a_t), tr[ar, s])
        has_boot[ar, s] = torch.where(L, torch.ones_like(L), has_boot[ar, s])
    return adv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default=None)
    ap.add_argument('--hours', type=float, default=1.0)
    ap.add_argument('--max-iters', type=int, default=0, help='optional iteration cap (0 = time-based via --hours)')
    ap.add_argument('--envs', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-end', type=float, default=None,
                    help='final learning rate for cosine annealing (default: 0.1 * --lr, i.e. 3e-5 at lr=3e-4)')
    ap.add_argument('--gamma', type=float, default=0.998)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--minibatch', type=int, default=4096)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--kl-coef', type=float, default=0.02,
                    help='R-NaD / magnetic mirror descent KL regularization weight (0 disables magnet forward pass)')
    ap.add_argument('--magnet-every', type=int, default=40,
                    help='iterations between updating the frozen magnet network (0 = never update)')
    ap.add_argument('--value-coef', type=float, default=0.5)
    ap.add_argument('--draw', type=float, default=-0.5)
    ap.add_argument('--belief-coef', type=float, default=0.25)
    ap.add_argument('--value-warmup', type=int, default=4, help='iterations that train the value head only')
    ap.add_argument('--p-self', type=float, default=0.5)
    ap.add_argument('--p-expert', type=float, default=0.25)
    ap.add_argument('--p-random', type=float, default=0.05)
    ap.add_argument('--p-snapshot', type=float, default=0.20)
    ap.add_argument('--snapshot-every', type=int, default=25)
    ap.add_argument('--league-size', type=int, default=8)
    ap.add_argument('--pfsp-power', type=float, default=2.0,
                    help='PFSP exponent p in (1 - win_rate)^p (0 = uniform)')
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--eval-games', type=int, default=400)
    ap.add_argument('--eval-ref', default=None,
                    help='optional fixed reference checkpoint (.pt) to play eval-games against (sampled T=1.0)')
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--expert-styles', default=None, help='JSON file: a list of Style parameter dicts; '
                    'games against the expert pick the original or one of these at random')
    ap.add_argument('--setups', default='diverse', choices=('diverse', 'heuristic'))
    ap.add_argument('--bank-size', type=int, default=20000)
    ap.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available else cpu)')
    args = ap.parse_args(argv)
    if args.lr_end is None:
        args.lr_end = args.lr * 0.1
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = dev.type == 'cuda'
    if use_amp:
        torch.backends.cudnn.benchmark = True
    log = open(os.path.join(args.out, 'log.txt'), 'a')
    try:
        def say(msg):
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()

        bank = (diverse_setup_bank if args.setups == 'diverse' else heuristic_setup_bank)(
            args.bank_size, seed=args.seed).to(dev)
        state_path = os.path.join(args.out, 'state.pt')
        league = League(args.league_size, pfsp_power=args.pfsp_power)
        history, iteration, total_samples = [], 0, 0
        if os.path.exists(state_path):
            ck = torch.load(state_path, map_location=dev, weights_only=False)
            net = PolicyValueNet(**ck['config']).to(dev).to(memory_format=torch.channels_last)
            net.load_state_dict(ck['state_dict'])
            opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
            opt.load_state_dict(ck['opt'])
            saved_stats = ck.get('league_stats')
            for i, sd in enumerate(ck['league']):
                snap = PolicyValueNet(**ck['config']).to(dev).to(memory_format=torch.channels_last)
                snap.load_state_dict(sd)
                st = saved_stats[i] if saved_stats is not None and i < len(saved_stats) else None
                league.add(snap, stats=st)
            if 'magnet' in ck:
                magnet = PolicyValueNet(**ck['config']).to(dev).to(memory_format=torch.channels_last)
                magnet.load_state_dict(ck['magnet'])
                magnet.eval()
            else:
                magnet = copy.deepcopy(net).eval()
            history, iteration, total_samples = ck['history'], ck['iteration'], ck['total_samples']
            say(f"[ppo] resumed at iteration {iteration} ({total_samples} samples, league {len(league)})")
        else:
            net = load(args.init, dev) if args.init else \
                PolicyValueNet(args.channels, args.blocks).to(dev).to(memory_format=torch.channels_last)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
            league.add(net)
            magnet = copy.deepcopy(net).eval()
        for p in magnet.parameters():
            p.requires_grad_(False)
        for g in opt.param_groups:
            g['lr'] = args.lr
        reference = copy.deepcopy(league.nets[0]).eval()     # the starting point, a fixed yardstick
        eval_ref = None
        if args.eval_ref:
            eval_ref = load(args.eval_ref, dev).eval()
            for p in eval_ref.parameters():
                p.requires_grad_(False)
        # The value head alone, for the warm-up: the cloned network's value head is untrained, and
        # training it through the shared trunk moved the policy (KL 0.26 in one iteration, and the
        # greedy win rate against random fell from 94% to 60%).
        opt_value = torch.optim.Adam(list(net.value_conv.parameters()) + list(net.value_fc.parameters()), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        say(f"[ppo] {args}")

        N, T = args.envs, args.steps
        env = VecStratego(N, dev)
        ar = torch.arange(N, device=dev)
        kind = torch.zeros(N, dtype=torch.long, device=dev)
        learner_side = torch.ones(N, dtype=torch.long, device=dev)

        if args.expert_styles:
            with open(args.expert_styles) as f:
                styles = json.load(f)
        else:
            styles = []
        experts = [VecExpert()] + [VecExpert(style=Style(**d), name=f'expert{i + 2}') for i, d in enumerate(styles)]
        variant = torch.zeros(N, dtype=torch.long, device=dev)   # which expert a game against the expert uses

        def assign(idx):
            m = idx.numel()
            p = torch.tensor([args.p_self, args.p_expert, args.p_random, args.p_snapshot], device=dev)
            k = torch.multinomial(p, m, replacement=True)
            snap = league.sample(m, device=dev)
            kind[idx] = torch.where(k == 3, KIND_SNAPSHOT + snap, k)
            learner_side[idx] = torch.randint(0, 2, (m,), device=dev) * 2 - 1
            variant[idx] = torch.randint(0, len(experts), (m,), device=dev)
            deal(env, idx, bank)

        def select_opponent_actions(obs_u8, other, actions):
            for v, ex in enumerate(experts):
                ie = (other & (kind == KIND_EXPERT) & (variant == v)).nonzero().flatten()
                if ie.numel():
                    actions[ie] = ex.act(env, ie)
            ir = (other & (kind == KIND_RANDOM)).nonzero().flatten()
            if ir.numel():
                actions[ir] = torch.multinomial(env.legal[ir].float(), 1).squeeze(1)
            for j, snap in enumerate(league.nets):
                ij = (other & (kind == KIND_SNAPSHOT + j)).nonzero().flatten()
                if ij.numel():
                    with torch.autocast('cuda', enabled=use_amp):
                        lg, _ = snap(dequantize(obs_u8[ij]).contiguous(memory_format=torch.channels_last))
                    actions[ij] = torch.multinomial(torch.softmax(masked_logits(lg, env.legal[ij]), 1), 1).squeeze(1)

        def save_state():
            torch.save({'config': net.config, 'state_dict': net.state_dict(), 'opt': opt.state_dict(),
                        'league': [s.state_dict() for s in league.nets], 'league_stats': league.stats,
                        'magnet': magnet.state_dict(), 'history': history,
                        'iteration': iteration, 'total_samples': total_samples}, state_path)

        assign(ar)
        obs_buf = torch.zeros(T, N, NUM_OBS_PLANES, 10, 10, dtype=torch.uint8, device=dev)
        hid_buf = torch.zeros(T, N, 100, dtype=torch.uint8, device=dev)   # true rank of each hidden enemy piece
        legal_buf = torch.zeros(T, N, NUM_ACTIONS, dtype=torch.bool, device=dev)
        act_buf = torch.zeros(T, N, dtype=torch.long, device=dev)
        logp_buf = torch.zeros(T, N, device=dev)
        val_buf = torch.zeros(T, N, device=dev)
        learner_buf = torch.zeros(T, N, dtype=torch.bool, device=dev)
        side_buf = torch.zeros(T, N, dtype=torch.long, device=dev)
        done_buf = torch.zeros(T, N, dtype=torch.bool, device=dev)
        res_buf = torch.zeros(T, N, device=dev)
        # running results of the learner by opponent kind: wins, losses, draws
        tally = torch.zeros(4, 3, device=dev)
        ends = torch.zeros(5, device=dev)
        plies_done, games_done = 0.0, 0

        def record_done(done, res, mover_side):
            nonlocal plies_done, games_done
            if not done.any():
                return
            idx = done.nonzero().flatten()
            k = kind[idx].clamp(max=KIND_SNAPSHOT)
            # the learner's result (self-play games count for the side that moved last)
            lres = torch.where(kind[idx] == KIND_SELF, res[idx],
                               res[idx] * torch.where(mover_side[idx] == learner_side[idx], 1, -1))
            col = torch.where(lres > 0, 0, torch.where(lres < 0, 1, 2))
            tally.index_put_((k, col), torch.ones(idx.numel(), device=dev), accumulate=True)
            is_snap = kind[idx] >= KIND_SNAPSHOT
            if is_snap.any():
                sj = (kind[idx[is_snap]] - KIND_SNAPSHOT).cpu().tolist()
                sc = col[is_snap].cpu().tolist()
                counts = {}
                for j, c in zip(sj, sc):
                    row = counts.setdefault(j, [0, 0, 0])
                    row[c] += 1
                for j, (w, l, d) in counts.items():
                    league.record(j, w, l, d)
            ends.add_(torch.bincount(env.end_type[idx], minlength=5).float())
            plies_done += float(env.ply[idx].sum())
            games_done += idx.numel()
            assign(idx)

        start = time.time()
        while (time.time() - start < args.hours * 3600) and (args.max_iters <= 0 or iteration < args.max_iters):
            t_roll = time.time()
            net.eval()
            with torch.no_grad():
                for t in range(T):
                    obs, dist = env.observe()
                    obs_u8 = quantize(obs)
                    is_learner = (kind == KIND_SELF) | (env.side == learner_side)
                    actions = torch.zeros(N, dtype=torch.long, device=dev)
                    logp_buf[t].zero_()
                    val_buf[t].zero_()
                    il = is_learner.nonzero().flatten()
                    if il.numel():
                        with torch.autocast('cuda', enabled=use_amp):
                            logits, value = net(dequantize(obs_u8[il]).contiguous(memory_format=torch.channels_last))
                        logits = masked_logits(logits, env.legal[il])
                        logp_all = F.log_softmax(logits, 1)
                        a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                        actions[il] = a
                        logp_buf[t, il] = logp_all.gather(1, a[:, None]).squeeze(1)
                        val_buf[t, il] = value.float()
                    select_opponent_actions(obs_u8, ~is_learner, actions)
                    obs_buf[t], legal_buf[t], act_buf[t] = obs_u8, env.legal, actions
                    hid_buf[t] = torch.where(env.opp_rev, torch.zeros_like(env.opp), env.opp).to(torch.uint8)
                    learner_buf[t], side_buf[t] = is_learner, (env.side < 0).long()
                    mover_side = env.side.clone()
                    done, res = env.step(actions)
                    done_buf[t], res_buf[t] = done, res.float()
                    record_done(done, res, mover_side)

                # Unbiased end-of-rollout bootstrap for BOTH sides via one lookahead ply:
                # - last_side (moving at s_T) gets V(s_T) from its own canonical view.
                # - 1 - last_side (waiting at s_T) observes ply T's outcome: either its exact
                #   terminal result if ply T ends the game, or V(s_{T+1}) from its own view.
                obs_T, _ = env.observe()
                obs_T_u8 = quantize(obs_T)
                last_side = (env.side < 0).long()
                is_learner_T = (kind == KIND_SELF) | (env.side == learner_side)
                actions_T = torch.zeros(N, dtype=torch.long, device=dev)
                v_last = torch.zeros(N, device=dev)
                il = is_learner_T.nonzero().flatten()
                if il.numel():
                    with torch.autocast('cuda', enabled=use_amp):
                        lg_T, val_T = net(dequantize(obs_T_u8[il]).contiguous(memory_format=torch.channels_last))
                    lg_T = masked_logits(lg_T, env.legal[il])
                    actions_T[il] = torch.multinomial(F.softmax(lg_T, 1), 1).squeeze(1)
                    v_last[il] = val_T.float()
                select_opponent_actions(obs_T_u8, ~is_learner_T, actions_T)
                mover_side_T = env.side.clone()
                done_T, res_T = env.step(actions_T)
                record_done(done_T, res_T, mover_side_T)

                v_other = torch.zeros(N, device=dev)
                other_tr = torch.zeros(N, device=dev)
                if done_T.any():
                    res_f = res_T.float()
                    zo_T = torch.where(res_T == 0, torch.full_like(res_f, args.draw), -res_f)
                    other_tr = torch.where(done_T, zo_T, other_tr)
                io = (~done_T & ((kind == KIND_SELF) | (env.side == learner_side))).nonzero().flatten()
                if io.numel():
                    obs_T1, _ = env.observe(io)
                    with torch.autocast('cuda', enabled=use_amp):
                        _, val_T1 = net(dequantize(quantize(obs_T1)).contiguous(memory_format=torch.channels_last))
                    v_other[io] = val_T1.float()

                adv = side_gae(val_buf, learner_buf, side_buf, done_buf, res_buf, v_last, last_side,
                               args.gamma, args.lam, args.draw, v_other=v_other, other_tr=other_tr)
                ret = (adv + val_buf).clamp(-1.0, 1.0)
            roll_time = time.time() - t_roll

            # ---------------------------------------------------------------- update
            t_train = time.time()
            if args.max_iters > 0:
                progress = iteration / args.max_iters
            else:
                progress = (time.time() - start) / max(args.hours * 3600.0, 1e-9)
            cur_lr = cosine_lr(progress, args.lr, args.lr_end)
            for g in opt.param_groups:
                g['lr'] = cur_lr

            sel = learner_buf.view(-1).nonzero().flatten()
            M = sel.numel()
            f_obs, f_legal = obs_buf.view(T * N, NUM_OBS_PLANES, 10, 10), legal_buf.view(T * N, NUM_ACTIONS)
            f_hid = hid_buf.view(T * N, 100)
            f_act, f_logp, f_adv, f_ret, f_val = (x.view(-1) for x in (act_buf, logp_buf, adv, ret, val_buf))
            value_only = iteration < args.value_warmup
            stats = torch.zeros(7, device=dev)
            n_mb = 0
            if M > 0:
                mb_size = min(args.minibatch, M)
                net.train()
                for epoch in range(args.epochs):
                    perm = sel[torch.randperm(M, device=dev)]
                    for i in range(0, M - mb_size + 1, mb_size):
                        b = perm[i:i + mb_size]
                        obs_in = dequantize(f_obs[b]).contiguous(memory_format=torch.channels_last)
                        with torch.autocast('cuda', enabled=use_amp):
                            raw_logits, value, belief = net(obs_in, with_belief=True)
                        hid = f_hid[b].long()
                        b_loss, b_acc = belief_loss(belief, hid, hid > 0)
                        logits = masked_logits(raw_logits, f_legal[b])
                        logp_all = F.log_softmax(logits, 1)
                        logp = logp_all.gather(1, f_act[b][:, None]).squeeze(1)
                        entropy = -(logp_all.exp() * logp_all.masked_fill(~f_legal[b], 0.0)).sum(1).mean()
                        a_b = f_adv[b]
                        a_b = (a_b - a_b.mean()) / (a_b.std(unbiased=False) + 1e-8)
                        ratio = (logp - f_logp[b]).exp()
                        pg = -torch.min(ratio * a_b, ratio.clamp(1 - args.clip, 1 + args.clip) * a_b).mean()
                        v_loss = F.mse_loss(value.float(), f_ret[b])
                        if not value_only and args.kl_coef > 0:
                            with torch.no_grad(), torch.autocast('cuda', enabled=use_amp):
                                m_logits, _ = magnet(obs_in)
                            mkl = masked_kl_loss(raw_logits, m_logits, f_legal[b])
                        else:
                            mkl = torch.zeros((), device=dev)
                        if value_only:
                            loss = args.value_coef * v_loss
                        else:
                            loss = pg + args.value_coef * v_loss - args.entropy * entropy + args.belief_coef * b_loss
                            if args.kl_coef > 0:
                                loss = loss + args.kl_coef * mkl
                        o = opt_value if value_only else opt
                        net.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(o)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        scaler.step(o)
                        scaler.update()
                        with torch.no_grad():
                            kl = (f_logp[b] - logp).mean()
                            clipped = ((ratio - 1).abs() > args.clip).float().mean()
                            stats += torch.stack([pg.detach(), v_loss.detach(), entropy.detach(),
                                                  kl, clipped, b_acc, mkl.detach()])
                        n_mb += 1
            stats /= max(n_mb, 1)
            ev = 1 - (f_ret[sel] - f_val[sel]).var() / f_ret[sel].var().clamp_min(1e-8) if M > 1 else torch.zeros(())
            iteration += 1
            total_samples += M
            train_time = time.time() - t_train
            if iteration % 5 == 0 or iteration == 1:
                tl = tally.cpu()
                rate = lambda r: f"{int(r[0])}/{int(r[1])}/{int(r[2])}"
                say(f"[ppo] it {iteration} samples {total_samples / 1e6:.2f}M roll {roll_time:.0f}s train {train_time:.0f}s "
                    f"lr {cur_lr:.2e} pg {stats[0]:.4f} v {stats[1]:.4f} ent {stats[2]:.3f} kl {stats[3]:.4f} "
                    f"mkl {stats[6]:.4f} clip {stats[4]:.2f} ev {float(ev):.2f} belief {stats[5]:.3f} | "
                    f"self {rate(tl[0])} expert {rate(tl[1])} random {rate(tl[2])} snap {rate(tl[3])} | "
                    f"len {plies_done / max(games_done, 1):.0f} ends flag {int(ends[1])} nomove {int(ends[2])} "
                    f"nobattle {int(ends[3])} limit {int(ends[4])}")
                tally.zero_()
                ends.zero_()
                plies_done, games_done = 0.0, 0
            if args.snapshot_every > 0 and iteration % args.snapshot_every == 0:
                league.add(net)
            if args.magnet_every > 0 and iteration % args.magnet_every == 0:
                magnet.load_state_dict(net.state_dict())
            if args.eval_every > 0 and iteration % args.eval_every == 0:
                entry = {'iteration': iteration, 'samples': total_samples, 'hours': (time.time() - start) / 3600}
                opponents = [('random', VecRandom()), ('expert', experts[0])] + [(ex.name, ex) for ex in experts[1:]] \
                    + [('start', NetPolicy(reference, 1.0))]
                if eval_ref is not None:
                    opponents.append(('ref', NetPolicy(eval_ref, 1.0)))
                for name, opp in opponents:
                    for mode, temp in (('greedy', 0.0), ('sample', 1.0)):
                        if (mode == 'sample' and name == 'random') or (mode == 'greedy' and name not in ('random', 'expert')):
                            continue
                        r = play_match(NetPolicy(net, temp), opp, args.eval_games, bank, dev)
                        entry[f'{mode}_{name}'] = r
                        say(f"[eval] it {iteration} {mode} vs {name}: {summary(r)}")
                history.append(entry)
                with open(os.path.join(args.out, 'history.json'), 'w') as f:
                    json.dump(history, f, indent=1)
                save(net, os.path.join(args.out, f'net_{iteration}.pt'), iteration=iteration)
                save_state()
        save(net, os.path.join(args.out, 'net_final.pt'), iteration=iteration)
        save_state()
        say(f"[ppo] stopped at iteration {iteration}")
    finally:
        log.close()


if __name__ == '__main__':
    main()
