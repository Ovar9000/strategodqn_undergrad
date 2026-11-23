"""
Distill a trained PolicyValueNet teacher (e.g. ppo_4, 6x64) into a larger PolicyValueNet
student (e.g. 10x128) on a stream of batched self-play / DAgger games.

Trains all three heads simultaneously on every ply:
  - Policy: legal-masked forward KL  KL(pi_teacher || pi_student)
  - Value:  MSE against the teacher's value output
  - Belief: cross-entropy against the true hidden enemy piece ranks (same as PPO)

Supports cosine LR annealing, periodic head-to-head evaluation against the teacher
(sampled T=1.0 and greedy T=0.0 on a separate setup bank), best-checkpoint saving,
and optional early stopping once a target head-to-head score is reached.

    python -m vec.distill --teacher ../runs/ppo_4/net_final.pt --out ../runs/distill_10x128 \\
        --channels 128 --blocks 10 --minutes 30 --envs 1024 --seed 4
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from vec.arena import deal, play_match, summary
from vec.compat import heuristic_setup_bank
from vec.engine import VecStratego
from vec.net import NetPolicy, PolicyValueNet, belief_loss, load, masked_logits, save
from vec.ppo import cosine_lr
from vec.setups import diverse_setup_bank


def legal_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     legal: torch.Tensor) -> torch.Tensor:
    """Mean KL(pi_teacher || pi_student) strictly over legal actions."""
    neg = -1e9
    s_m = student_logits.float().masked_fill(~legal, neg)
    t_m = teacher_logits.detach().float().masked_fill(~legal, neg)
    log_s = F.log_softmax(s_m, dim=-1)
    log_t = F.log_softmax(t_m, dim=-1)
    p_t = log_t.exp().masked_fill(~legal, 0.0)
    kl = (p_t * (log_t - log_s)).masked_fill(~legal, 0.0)
    return kl.sum(dim=-1).mean()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Distill a teacher PolicyValueNet into a student PolicyValueNet.')
    ap.add_argument('--teacher', required=True, help='Frozen teacher checkpoint (.pt)')
    ap.add_argument('--init', default=None, help='Optional student checkpoint to resume/fine-tune distillation from')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=10)
    ap.add_argument('--minutes', type=float, default=35.0)
    ap.add_argument('--max-steps', type=int, default=0, help='Optional step cap (0 = time-based via --minutes)')
    ap.add_argument('--envs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-end', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--value-coef', type=float, default=1.0)
    ap.add_argument('--belief-coef', type=float, default=0.5)
    ap.add_argument('--warmup', type=int, default=200, help='Steps before DAgger student driving starts')
    ap.add_argument('--p-student', type=float, default=0.40, help='Fraction of sides driven by student after warmup')
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--eval-games', type=int, default=400)
    ap.add_argument('--stop-score', type=float, default=0.0,
                    help='Optional early stop if sampled H2H score vs teacher >= this (0 = disabled)')
    ap.add_argument('--setups', default='diverse', choices=('diverse', 'heuristic'))
    ap.add_argument('--bank-size', type=int, default=20000)
    ap.add_argument('--eval-bank-size', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--device', default=None)
    return ap


def train_distill(args: argparse.Namespace) -> Dict:
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = dev.type == 'cuda'
    if use_amp:
        torch.backends.cudnn.benchmark = True

    teacher = load(args.teacher, dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    in_planes = teacher.config['in_planes']

    if getattr(args, 'init', None):
        student = load(args.init, dev)
    else:
        student = PolicyValueNet(
            channels=args.channels, blocks=args.blocks, in_planes=in_planes,
        ).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    bank_fn = diverse_setup_bank if args.setups == 'diverse' else heuristic_setup_bank
    bank = bank_fn(args.bank_size, seed=args.seed).to(dev)
    eval_bank = bank_fn(args.eval_bank_size, seed=args.seed + 10000).to(dev)

    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'student_params': n_params, 'teacher_params': t_params}, f, indent=2)

    log = open(os.path.join(args.out, 'log.txt'), 'a')
    try:
        def say(msg: str):
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()

        say(f"[distill] teacher={args.teacher} ({t_params:,} params) -> student {args.blocks}x{args.channels} "
            f"({n_params:,} params) device={dev}")
        say(f"[distill] {args}")

        N = args.envs
        env = VecStratego(N, dev)
        all_idx = torch.arange(N, device=dev)
        # side_is_student[g, s]: whether side s uses student policy after warmup (DAgger)
        side_is_student = torch.rand(N, 2, device=dev) < args.p_student
        deal(env, all_idx, bank)

        history: List[Dict] = []
        best_score = -1.0
        step = 0
        run_kl = run_v = run_b = run_agree = 0.0
        start = time.time()
        budget_sec = args.minutes * 60.0

        def run_eval(elapsed: float) -> Dict:
            nonlocal best_score
            student.eval()
            r_s = play_match(NetPolicy(student, 1.0, amp=use_amp), NetPolicy(teacher, 1.0, amp=use_amp),
                             args.eval_games, eval_bank, dev)
            r_g = play_match(NetPolicy(student, 0.0, amp=use_amp), NetPolicy(teacher, 0.0, amp=use_amp),
                             args.eval_games, eval_bank, dev)
            s_score = (r_s['wins'] + 0.5 * r_s['draws']) / max(r_s['games'], 1)
            g_score = (r_g['wins'] + 0.5 * r_g['draws']) / max(r_g['games'], 1)
            entry = {
                'step': step,
                'positions': step * N,
                'wall_sec': round(elapsed, 2),
                'sample_vs_teacher': {**r_s, 'score_rate': s_score},
                'greedy_vs_teacher': {**r_g, 'score_rate': g_score},
            }
            history.append(entry)
            say(f"[eval] step {step} ({elapsed / 60:.1f}m, {step * N / 1e6:.2f}M pos) "
                f"sample vs teacher: {summary(r_s)} score={100 * s_score:.1f}% | "
                f"greedy: {summary(r_g)} score={100 * g_score:.1f}%")
            with open(os.path.join(args.out, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            if s_score > best_score:
                best_score = s_score
                save(student, os.path.join(args.out, 'net_best.pt'), step=step, score_vs_teacher=s_score)
            save(student, os.path.join(args.out, f'net_{step}.pt'), step=step, score_vs_teacher=s_score)
            return entry

        while (time.time() - start < budget_sec) and (args.max_steps <= 0 or step < args.max_steps):
            progress = (step / args.max_steps) if args.max_steps > 0 else ((time.time() - start) / max(budget_sec, 1e-9))
            cur_lr = cosine_lr(progress, args.lr, args.lr_end)
            for g in opt.param_groups:
                g['lr'] = cur_lr

            obs, _ = env.observe()
            obs_cl = obs.contiguous(memory_format=torch.channels_last)
            legal = env.legal

            with torch.no_grad(), torch.autocast('cuda', enabled=use_amp):
                t_logits, t_val = teacher(obs_cl)

            student.train()
            with torch.autocast('cuda', enabled=use_amp):
                s_logits, s_val, s_belief = student(obs_cl, with_belief=True)

            kl_loss = legal_forward_kl(s_logits, t_logits, legal)
            v_loss = F.mse_loss(s_val.float(), t_val.float())
            b_loss, b_acc = belief_loss(s_belief, env.opp, (env.opp > 0) & ~env.opp_rev)
            loss = kl_loss + args.value_coef * v_loss + args.belief_coef * b_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                t_m = masked_logits(t_logits, legal)
                s_m = masked_logits(s_logits.detach(), legal)
                t_act = torch.multinomial(F.softmax(t_m, dim=1), 1).squeeze(1)
                if step >= args.warmup:
                    s_act = torch.multinomial(F.softmax(s_m, dim=1), 1).squeeze(1)
                    use_s = side_is_student[all_idx, (env.side < 0).long()]
                    action = torch.where(use_s, s_act, t_act)
                else:
                    action = t_act
                run_kl += float(kl_loss)
                run_v += float(v_loss)
                run_b += float(b_acc)
                run_agree += float((s_m.argmax(1) == t_m.argmax(1)).float().mean())
                done, _ = env.step(action)
                if done.any():
                    idx = done.nonzero().flatten()
                    deal(env, idx, bank)
                    side_is_student[idx] = torch.rand(idx.numel(), 2, device=dev) < args.p_student

            step += 1
            if step % 100 == 0 or step == 1:
                denom = 1 if step == 1 else 100
                elapsed = time.time() - start
                say(f"[distill] step {step} ({elapsed / 60:.1f}m) lr {cur_lr:.2e} "
                    f"kl {run_kl / denom:.4f} v_mse {run_v / denom:.4f} "
                    f"agree {run_agree / denom:.3f} belief {run_b / denom:.3f} "
                    f"({step * N / max(elapsed, 1e-6):.0f} pos/s)")
                run_kl = run_v = run_b = run_agree = 0.0

            if args.eval_every > 0 and step % args.eval_every == 0 and args.eval_games > 0:
                ev = run_eval(time.time() - start)
                if args.stop_score > 0 and ev['sample_vs_teacher']['score_rate'] >= args.stop_score:
                    say(f"[distill] reached stop_score {args.stop_score:.3f} at step {step}; stopping early.")
                    break

        elapsed = time.time() - start
        if args.eval_games > 0 and (not history or history[-1]['step'] != step):
            run_eval(elapsed)
        save(student, os.path.join(args.out, 'net_final.pt'), step=step, score_vs_teacher=best_score)
        say(f"[distill] done at step {step} ({step * N / 1e6:.2f}M pos, {elapsed / 60:.1f}m), "
            f"best sampled score vs teacher = {100 * best_score:.1f}%")
        return {'step': step, 'positions': step * N, 'best_score': best_score, 'history': history}
    finally:
        log.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    return train_distill(args)


if __name__ == '__main__':
    main()
