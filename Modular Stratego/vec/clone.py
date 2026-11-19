"""
Clone the outlook expert into the policy-value network on a stream of batched games.

Every ply of every game is a training position: the expert scores all legal moves and the
network is trained towards softmax(score / TARGET_TEMP). The games are driven by a mix of
players (the expert, the expert with random moves, a random player and, after a warm-up, the
network itself), so the network is also labelled on positions it gets into by its own mistakes
(DAgger) and on positions against an opponent that attacks, which two experts never reach.

    python -m vec.clone --out ../runs/pv_clone_1 --minutes 30
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from vec.arena import deal, play_match, summary
from vec.compat import heuristic_setup_bank
from vec.setups import diverse_setup_bank
from vec.engine import VecStratego
from vec.expert import VecExpert, VecRandom, expert_scores
from vec.net import NetPolicy, PolicyValueNet, belief_loss, masked_logits, save

TARGET_TEMP = 0.03
# who drives a side of a game: (name, probability)
DRIVERS = (('expert', 0.30), ('noisy', 0.20), ('random', 0.10), ('student', 0.40))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--minutes', type=float, default=30)
    ap.add_argument('--envs', type=int, default=2048)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=1500)
    ap.add_argument('--eval-games', type=int, default=400)
    ap.add_argument('--belief-coef', type=float, default=0.5)
    ap.add_argument('--warmup', type=int, default=300, help='updates before the student drives games')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--setups', default='diverse', choices=('diverse', 'heuristic'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    bank = (diverse_setup_bank if args.setups == 'diverse' else heuristic_setup_bank)(20000, seed=args.seed).to(dev)
    net = PolicyValueNet(args.channels, args.blocks).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    env = VecStratego(args.envs, dev)
    N = args.envs
    all_idx = torch.arange(N, device=dev)
    probs = torch.tensor([p for _, p in DRIVERS], device=dev)
    # driver[g, s]: who plays side s (0: Player 1, 1: Player 2) of game g
    driver = torch.multinomial(probs, 2 * N, replacement=True).view(N, 2)
    deal(env, all_idx, bank)
    log = open(os.path.join(args.out, 'log.txt'), 'a')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say(f"[clone] {args}")
    start = time.time()
    step = 0
    run_loss = run_agree = run_belief = 0.0
    history = []
    while time.time() - start < args.minutes * 60:
        obs, dist = env.observe()
        scores = expert_scores(env, dist=dist)
        legal = env.legal
        target = torch.softmax(scores / TARGET_TEMP, 1)
        net.train()
        with torch.autocast('cuda'):
            logits, _, belief = net(obs.contiguous(memory_format=torch.channels_last), with_belief=True)
        logits = masked_logits(logits, legal)
        logp = F.log_softmax(logits, 1)
        bc = -(target * logp.masked_fill(~legal, 0.0)).sum(1).mean()
        b_loss, b_acc = belief_loss(belief, env.opp, (env.opp > 0) & ~env.opp_rev)
        loss = bc + args.belief_coef * b_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            expert_move = (scores + 0.02 * torch.rand_like(scores)).argmax(1)
            student_move = torch.multinomial(torch.softmax(logits.detach(), 1), 1).squeeze(1)
            random_move = torch.multinomial(legal.float(), 1).squeeze(1)
            who = driver[all_idx, (env.side < 0).long()]
            noisy_move = torch.where(torch.rand(N, device=dev) < 0.25, random_move, expert_move)
            action = expert_move
            action = torch.where(who == 1, noisy_move, action)
            action = torch.where(who == 2, random_move, action)
            if step >= args.warmup:
                action = torch.where(who == 3, student_move, action)
            run_loss += float(bc)
            run_belief += float(b_acc)
            run_agree += float((logits.argmax(1) == scores.argmax(1)).float().mean())
            done, _ = env.step(action)
            if done.any():
                idx = done.nonzero().flatten()
                deal(env, idx, bank)
                driver[idx] = torch.multinomial(probs, 2 * idx.numel(), replacement=True).view(-1, 2)
        step += 1
        if step % 100 == 0:
            say(f"[clone] step {step} {time.time() - start:.0f}s loss {run_loss / 100:.4f} agree {run_agree / 100:.3f} belief acc {run_belief / 100:.3f} "
                f"({step * N / (time.time() - start):.0f} positions/s)")
            run_loss = run_agree = run_belief = 0.0
        if step % args.eval_every == 0:
            evaluate(net, bank, dev, args.eval_games, say, history, step)
            save(net, os.path.join(args.out, f'net_{step}.pt'), step=step)
    evaluate(net, bank, dev, args.eval_games, say, history, step)
    save(net, os.path.join(args.out, 'net_final.pt'), step=step)
    json.dump(history, open(os.path.join(args.out, 'history.json'), 'w'), indent=1)


def evaluate(net, bank, dev, games, say, history, step):
    entry = {'step': step}
    for name, opp in (('random', VecRandom()), ('expert', VecExpert())):
        r = play_match(NetPolicy(net), opp, games, bank, dev)
        entry[name] = r
        say(f"[eval] step {step} greedy vs {name}: {summary(r)}")
    history.append(entry)


if __name__ == '__main__':
    main()
