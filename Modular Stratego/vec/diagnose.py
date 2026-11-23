"""
What a policy does in its games: its attacks by kind of target and how they end, which of its
high pieces it loses and to what, and how its games end. A batched version of the measures
Appendix C used ("share of attacks on revealed pieces lost"), with more detail.

    python -m vec.diagnose --net ../runs/ppo_1/net_final.pt --opponent expert --games 1000
"""

import argparse
import json

import torch

from vec.engine import VecStratego, BOMB, FLAG, SPY, SCOUT, MINER
from vec.expert import Style, VecExpert, VecRandom
from vec.net import NetPolicy, load
from vec.setups import diverse_setup_bank

NAMES = {1: 'Flag', 2: 'Spy', 3: 'Scout', 4: 'Miner', 5: 'Sergeant', 6: 'Lieutenant', 7: 'Captain', 8: 'Major',
         9: 'Colonel', 10: 'General', 11: 'Marshal', 12: 'Bomb'}


@torch.no_grad()
def diagnose(policy_a, policy_b, num_games: int, bank, device):
    env = VecStratego(num_games, device)
    t = env.t
    idx_all = torch.arange(num_games, device=device)
    a_side = torch.where(idx_all % 2 == 0, 1, -1)
    start = torch.randint(0, 2, (num_games,), device=device) * 2 - 1
    pick = lambda: bank[torch.randint(0, bank.size(0), (num_games,), device=bank.device)]
    env.reset(idx_all, pick(), pick(), start)
    actions = torch.zeros(num_games, dtype=torch.long, device=device)
    # A's attacks: [target kind (0 revealed, 1 hidden moved, 2 hidden unmoved), attacker rank, result (0 won, 1 lost, 2 tie)]
    att = torch.zeros(3, 13, 3, device=device)
    # A's pieces lost: [own rank, cause (0 its own attack, 1 attacked by the enemy), enemy rank]
    lost = torch.zeros(13, 2, 13, device=device)
    sure_loss = torch.zeros(2, device=device)      # [attacks on a revealed piece that beats the attacker, by non-Scouts only]
    quiet = torch.zeros(1, device=device)
    moves_a = torch.zeros(1, device=device)
    while not bool(env.done.all()):
        live = ~env.done
        is_a = live & (env.side == a_side)
        ia, ib = is_a.nonzero().flatten(), (live & ~is_a).nonzero().flatten()
        if ia.numel():
            actions[ia] = policy_a.act(env, ia)
        if ib.numel():
            actions[ib] = policy_b.act(env, ib)
        src, dst = t.act_from[actions], t.act_to[actions].clamp(max=99)
        ar = idx_all
        a_rank, d_rank = env.own[ar, src], env.opp[ar, dst]
        battle = live & (d_rank > 0)
        res = t.battle[a_rank, d_rank]                       # 1 attacker wins, -1 defender, 0 both
        rcol = torch.where(res == 1, 0, torch.where(res == -1, 1, 2))
        kind = torch.where(env.opp_rev[ar, dst], 0, torch.where(env.opp_mov[ar, dst], 1, 2))
        m = battle & is_a
        att.index_put_((kind[m], a_rank[m], rcol[m]), torch.ones(int(m.sum()), device=device), accumulate=True)
        sl = m & (kind == 0) & (res == -1)
        sure_loss[0] += sl.sum()
        sure_loss[1] += (sl & (a_rank != SCOUT)).sum()
        quiet += (is_a & ~battle).sum()
        moves_a += is_a.sum()
        # pieces A loses
        m1 = battle & is_a & (res <= 0)                      # A attacked and lost its piece
        lost.index_put_((a_rank[m1], torch.zeros_like(a_rank[m1]), d_rank[m1]), torch.ones(int(m1.sum()), device=device), accumulate=True)
        m2 = battle & ~is_a & (res >= 0)                     # the enemy attacked and took (or traded) A's piece
        lost.index_put_((d_rank[m2], torch.ones_like(d_rank[m2]), a_rank[m2]), torch.ones(int(m2.sum()), device=device), accumulate=True)
        env.step(actions)
    resA = env.winner * a_side
    out = {'games': num_games, 'wins': int((resA == 1).sum()), 'losses': int((resA == -1).sum()),
           'draws': int((resA == 0).sum()), 'plies': float(env.ply.float().mean()),
           'moves': float(moves_a), 'attacks': att.cpu(), 'lost': lost.cpu(), 'sure_loss': sure_loss.cpu().tolist()}
    return out


def report(d) -> str:
    att, lost = d['attacks'], d['lost']
    lines = [f"{d['wins']}/{d['losses']}/{d['draws']} of {d['games']}, {d['plies']:.0f} plies; "
             f"{att.sum() / d['games']:.1f} attacks per game ({100 * att.sum() / d['moves']:.0f}% of its moves)"]
    for k, name in enumerate(('revealed target', 'hidden, moved', 'hidden, unmoved')):
        w, l, tie = att[k].sum(0).tolist()
        n = max(w + l + tie, 1)
        lines.append(f"  attacks on {name:16s} {n / d['games']:5.1f} per game: won {100 * w / n:.0f}%  lost {100 * l / n:.0f}%  tied {100 * tie / n:.0f}%")
    lines.append(f"  sure losses (attacks on a revealed piece that wins): {d['sure_loss'][0] / d['games']:.2f} per game, "
                 f"{d['sure_loss'][1] / d['games']:.2f} by pieces other than Scouts")
    for rank in (11, 10, 9, 2, 4):
        tot = lost[rank].sum()
        own, taken = lost[rank, 0], lost[rank, 1]
        top = lambda v: ', '.join(f"{NAMES[int(i)]} {100 * float(v[i]) / max(float(tot), 1):.0f}%" for i in v.argsort(descending=True)[:3] if v[i] > 0)
        lines.append(f"  {NAMES[rank]:8s} lost {float(tot) / d['games']:.2f} per game: attacking into [{top(own)}]; taken by [{top(taken)}]")
    return '\n'.join(lines)


def make_policy(spec: str, device, temperature: float, safe: bool = False):
    if spec == 'expert':
        return VecExpert()
    if spec == 'random':
        return VecRandom()
    if spec.endswith('.json'):
        return VecExpert(style=Style(**json.load(open(spec))['champion']))
    return NetPolicy(load(spec, device), temperature, safe=safe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True, help='a network .pt, or expert / random / a tune_expert champion.json')
    ap.add_argument('--opponent', default='expert')
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--safe', action='store_true', help='apply play-time safety filter to --net')
    ap.add_argument('--opp-safe', action='store_true', help='apply play-time safety filter to --opponent')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(args.seed)
    bank = diverse_setup_bank(8000, seed=args.seed + 1).to(dev)
    d = diagnose(make_policy(args.net, dev, args.temperature, safe=args.safe),
                 make_policy(args.opponent, dev, args.temperature, safe=args.opp_safe),
                 args.games, bank, dev)
    tag = ' +safe' if args.safe else ''
    otag = ' +safe' if args.opp_safe else ''
    print(f"{args.net}{tag} (temperature {args.temperature}) vs {args.opponent}{otag}:")
    print(report(d))


if __name__ == '__main__':
    main()
