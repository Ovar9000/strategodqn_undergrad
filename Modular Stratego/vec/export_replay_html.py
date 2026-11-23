"""
Extract a standalone single-file HTML visualizer for 1 Stratego game.

Embeds a schema-v1 replay JSON directly inside a self-contained copy of `web/replay.html`
so the resulting `.html` file opens with a double-click (no local HTTP server needed) with
the full interactive board, fog-of-war toggles, belief overlays, top-5 move probabilities,
value curve, battle/blunder jumps, and playback controls.

Usage (from Modular Stratego/):
    # Wrap an existing replay JSON
    python -m vec.export_replay_html --json ../runs/replays/some_game.json --out ../runs/replays/game.html

    # Record a fresh showcase game and extract its standalone HTML directly
    python -m vec.export_replay_html \\
        --net ../runs/ppo_5_10x128/net_final.pt --safe \\
        --opponent ../runs/ppo_4/net_final.pt \\
        --out ../runs/replays/ppo5_10x128_vs_ppo4_showcase.html \\
        --min-plies 140 --max-target-plies 240 --seed 2026
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
from typing import Any, Dict, Optional

import torch

from vec.engine import VecStratego
from vec.net import NetPolicy, load as load_net, masked_logits
from vec import record_games
from vec.safety import safe_mask_for_env
from vec.setups import diverse_setup_bank

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_TEMPLATE = os.path.join(ROOT, 'web', 'replay.html')


def embed_replay_html(
    replay: Dict[str, Any],
    out_html: str,
    title: Optional[str] = None,
    data_name: Optional[str] = None,
    template_path: str = REPLAY_TEMPLATE,
) -> str:
    """Write a standalone single-file HTML replay containing `replay`."""
    errs = record_games.verify_replay(replay)
    if errs:
        raise ValueError(f"Replay failed verification before HTML export: {errs[:3]}")
    with open(template_path, encoding='utf-8') as f:
        tpl = f.read()
    if '</body>' not in tpl:
        raise ValueError(f"Template {template_path!r} is missing </body>")

    m = replay.get('meta', {})
    disp_title = title or f"{m.get('red', 'Red')} vs {m.get('blue', 'Blue')} — Stratego Replay"
    fname = data_name or os.path.splitext(os.path.basename(out_html))[0] + '.json'

    # Escape </ so JSON can never prematurely close the <script> tag
    safe_json = json.dumps(replay, separators=(',', ':')).replace('</', '<\\/')
    tag = (
        f'<script id="embedded-replay" type="application/json" '
        f'data-name="{html_lib.escape(fname, quote=True)}">{safe_json}</script>\n'
    )
    out = tpl.replace(
        '<title>MARQ Stratego — Replay &amp; Blunder Inspector</title>',
        f'<title>{html_lib.escape(disp_title)}</title>',
        1,
    )
    anchor = '<script id="replay-core">'
    if anchor in out:
        out = out.replace(anchor, tag + anchor, 1)
    else:
        out = out.replace('</body>', tag + '</body>', 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(out)
    return out_html


@torch.no_grad()
def record_showcase_game(
    net_path: str,
    opponent: str,
    device: torch.device,
    safe: bool = True,
    opp_safe: bool = False,
    temperature: float = 1.0,
    seed: int = 2026,
    pool_size: int = 64,
    min_plies: int = 120,
    max_target_plies: int = 260,
    model_label: Optional[str] = None,
    opp_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Search a batch of games to pick a clean, decisive Flag-capture win by `net` (as Red)
    within `[min_plies, max_target_plies]`, then record its full per-ply policy/value/belief."""
    net = load_net(net_path, device).eval()
    if os.path.exists(opponent):
        opp_net = load_net(opponent, device).eval()
        opp_pol = NetPolicy(opp_net, temperature=temperature, safe=opp_safe)
        o_name = opp_label or os.path.basename(os.path.dirname(os.path.abspath(opponent)))
    else:
        opp_pol, o_name = record_games.make_opponent(opponent, device, temperature=temperature)
        if opp_label:
            o_name = opp_label

    m_label = model_label or (
        os.path.basename(os.path.dirname(os.path.abspath(net_path))) + ('+safe' if safe else '')
    )

    bank = diverse_setup_bank(max(512, pool_size * 4), seed=seed + 1)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    a_side = torch.ones(pool_size, dtype=torch.long)  # Red (+1, bottom of board)
    start_side = torch.randint(0, 2, (pool_size,), generator=gen) * 2 - 1
    mover_setup = bank[torch.randint(0, bank.size(0), (pool_size,), generator=gen)]
    other_setup = bank[torch.randint(0, bank.size(0), (pool_size,), generator=gen)]

    env = VecStratego(pool_size, device)
    idx_all = torch.arange(pool_size, device=device)
    a_side_dev = a_side.to(device)
    env.reset(idx_all, mover_setup.to(device), other_setup.to(device), start_side.to(device))
    use_amp = device.type == 'cuda'

    actions_seq = []
    battles = torch.zeros(pool_size, dtype=torch.long, device=device)
    torch.manual_seed(seed + 77)
    while not bool(env.done.all()):
        live = ~env.done
        is_net = live & (env.side == a_side_dev)
        in_idx = is_net.nonzero().flatten()
        io_idx = (live & ~is_net).nonzero().flatten()
        acts = torch.zeros(pool_size, dtype=torch.long, device=device)
        if in_idx.numel():
            obs, _ = env.observe(in_idx)
            legal = safe_mask_for_env(env, in_idx) if safe else env.legal[in_idx]
            with torch.autocast('cuda', enabled=use_amp):
                logits, _ = net(obs.contiguous(memory_format=torch.channels_last))
            ml = masked_logits(logits, legal)
            if temperature > 0:
                acts[in_idx] = torch.multinomial(torch.softmax(ml / temperature, dim=1), 1).squeeze(1)
            else:
                acts[in_idx] = ml.argmax(1)
        if io_idx.numel():
            acts[io_idx] = opp_pol.act(env, io_idx)
        dst = env.t.act_to[acts].clamp(max=99)
        battles += (live & (env.opp[idx_all, dst] > 0)).long()
        actions_seq.append(acts.cpu())
        env.step(acts)

    # Pick the best Red Flag-capture win closest to the target mid-length range with plenty of action
    winner = env.winner.cpu()
    end_type = env.end_type.cpu()
    plies = env.ply.cpu()
    bat_cpu = battles.cpu()
    mid = 0.5 * (min_plies + max_target_plies)

    wins = [g for g in range(pool_size) if int(winner[g]) == 1 and int(end_type[g]) == 1]
    in_range = [g for g in wins if min_plies <= int(plies[g]) <= max_target_plies]
    candidates = in_range or wins or [g for g in range(pool_size) if int(winner[g]) == 1]
    if not candidates:
        candidates = list(range(pool_size))
    best_g = min(candidates, key=lambda g: (abs(int(plies[g]) - mid), -int(bat_cpu[g])))

    ch_t = torch.tensor([best_g], dtype=torch.long)
    forced = [step_acts[ch_t] for step_acts in actions_seq]
    reps = record_games._record_batch(
        net=net, opp_policy=opp_pol, is_self_play=False,
        mover_setup=mover_setup[ch_t], other_setup=other_setup[ch_t],
        start_side=start_side[ch_t], a_side=a_side[ch_t],
        device=device, temperature=temperature, seed=seed,
        max_plies=1500, no_battle_plies=200,
        model_name=m_label, model_label=m_label, opp_name=o_name,
        amp=use_amp, forced_actions=forced,
    )
    return reps[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description='Extract a standalone single-file HTML Stratego replay.')
    ap.add_argument('--json', default=None, help='Existing replay JSON to embed (skips recording)')
    ap.add_argument('--net', default=None, help='Checkpoint .pt to record a showcase game from')
    ap.add_argument('--opponent', default='expert')
    ap.add_argument('--safe', action='store_true', help='Apply B1 safety filter to --net')
    ap.add_argument('--opp-safe', action='store_true', help='Apply B1 safety filter to --opponent')
    ap.add_argument('--model-label', default=None)
    ap.add_argument('--opp-label', default=None)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--min-plies', type=int, default=140)
    ap.add_argument('--max-target-plies', type=int, default=240)
    ap.add_argument('--pool', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', required=True, help='Output standalone .html path')
    ap.add_argument('--save-json', default=None, help='Optional path to also save the raw replay .json')
    ap.add_argument('--title', default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args(argv)

    if args.json:
        with open(args.json, encoding='utf-8') as f:
            replay = json.load(f)
    elif args.net:
        dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        replay = record_showcase_game(
            net_path=args.net, opponent=args.opponent, device=dev,
            safe=args.safe, opp_safe=args.opp_safe, temperature=args.temperature,
            seed=args.seed, pool_size=args.pool,
            min_plies=args.min_plies, max_target_plies=args.max_target_plies,
            model_label=args.model_label, opp_label=args.opp_label,
        )
        if args.save_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(replay, f, separators=(',', ':'))
    else:
        raise ValueError('Pass either --json or --net')

    out_path = embed_replay_html(replay, args.out, title=args.title)
    m = replay['meta']
    battles = sum(1 for p in replay['plies'] if p.get('battle') is not None)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(
        f"[export_replay_html] {m['red']} vs {m['blue']} | winner={m['winner']} ({m['end_type']}) | "
        f"{m['num_plies']} plies, {battles} battles, blunders(ns)={m['blunders']['sure_loss_non_scout']} | "
        f"{size_kb:.1f} KB -> {out_path}"
    )


if __name__ == '__main__':
    main()
