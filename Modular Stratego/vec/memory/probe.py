"""
Supervised belief probe training, evaluation, and decision report for Task A2.

Trains and evaluates four belief models on IDENTICAL game-split data:
  (a)  a_baseline : 59 planes only
  (b)  b_history  : 59 planes + 48 hand-made spatial history planes
  (c1) c1_aaren   : 59 planes + per-piece AAREN temporal encoder
  (c2) c2_gru     : 59 planes + per-piece GRU temporal encoder
alongside the exact analytical public-count baseline (`rank_distribution`) and
optionally `ppo_4`'s frozen auxiliary belief head.

Usage:
    python -m vec.memory.probe --dataset ../runs/memory_probe/dataset.pt \
        --ppo-net ../runs/ppo_4/net_final.pt --out ../runs/memory_probe
"""

import argparse
import copy
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F

from vec.engine import (
    BOMB,
    FLAG,
    MINER,
    NUM_RANKS,
    SPY,
    PieceType,
    rank_distribution,
    tables,
)
from vec.net import load as load_ppo_net

from vec.memory.features import (
    SETUP_KIND_NAMES,
    extract_batch_sequences,
    scatter_history_planes,
)
from vec.memory.models import MODEL_KINDS, build_model, count_parameters
from vec.memory.recorder import dequantize_obs, load_dataset

GENERAL = PieceType.GENERAL.value   # 10
MARSHAL = PieceType.MARSHAL.value   # 11


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Exact tie-aware ROC-AUC via Mann-Whitney U / average ranks (matches sklearn.metrics.roc_auc_score)."""
    y = np.asarray(y_true, dtype=np.bool_)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s, method='average')
    pos_rank_sum = float(ranks[y].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def split_by_game_id(
    game_id: torch.Tensor,
    matchup_kind: Optional[torch.Tensor] = None,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Split sample indices strictly by `game_id` with zero game overlap between train/val/test.

    When `matchup_kind` is supplied and each stratum has >= 3 games, stratifies games by matchup.
    """
    r_tr, r_va, r_te = ratios
    if abs((r_tr + r_va + r_te) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    gid_np = game_id.cpu().numpy()
    unique_games = np.unique(gid_np)
    n_games = int(unique_games.size)
    if n_games < 3:
        raise ValueError(f"Need at least 3 unique games for train/val/test split, got {n_games}")

    rng = np.random.default_rng(seed)

    def _split_group(games: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        perm = rng.permutation(games)
        n = int(perm.size)
        n_te = max(1, int(round(n * r_te)))
        n_va = max(1, int(round(n * r_va)))
        if n_te + n_va >= n:
            n_te = 1
            n_va = 1
        n_tr = n - n_va - n_te
        return perm[:n_tr].tolist(), perm[n_tr:n_tr + n_va].tolist(), perm[n_tr + n_va:].tolist()

    tr_games: List[int] = []
    va_games: List[int] = []
    te_games: List[int] = []

    use_strat = False
    if matchup_kind is not None:
        mk_np = matchup_kind.cpu().numpy()
        # Map each game_id to its matchup_kind (all samples of one game share matchup_kind)
        first_idx = np.searchsorted(gid_np, unique_games) if np.all(gid_np[:-1] <= gid_np[1:]) else None
        if first_idx is None:
            game_to_mk = {}
            for g, m in zip(gid_np.tolist(), mk_np.tolist()):
                if g not in game_to_mk:
                    game_to_mk[g] = m
            g_mk = np.array([game_to_mk[int(g)] for g in unique_games])
        else:
            g_mk = mk_np[first_idx]
        strata = np.unique(g_mk)
        if all(int((g_mk == s).sum()) >= 3 for s in strata):
            use_strat = True
            for s in strata:
                sub = unique_games[g_mk == s]
                a, b, c = _split_group(sub)
                tr_games.extend(a)
                va_games.extend(b)
                te_games.extend(c)

    if not use_strat:
        tr_games, va_games, te_games = _split_group(unique_games)

    tr_set, va_set, te_set = set(tr_games), set(va_games), set(te_games)
    assert tr_set.isdisjoint(va_set) and tr_set.isdisjoint(te_set) and va_set.isdisjoint(te_set)

    tr_mask = np.isin(gid_np, list(tr_set))
    va_mask = np.isin(gid_np, list(va_set))
    te_mask = np.isin(gid_np, list(te_set))

    train_idx = torch.from_numpy(np.nonzero(tr_mask)[0]).long()
    val_idx = torch.from_numpy(np.nonzero(va_mask)[0]).long()
    test_idx = torch.from_numpy(np.nonzero(te_mask)[0]).long()

    info = dict(
        seed=int(seed),
        ratios=list(ratios),
        stratified_by_matchup=use_strat,
        total_games=n_games,
        train_games=len(tr_set),
        val_games=len(va_set),
        test_games=len(te_set),
        train_samples=int(train_idx.numel()),
        val_samples=int(val_idx.numel()),
        test_samples=int(test_idx.numel()),
    )
    return train_idx, val_idx, test_idx, info


def _forward_model_on_batch(
    model: nn.Module,
    kind: str,
    dataset: Dict[str, torch.Tensor],
    batch_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    obs = dequantize_obs(dataset['obs'][batch_idx]).to(device, non_blocking=True)
    if kind == 'a_baseline':
        return model(obs)
    if kind == 'b_history':
        hist = scatter_history_planes(
            piece_summary=dataset['piece_summary'],
            piece_ptr=dataset['piece_ptr'],
            piece_sq=dataset['piece_sq'],
            opp_present=dataset['opp'] > 0,
            opp_mov=dataset['opp_mov'],
            ply=dataset['ply'],
            batch_idx=batch_idx,
            dtype=torch.float32,
            device=device,
        )
        return model(obs, hist_planes=hist)
    if kind in ('c1_aaren', 'c2_gru'):
        p_batch, p_sq, seq, lens = extract_batch_sequences(
            piece_ptr=dataset['piece_ptr'],
            piece_sq=dataset['piece_sq'],
            event_ptr=dataset['event_ptr'],
            event_len=dataset['event_len'],
            events=dataset['events'],
            batch_idx=batch_idx,
            device=device,
        )
        return model(obs, piece_batch=p_batch, piece_sq=p_sq, seq=seq, lengths=lens)
    raise ValueError(f"Unsupported model kind: {kind}")


@torch.no_grad()
def evaluate_predictor(
    predict_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    dataset: Dict[str, torch.Tensor],
    eval_idx: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Evaluate a belief predictor on `eval_idx`.

    `predict_fn(batch_idx)` returns `(probs, log_probs)` on `device`, both `(B, 12, 100)` float32.
    """
    bucket_names = (
        'overall',
        'moved',
        'moved_non_long',
        'unmoved',
        'early',
        'mid',
        'late',
        'setup_heuristic',
        'setup_structured',
        'setup_random',
        'matchup_self_play',
        'matchup_vs_experts',
    )
    correct = {k: 0 for k in bucket_names}
    total = {k: 0 for k in bucket_names}
    nll_sum = {k: 0.0 for k in bucket_names}

    # Buffers for ROC-AUCs on hidden unmoved and hidden moved pieces
    unm_target_chunks: List[np.ndarray] = []
    unm_p_bomb_chunks: List[np.ndarray] = []
    unm_p_flag_chunks: List[np.ndarray] = []

    mov_target_chunks: List[np.ndarray] = []
    mov_non_long_chunks: List[np.ndarray] = []
    mov_p_mg_chunks: List[np.ndarray] = []
    mov_p_marshal_chunks: List[np.ndarray] = []
    mov_p_general_chunks: List[np.ndarray] = []
    mov_p_spy_chunks: List[np.ndarray] = []
    mov_p_miner_chunks: List[np.ndarray] = []

    n_eval = int(eval_idx.numel())
    for start in range(0, n_eval, batch_size):
        b_idx = eval_idx[start:start + batch_size]
        probs, log_probs = predict_fn(b_idx)
        probs = probs.float()
        log_probs = log_probs.float()

        opp = dataset['opp'][b_idx].long().to(device, non_blocking=True)
        opp_rev = dataset['opp_rev'][b_idx].to(device, non_blocking=True)
        opp_mov = dataset['opp_mov'][b_idx].to(device, non_blocking=True)
        opp_long = dataset['opp_long'][b_idx].to(device, non_blocking=True)
        ply = dataset['ply'][b_idx].long().to(device, non_blocking=True)
        sk = dataset['setup_kind'][b_idx].long().to(device, non_blocking=True)
        mk = dataset['matchup_kind'][b_idx].long().to(device, non_blocking=True)

        hidden = (opp > 0) & ~opp_rev
        target = (opp - 1).clamp_min(0)            # 0..11
        pred = probs.argmax(dim=1)                 # (B, 100)
        is_ok = (pred == target)
        ce = F.nll_loss(log_probs, target, reduction='none')  # (B, 100)

        masks = {
            'overall': hidden,
            'moved': hidden & opp_mov,
            'moved_non_long': hidden & opp_mov & ~opp_long,
            'unmoved': hidden & ~opp_mov,
            'early': hidden & (ply < 40)[:, None],
            'mid': hidden & ((ply >= 40) & (ply < 100))[:, None],
            'late': hidden & (ply >= 100)[:, None],
            'setup_heuristic': hidden & (sk == 0)[:, None],
            'setup_structured': hidden & (sk == 1)[:, None],
            'setup_random': hidden & (sk == 2)[:, None],
            'matchup_self_play': hidden & (mk == 0)[:, None],
            'matchup_vs_experts': hidden & (mk > 0)[:, None],
        }
        for name, m in masks.items():
            cnt = int(m.sum().item())
            if cnt > 0:
                total[name] += cnt
                correct[name] += int((is_ok & m).sum().item())
                nll_sum[name] += float(ce[m].sum().item())

        m_unm = masks['unmoved']
        if m_unm.any():
            unm_target_chunks.append(target[m_unm].cpu().numpy().astype(np.int16))
            unm_p_bomb_chunks.append(probs[:, BOMB - 1, :][m_unm].cpu().numpy().astype(np.float32))
            unm_p_flag_chunks.append(probs[:, FLAG - 1, :][m_unm].cpu().numpy().astype(np.float32))

        m_mov = masks['moved']
        if m_mov.any():
            p_m = probs[:, MARSHAL - 1, :][m_mov]
            p_g = probs[:, GENERAL - 1, :][m_mov]
            mov_target_chunks.append(target[m_mov].cpu().numpy().astype(np.int16))
            mov_non_long_chunks.append((~opp_long[m_mov]).cpu().numpy().astype(np.bool_))
            mov_p_mg_chunks.append((p_m + p_g).cpu().numpy().astype(np.float32))
            mov_p_marshal_chunks.append(p_m.cpu().numpy().astype(np.float32))
            mov_p_general_chunks.append(p_g.cpu().numpy().astype(np.float32))
            mov_p_spy_chunks.append(probs[:, SPY - 1, :][m_mov].cpu().numpy().astype(np.float32))
            mov_p_miner_chunks.append(probs[:, MINER - 1, :][m_mov].cpu().numpy().astype(np.float32))

    accuracy = {k: (correct[k] / total[k] if total[k] > 0 else None) for k in bucket_names}
    nll = {k: (nll_sum[k] / total[k] if total[k] > 0 else None) for k in bucket_names}
    counts = {k: total[k] for k in bucket_names}

    aucs: Dict[str, Optional[float]] = {
        'bomb_unmoved': None,
        'flag_unmoved': None,
        'marshal_or_general_moved': None,
        'marshal_moved': None,
        'general_moved': None,
        'marshal_or_general_moved_non_long': None,
        'spy_moved': None,
        'miner_moved': None,
    }
    auc_counts: Dict[str, int] = {}

    if unm_target_chunks:
        ut = np.concatenate(unm_target_chunks)
        pb = np.concatenate(unm_p_bomb_chunks)
        pf = np.concatenate(unm_p_flag_chunks)
        y_bomb = ut == (BOMB - 1)
        y_flag = ut == (FLAG - 1)
        aucs['bomb_unmoved'] = binary_roc_auc(y_bomb, pb)
        aucs['flag_unmoved'] = binary_roc_auc(y_flag, pf)
        auc_counts['unmoved_total'] = int(ut.size)
        auc_counts['unmoved_bomb_pos'] = int(y_bomb.sum())
        auc_counts['unmoved_flag_pos'] = int(y_flag.sum())

    if mov_target_chunks:
        mt = np.concatenate(mov_target_chunks)
        mnl = np.concatenate(mov_non_long_chunks)
        pmg = np.concatenate(mov_p_mg_chunks)
        pm = np.concatenate(mov_p_marshal_chunks)
        pg = np.concatenate(mov_p_general_chunks)
        ps = np.concatenate(mov_p_spy_chunks)
        pmi = np.concatenate(mov_p_miner_chunks)

        y_m = mt == (MARSHAL - 1)
        y_g = mt == (GENERAL - 1)
        y_mg = y_m | y_g
        y_spy = mt == (SPY - 1)
        y_miner = mt == (MINER - 1)

        aucs['marshal_or_general_moved'] = binary_roc_auc(y_mg, pmg)
        aucs['marshal_moved'] = binary_roc_auc(y_m, pm)
        aucs['general_moved'] = binary_roc_auc(y_g, pg)
        if mnl.any():
            aucs['marshal_or_general_moved_non_long'] = binary_roc_auc(y_mg[mnl], pmg[mnl])
        aucs['spy_moved'] = binary_roc_auc(y_spy, ps)
        aucs['miner_moved'] = binary_roc_auc(y_miner, pmi)

        auc_counts['moved_total'] = int(mt.size)
        auc_counts['moved_non_long_total'] = int(mnl.sum())
        auc_counts['moved_mg_pos'] = int(y_mg.sum())
        auc_counts['moved_marshal_pos'] = int(y_m.sum())
        auc_counts['moved_general_pos'] = int(y_g.sum())
        auc_counts['moved_spy_pos'] = int(y_spy.sum())
        auc_counts['moved_miner_pos'] = int(y_miner.sum())

    return dict(
        accuracy=accuracy,
        nll=nll,
        counts=counts,
        aucs=aucs,
        auc_counts=auc_counts,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    kind: str,
    dataset: Dict[str, torch.Tensor],
    eval_idx: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = True,
) -> Dict[str, object]:
    model.eval()
    amp_on = bool(use_amp and device.type == 'cuda')

    def _pred(b_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast('cuda', enabled=amp_on):
            logits = _forward_model_on_batch(model, kind, dataset, b_idx, device)
        log_p = F.log_softmax(logits.float(), dim=1)
        return log_p.exp(), log_p

    return evaluate_predictor(_pred, dataset, eval_idx, device=device, batch_size=batch_size)


@torch.no_grad()
def evaluate_public_baseline(
    dataset: Dict[str, torch.Tensor],
    eval_idx: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, object]:
    """Evaluate exact analytical public-count baseline (`vec.engine.rank_distribution`)."""
    t = tables(device)

    def _pred(b_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        opp = dataset['opp'][b_idx].long().to(device, non_blocking=True)
        opp_rev = dataset['opp_rev'][b_idx].to(device, non_blocking=True)
        opp_mov = dataset['opp_mov'][b_idx].to(device, non_blocking=True)
        opp_long = dataset['opp_long'][b_idx].to(device, non_blocking=True)
        dist = rank_distribution(opp, opp_rev, opp_mov, opp_long, t).float()
        log_p = dist.clamp_min(1e-7).log()
        return dist, log_p

    return evaluate_predictor(_pred, dataset, eval_idx, device=device, batch_size=batch_size)


@torch.no_grad()
def evaluate_ppo_belief_head(
    ppo_net_path: str,
    dataset: Dict[str, torch.Tensor],
    eval_idx: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = True,
) -> Dict[str, object]:
    """Evaluate the frozen auxiliary belief head of a trained PolicyValueNet checkpoint (e.g. ppo_4)."""
    net = load_ppo_net(ppo_net_path, device)
    net.eval()
    amp_on = bool(use_amp and device.type == 'cuda')

    def _pred(b_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = dequantize_obs(dataset['obs'][b_idx]).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_on):
            _, _, belief = net(obs.contiguous(memory_format=torch.channels_last), with_belief=True)
        log_p = F.log_softmax(belief.float(), dim=1)
        return log_p.exp(), log_p

    return evaluate_predictor(_pred, dataset, eval_idx, device=device, batch_size=batch_size)


def train_belief_model(
    kind: str,
    dataset: Dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    lr_end: float = 1e-5,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    channels: int = 64,
    blocks: int = 6,
    embed_dim: int = 48,
    seq_layers: int = 2,
    seed: int = 0,
    use_amp: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Train one belief model under the shared controlled protocol, selecting best val NLL."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = build_model(kind, channels=channels, blocks=blocks, embed_dim=embed_dim, seq_layers=seq_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr_end)
    amp_on = bool(use_amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_on)

    n_train = int(train_idx.numel())
    n_val = int(val_idx.numel())
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        gen = torch.Generator().manual_seed(seed + 10000 + epoch)
        perm = train_idx[torch.randperm(n_train, generator=gen)]

        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for start in range(0, n_train, batch_size):
            b_idx = perm[start:start + batch_size]
            opp = dataset['opp'][b_idx].long().to(device, non_blocking=True)
            opp_rev = dataset['opp_rev'][b_idx].to(device, non_blocking=True)
            hidden = (opp > 0) & ~opp_rev
            n_hid = int(hidden.sum().item())
            if n_hid == 0:
                continue
            target = (opp - 1).clamp_min(0)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_on):
                logits = _forward_model_on_batch(model, kind, dataset, b_idx, device)
            ce = F.cross_entropy(logits.float(), target, reduction='none')
            loss = (ce * hidden).sum() / float(n_hid)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                tr_loss_sum += float(loss.item()) * n_hid
                tr_correct += int(((logits.argmax(dim=1) == target) & hidden).sum().item())
                tr_total += n_hid

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                b_idx = val_idx[start:start + batch_size]
                opp = dataset['opp'][b_idx].long().to(device, non_blocking=True)
                opp_rev = dataset['opp_rev'][b_idx].to(device, non_blocking=True)
                hidden = (opp > 0) & ~opp_rev
                n_hid = int(hidden.sum().item())
                if n_hid == 0:
                    continue
                target = (opp - 1).clamp_min(0)
                with torch.amp.autocast('cuda', enabled=amp_on):
                    logits = _forward_model_on_batch(model, kind, dataset, b_idx, device)
                ce = F.cross_entropy(logits.float(), target, reduction='none')
                val_loss_sum += float(ce[hidden].sum().item())
                val_correct += int(((logits.argmax(dim=1) == target) & hidden).sum().item())
                val_total += n_hid

        tr_loss = tr_loss_sum / max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        cur_lr = float(opt.param_groups[0]['lr'])
        ep_sec = time.time() - t_ep

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        history.append(dict(
            epoch=epoch,
            train_loss=round(tr_loss, 6),
            train_acc=round(tr_acc, 6),
            val_loss=round(val_loss, 6),
            val_acc=round(val_acc, 6),
            lr=cur_lr,
            epoch_sec=round(ep_sec, 2),
        ))
        sched.step()

        if verbose:
            print(
                f"[probe:{kind}] ep {epoch:02d}/{epochs:02d} "
                f"train_nll={tr_loss:.4f} train_acc={100 * tr_acc:.2f}% | "
                f"val_nll={val_loss:.4f} val_acc={100 * val_acc:.2f}% "
                f"lr={cur_lr:.2e} ({ep_sec:.1f}s)",
                flush=True,
            )

    model.load_state_dict(best_state)
    train_info = dict(
        kind=kind,
        config=model.config,
        num_parameters=count_parameters(model),
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 6),
        total_train_sec=round(time.time() - t_start, 2),
        history=history,
    )
    return model, train_info


def compute_deltas_and_decision(results: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Compute exact deltas vs `a_baseline` and apply the Phase-A decision rule."""
    base = results['a_baseline']
    base_acc = base['accuracy']
    base_nll = base['nll']
    base_auc = base['aucs']

    def _diff_pp(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if (a is None or b is None) else round(100.0 * (a - b), 3)

    def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if (a is None or b is None) else round(a - b, 5)

    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for name in ('b_history', 'c1_aaren', 'c2_gru'):
        if name not in results:
            continue
        r = results[name]
        deltas[name] = dict(
            overall_pp=_diff_pp(r['accuracy']['overall'], base_acc['overall']),
            moved_pp=_diff_pp(r['accuracy']['moved'], base_acc['moved']),
            moved_non_long_pp=_diff_pp(r['accuracy']['moved_non_long'], base_acc['moved_non_long']),
            unmoved_pp=_diff_pp(r['accuracy']['unmoved'], base_acc['unmoved']),
            early_pp=_diff_pp(r['accuracy']['early'], base_acc['early']),
            mid_pp=_diff_pp(r['accuracy']['mid'], base_acc['mid']),
            late_pp=_diff_pp(r['accuracy']['late'], base_acc['late']),
            setup_heuristic_pp=_diff_pp(r['accuracy']['setup_heuristic'], base_acc['setup_heuristic']),
            setup_structured_pp=_diff_pp(r['accuracy']['setup_structured'], base_acc['setup_structured']),
            setup_random_pp=_diff_pp(r['accuracy']['setup_random'], base_acc['setup_random']),
            overall_nll=_diff(r['nll']['overall'], base_nll['overall']),
            bomb_unmoved_auc=_diff(r['aucs']['bomb_unmoved'], base_auc['bomb_unmoved']),
            marshal_or_general_moved_auc=_diff(
                r['aucs']['marshal_or_general_moved'], base_auc['marshal_or_general_moved']
            ),
            marshal_moved_auc=_diff(r['aucs']['marshal_moved'], base_auc['marshal_moved']),
            general_moved_auc=_diff(r['aucs']['general_moved'], base_auc['general_moved']),
            marshal_or_general_moved_non_long_auc=_diff(
                r['aucs']['marshal_or_general_moved_non_long'], base_auc['marshal_or_general_moved_non_long']
            ),
        )

    # Apply decision rule: >= +5.0 pp overall OR clear gain (>= +0.03) on Bomb / Marshal+General AUCs
    candidates = [k for k in ('b_history', 'c1_aaren', 'c2_gru') if k in deltas]
    qualifying = []
    for k in candidates:
        d = deltas[k]
        ov = d['overall_pp'] or 0.0
        b_auc = d['bomb_unmoved_auc'] or 0.0
        mg_auc = d['marshal_or_general_moved_auc'] or 0.0
        if ov >= 5.0 or b_auc >= 0.03 or mg_auc >= 0.03:
            qualifying.append(k)

    best_overall_model = max(candidates, key=lambda k: (deltas[k]['overall_pp'] or -1e9)) if candidates else 'a_baseline'
    best_mg_model = max(candidates, key=lambda k: (deltas[k]['marshal_or_general_moved_auc'] or -1e9)) if candidates else 'a_baseline'

    if not qualifying:
        recommendation = 'NO_MEMORY'
        rationale = (
            "Neither hand-made history planes (b) nor learned per-piece sequence memory (c1 AAREN, c2 GRU) "
            "reached +5.0 pp overall or >= +0.03 ROC-AUC on Bomb (unmoved) / Marshal+General (moved) over "
            "(a) 59-plane baseline. Memory is not the primary bottleneck; Phase B should test capacity/search."
        )
    else:
        # Compare (b) vs best of (c1, c2)
        c_models = [k for k in ('c1_aaren', 'c2_gru') if k in deltas]
        best_c = max(c_models, key=lambda k: (deltas[k]['overall_pp'] or -1e9)) if c_models else None
        if 'b_history' in deltas and best_c is not None:
            b_ov = deltas['b_history']['overall_pp'] or 0.0
            c_ov = deltas[best_c]['overall_pp'] or 0.0
            b_mg = deltas['b_history']['marshal_or_general_moved_auc'] or 0.0
            c_mg = deltas[best_c]['marshal_or_general_moved_auc'] or 0.0
            b_bm = deltas['b_history']['bomb_unmoved_auc'] or 0.0
            c_bm = deltas[best_c]['bomb_unmoved_auc'] or 0.0
            if (c_ov - b_ov) < 1.0 and (c_mg - b_mg) < 0.015 and (c_bm - b_bm) < 0.015:
                recommendation = 'ADD_MEMORY_HANDMADE_B'
                rationale = (
                    f"(b_history) qualifies ({b_ov:+.2f} pp overall, {b_bm:+.4f} Bomb AUC, {b_mg:+.4f} M/G AUC) "
                    f"and matches or beats the best sequence model {best_c} ({c_ov:+.2f} pp, {c_bm:+.4f} Bomb AUC, "
                    f"{c_mg:+.4f} M/G AUC) within 1 pp / 0.015 AUC. Prefer (b) for PPO in Phase B (simpler, zero recurrence overhead)."
                )
            else:
                recommendation = f'ADD_MEMORY_{best_c.upper()}'
                rationale = (
                    f"{best_c} beats both (a_baseline) ({c_ov:+.2f} pp overall, {c_bm:+.4f} Bomb AUC, {c_mg:+.4f} M/G AUC) "
                    f"and (b_history) ({b_ov:+.2f} pp, {b_bm:+.4f} Bomb AUC, {b_mg:+.4f} M/G AUC). "
                    f"Recommend {best_c} for Phase B."
                )
        else:
            recommendation = f'ADD_MEMORY_{best_overall_model.upper()}'
            rationale = f"{best_overall_model} meets the decision threshold."

    return dict(
        deltas_vs_a=deltas,
        qualifying_models=qualifying,
        best_overall_model=best_overall_model,
        best_mg_auc_model=best_mg_model,
        recommendation=recommendation,
        rationale=rationale,
    )


def format_summary_markdown(payload: Dict[str, object]) -> str:
    """Render a concise, thorough markdown decision report for `runs/memory_probe/summary.md`."""
    meta = payload['dataset_meta']
    split = payload['split']
    hp = payload['hyperparams']
    results = payload['test_results']
    decision = payload['decision']
    deltas = decision['deltas_vs_a']
    train_info = payload['train_info']

    def pct(v: Optional[float]) -> str:
        return 'n/a' if v is None else f"{100.0 * v:.2f}%"

    def f4(v: Optional[float]) -> str:
        return 'n/a' if v is None else f"{v:.4f}"

    def dpp(v: Optional[float]) -> str:
        return 'n/a' if v is None else f"{v:+.2f} pp"

    def df4(v: Optional[float]) -> str:
        return 'n/a' if v is None else f"{v:+.4f}"

    order = [k for k in ('public_baseline', 'ppo_4_belief_head', 'a_baseline', 'b_history', 'c1_aaren', 'c2_gru')
             if k in results]
    labels = {
        'public_baseline': 'Public counts (`rank_distribution`)',
        'ppo_4_belief_head': '`ppo_4` frozen belief head (ref)',
        'a_baseline': '(a) 59 planes only',
        'b_history': '(b) 59 + hand-made history (48ch)',
        'c1_aaren': '(c1) 59 + per-piece AAREN',
        'c2_gru': '(c2) 59 + per-piece GRU',
    }

    lines = [
        "# Task A2: Supervised Belief Memory Probe (`ppo_4` Games)",
        "",
        f"**Recommendation:** `{decision['recommendation']}`",
        "",
        f"> {decision['rationale']}",
        "",
        "## 1. Dataset & Protocol",
        f"- Source policy: `{meta.get('net_path', 'n/a')}` (`50%` self-play, `50%` vs `expert`/`expert2`/`expert3` from `{meta.get('expert_styles_path', 'n/a')}`)",
        f"- Games recorded: **{meta.get('games_finished', meta.get('num_games')):,}** (`{meta.get('self_play_games', 0):,}` self-play, `{meta.get('vs_expert_games', 0):,}` vs experts)",
        f"- Sampled plies: **{meta.get('num_samples', 0):,}** (early `{meta.get('phase_samples', {}).get('early', 0):,}`, mid `{meta.get('phase_samples', {}).get('mid', 0):,}`, late `{meta.get('phase_samples', {}).get('late', 0):,}`)",
        f"- Setup-kind samples (enemy side): heuristic `{meta.get('setup_kind_samples', {}).get('heuristic', 0):,}`, structured `{meta.get('setup_kind_samples', {}).get('structured', 0):,}`, random `{meta.get('setup_kind_samples', {}).get('random', 0):,}`",
        f"- Strict `game_id` split (zero game overlap, stratified={split['stratified_by_matchup']}): "
        f"train **{split['train_games']:,}** games (`{split['train_samples']:,}` plies), "
        f"val **{split['val_games']:,}** games (`{split['val_samples']:,}` plies), "
        f"test **{split['test_games']:,}** games (`{split['test_samples']:,}` plies)",
        f"- Controlled training budget (identical across a/b/c1/c2): `channels={hp['channels']}`, `blocks={hp['blocks']}`, "
        f"`embed_dim={hp['embed_dim']}`, `seq_layers={hp['seq_layers']}`, `epochs={hp['epochs']}`, `batch_size={hp['batch_size']}`, "
        f"`AdamW(lr={hp['lr']} -> {hp['lr_end']} cosine, wd={hp['weight_decay']})`, `seed={hp['seed']}`, best val-NLL checkpoint.",
        "",
        "## 2. Test-Set Hidden-Piece Accuracy & NLL",
        "",
        "| Model | Params | Overall Acc | NLL | Moved | Moved (non-long) | Unmoved | Early (<40) | Mid (40-99) | Late (>=100) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        r = results[k]
        p_cnt = f"{train_info[k]['num_parameters']:,}" if k in train_info else "—"
        acc = r['accuracy']
        lines.append(
            f"| {labels[k]} | {p_cnt} | **{pct(acc['overall'])}** | {f4(r['nll']['overall'])} | "
            f"{pct(acc['moved'])} | {pct(acc['moved_non_long'])} | {pct(acc['unmoved'])} | "
            f"{pct(acc['early'])} | {pct(acc['mid'])} | {pct(acc['late'])} |"
        )

    lines += [
        "",
        "### By Enemy Setup Kind & Matchup",
        "",
        "| Model | Heuristic | Structured | Random | Self-Play | Vs Experts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        acc = results[k]['accuracy']
        lines.append(
            f"| {labels[k]} | {pct(acc['setup_heuristic'])} | {pct(acc['setup_structured'])} | "
            f"{pct(acc['setup_random'])} | {pct(acc['matchup_self_play'])} | {pct(acc['matchup_vs_experts'])} |"
        )

    lines += [
        "",
        "## 3. Gameplay-Critical ROC-AUCs (Test Set)",
        "",
        "| Model | Bomb (unmoved) | Flag (unmoved) | Marshal/Gen (moved) | Marshal (moved) | General (moved) | M/G (moved non-long) | Spy (moved) | Miner (moved) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        au = results[k]['aucs']
        lines.append(
            f"| {labels[k]} | **{f4(au['bomb_unmoved'])}** | {f4(au['flag_unmoved'])} | "
            f"**{f4(au['marshal_or_general_moved'])}** | {f4(au['marshal_moved'])} | {f4(au['general_moved'])} | "
            f"{f4(au['marshal_or_general_moved_non_long'])} | {f4(au['spy_moved'])} | {f4(au['miner_moved'])} |"
        )

    lines += [
        "",
        "## 4. Deltas vs (a) 59-Plane Baseline",
        "",
        "| Model | Overall | Moved | Moved (non-long) | Unmoved | Random Setup | Bomb AUC (unmoved) | M/G AUC (moved) | Marshal AUC | General AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ('b_history', 'c1_aaren', 'c2_gru'):
        if k not in deltas:
            continue
        d = deltas[k]
        lines.append(
            f"| {labels[k]} | **{dpp(d['overall_pp'])}** | {dpp(d['moved_pp'])} | {dpp(d['moved_non_long_pp'])} | "
            f"{dpp(d['unmoved_pp'])} | {dpp(d['setup_random_pp'])} | **{df4(d['bomb_unmoved_auc'])}** | "
            f"**{df4(d['marshal_or_general_moved_auc'])}** | {df4(d['marshal_moved_auc'])} | {df4(d['general_moved_auc'])} |"
        )

    base_counts = results['a_baseline']['counts']
    base_auc_counts = results['a_baseline']['auc_counts']
    lines += [
        "",
        "## 5. Test Sample Counts & Caveats",
        f"- Test hidden enemy piece instances: overall `{base_counts['overall']:,}` "
        f"(moved `{base_counts['moved']:,}`, moved non-long `{base_counts['moved_non_long']:,}`, unmoved `{base_counts['unmoved']:,}`; "
        f"early `{base_counts['early']:,}`, mid `{base_counts['mid']:,}`, late `{base_counts['late']:,}`; "
        f"heuristic `{base_counts['setup_heuristic']:,}`, structured `{base_counts['setup_structured']:,}`, random `{base_counts['setup_random']:,}`).",
        f"- ROC-AUC positives: unmoved Bombs `{base_auc_counts.get('unmoved_bomb_pos', 0):,}` / `{base_auc_counts.get('unmoved_total', 0):,}`, "
        f"moved Marshal+General `{base_auc_counts.get('moved_mg_pos', 0):,}` "
        f"(Marshal `{base_auc_counts.get('moved_marshal_pos', 0):,}`, General `{base_auc_counts.get('moved_general_pos', 0):,}`) "
        f"/ `{base_auc_counts.get('moved_total', 0):,}`.",
        "- **Caveats:**",
        "  1. Single training seed (`seed=" + str(hp['seed']) + "`) and single dataset seed; small differences (< 0.5 pp) may reflect seed variance.",
        "  2. Data distribution reflects `ppo_4` self-play and `ppo_4` vs three scripted expert styles; human opponents or a memory-aware PPO policy may exhibit different bluffing/movement tells.",
        "  3. Supervised belief accuracy is an upper bound on information present in move history — it tests whether history *carries* predictive signal, not whether RL credit assignment will automatically exploit it in play.",
        "  4. `(b)` tracks exact full-game summary statistics per piece, whereas `(c1)`/`(c2)` observe the most recent `K="
        + str(meta.get('max_events', 24)) + "` events (augmented with first-move ply and total move count).",
        "",
    ]
    return "\n".join(lines)


def run_probe(
    dataset: Dict[str, torch.Tensor],
    out_dir: str,
    device: torch.device,
    models_to_run: Sequence[str] = MODEL_KINDS,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    lr_end: float = 1e-5,
    weight_decay: float = 1e-4,
    channels: int = 64,
    blocks: int = 6,
    embed_dim: int = 48,
    seq_layers: int = 2,
    seed: int = 0,
    ppo_net_path: Optional[str] = None,
    use_amp: bool = True,
    save_checkpoints: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the full controlled supervised memory probe and write `results.json` + `summary.md`."""
    os.makedirs(out_dir, exist_ok=True)
    train_idx, val_idx, test_idx, split_info = split_by_game_id(
        dataset['game_id'], dataset.get('matchup_kind'), seed=seed
    )
    if verbose:
        print(f"[probe] split: {json.dumps(split_info)}", flush=True)

    test_results: Dict[str, Dict[str, object]] = {}
    train_info: Dict[str, Dict[str, object]] = {}

    if verbose:
        print("[probe] evaluating analytical public baseline...", flush=True)
    test_results['public_baseline'] = evaluate_public_baseline(dataset, test_idx, device=device, batch_size=batch_size)

    if ppo_net_path and os.path.exists(ppo_net_path):
        if verbose:
            print(f"[probe] evaluating frozen ppo belief head ({ppo_net_path})...", flush=True)
        test_results['ppo_4_belief_head'] = evaluate_ppo_belief_head(
            ppo_net_path, dataset, test_idx, device=device, batch_size=batch_size, use_amp=use_amp
        )

    for kind in models_to_run:
        if verbose:
            print(f"\n[probe] === Training {kind} ===", flush=True)
        model, t_info = train_belief_model(
            kind=kind,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lr_end=lr_end,
            weight_decay=weight_decay,
            channels=channels,
            blocks=blocks,
            embed_dim=embed_dim,
            seq_layers=seq_layers,
            seed=seed,
            use_amp=use_amp,
            verbose=verbose,
        )
        train_info[kind] = t_info
        test_results[kind] = evaluate_model(
            model, kind, dataset, test_idx, device=device, batch_size=batch_size, use_amp=use_amp
        )
        if save_checkpoints:
            torch.save(
                {'config': model.config, 'state_dict': model.state_dict(), 'train_info': t_info},
                os.path.join(out_dir, f'{kind}.pt'),
            )
        if verbose:
            tr = test_results[kind]
            print(
                f"[probe:{kind}] TEST acc={100 * (tr['accuracy']['overall'] or 0):.2f}% "
                f"(mov={100 * (tr['accuracy']['moved'] or 0):.2f}%, unm={100 * (tr['accuracy']['unmoved'] or 0):.2f}%) | "
                f"BombAUC={tr['aucs']['bomb_unmoved']} | M/GAUC={tr['aucs']['marshal_or_general_moved']}",
                flush=True,
            )

    decision = compute_deltas_and_decision(test_results) if 'a_baseline' in test_results else {}

    payload = dict(
        dataset_meta=dataset.get('meta', {}),
        split=split_info,
        hyperparams=dict(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lr_end=lr_end,
            weight_decay=weight_decay,
            channels=channels,
            blocks=blocks,
            embed_dim=embed_dim,
            seq_layers=seq_layers,
            seed=seed,
            device=str(device),
            use_amp=use_amp,
            ppo_net_path=ppo_net_path or '',
        ),
        train_info=train_info,
        test_results=test_results,
        decision=decision,
    )

    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    if 'a_baseline' in test_results:
        summary_md = format_summary_markdown(payload)
        summary_path = os.path.join(out_dir, 'summary.md')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_md)

    return payload


def main():
    ap = argparse.ArgumentParser(description='Train and evaluate A2 supervised belief probe models.')
    ap.add_argument('--dataset', required=True, help='Path to recorded dataset.pt')
    ap.add_argument('--out', required=True, help='Output directory (e.g. ../runs/memory_probe)')
    ap.add_argument('--ppo-net', default='', help='Optional PPO net checkpoint to evaluate its frozen belief head')
    ap.add_argument('--models', nargs='+', default=list(MODEL_KINDS), choices=list(MODEL_KINDS))
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-end', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--embed-dim', type=int, default=48)
    ap.add_argument('--seq-layers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--no-checkpoints', action='store_true')
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    ds = load_dataset(args.dataset)
    run_probe(
        dataset=ds,
        out_dir=args.out,
        device=dev,
        models_to_run=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_end=args.lr_end,
        weight_decay=args.weight_decay,
        channels=args.channels,
        blocks=args.blocks,
        embed_dim=args.embed_dim,
        seq_layers=args.seq_layers,
        seed=args.seed,
        ppo_net_path=args.ppo_net or None,
        use_amp=not args.no_amp,
        save_checkpoints=not args.no_checkpoints,
        verbose=True,
    )


if __name__ == '__main__':
    main()
