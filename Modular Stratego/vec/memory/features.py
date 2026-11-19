"""
External piece-identity tracking, per-piece move event histories, and spatial
summary planes for VecStratego (without modifying vec/engine.py).

All features computed in this module depend strictly on PUBLIC observations and
executed (src, dst) move history from the side-to-move's canonical view. Hidden
enemy ranks are NEVER read.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from vec.engine import FLAG, NUM_RANKS, NUM_SQUARES, VecStratego

MAX_EVENTS = 24
MAX_PIECES_PER_SIDE = 40
APPROACH_RADIUS = 4.0
PLY_NORM_SCALE = 1500.0
DWELL_NORM_SCALE = 200.0
DELTA_PLY_SCALE = 100.0
LOG_MOVES_DENOM = math.log(51.0)

SETUP_KIND_NAMES = ('heuristic', 'structured', 'random')
MATCHUP_KIND_NAMES = ('self_play', 'vs_expert', 'vs_expert2', 'vs_expert3')

EVENT_FEATURE_NAMES = (
    'dr',
    'dc',
    'dist',
    'dir_up',
    'dir_down',
    'dir_left',
    'dir_right',
    'long_move',
    'was_attack',
    'dst_row',
    'dst_col',
    'ply_norm',
    'first_move_ply_norm',
    'delta_ply_norm',
    'move_count_norm',
    'age_at_query_norm',
    'toward_flag',
    'approach_own',
    'retreat_own',
    'adj_own_after',
    'any_approach_rev',
    'any_retreat_rev',
    'max_rev_rank_approached',
    'max_rev_rank_retreated',
) + tuple(f'rev_interact_rank_{r}' for r in range(1, NUM_RANKS + 1))

EVENT_DIM = len(EVENT_FEATURE_NAMES)  # 36
EVENT_AGE_IDX = 15

HISTORY_PLANE_NAMES = (
    'times_moved_log',
    'first_move_ply_norm',
    'plies_since_last_move_norm',
    'unmoved_dwell_norm',
    'total_dist_norm',
    'max_dist_norm',
    'mean_dist_norm',
    'long_move_count_norm',
    'frac_advancing',
    'frac_retreating',
    'frac_lateral',
    'frac_toward_flag',
    'ever_attacked',
    'ever_adj_own',
    'ever_approached_own',
    'ever_retreated_own',
    'ever_approached_rev',
    'max_rev_rank_approached',
    'min_rev_rank_approached',
    'ever_retreated_rev',
    'max_rev_rank_retreated',
    'min_rev_rank_retreated',
    'net_row_progress',
    'manhattan_from_start',
) + tuple(f'ever_app_rev_rank_{r}' for r in range(1, NUM_RANKS + 1)) \
  + tuple(f'ever_ret_rev_rank_{r}' for r in range(1, NUM_RANKS + 1))

NUM_HISTORY_PLANES = len(HISTORY_PLANE_NAMES)  # 48
UNMOVED_DWELL_PLANE_IDX = 3


class PieceHistoryTracker:
    """Tracks persistent piece identities and public move-event histories outside VecStratego.

    Each side (0 for Player +1, 1 for Player -1) has up to 40 piece slots (0..39).
    `own_pid` and `opp_pid` mirror `env.own` and `env.opp` in the current mover's canonical
    frame (-1 where empty), flipping 180 degrees and swapping sides on every live step.
    """

    def __init__(self, num_games: int, device, max_events: int = MAX_EVENTS):
        self.n = num_games
        self.device = torch.device(device)
        self.max_events = max_events
        dev = self.device
        self._arange = torch.arange(num_games, device=dev)
        self._rank_vals = torch.arange(1, NUM_RANKS + 1, device=dev, dtype=torch.float32)[None, :]
        self._sq_grid = torch.arange(NUM_SQUARES, device=dev, dtype=torch.long)

        self.own_pid = torch.full((num_games, NUM_SQUARES), -1, dtype=torch.long, device=dev)
        self.opp_pid = torch.full((num_games, NUM_SQUARES), -1, dtype=torch.long, device=dev)

        shape = (num_games, 2, MAX_PIECES_PER_SIDE)
        self.start_sq = torch.zeros(shape, dtype=torch.long, device=dev)
        self.move_count = torch.zeros(shape, dtype=torch.long, device=dev)
        self.first_move_ply = torch.full(shape, -1, dtype=torch.long, device=dev)
        self.last_move_ply = torch.full(shape, -1, dtype=torch.long, device=dev)

        self.total_dist = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.max_dist = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.long_move_count = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.adv_moves = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.ret_moves = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.lat_moves = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.toward_flag_moves = torch.zeros(shape, dtype=torch.float32, device=dev)

        self.ever_attacked = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.ever_adj_own = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.ever_app_own = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.ever_ret_own = torch.zeros(shape, dtype=torch.float32, device=dev)

        self.ever_app_rev = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.ever_ret_rev = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.max_app_rev_rank = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.min_app_rev_rank = torch.full(shape, 13.0, dtype=torch.float32, device=dev)
        self.max_ret_rev_rank = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.min_ret_rev_rank = torch.full(shape, 13.0, dtype=torch.float32, device=dev)

        self.app_rev_by_rank = torch.zeros(shape + (NUM_RANKS,), dtype=torch.float32, device=dev)
        self.ret_rev_by_rank = torch.zeros(shape + (NUM_RANKS,), dtype=torch.float32, device=dev)

        self.event_buf = torch.zeros(shape + (max_events, EVENT_DIM), dtype=torch.float16, device=dev)
        self.event_ply = torch.zeros(shape + (max_events,), dtype=torch.int16, device=dev)
        self.event_len = torch.zeros(shape, dtype=torch.long, device=dev)

    @torch.no_grad()
    def reset_from_env(self, env: VecStratego, idx: Optional[torch.Tensor] = None):
        """Initialize piece IDs and clear all tracked history for rows `idx` after `env.reset`.

        Reads ONLY occupancy (`own > 0`, `opp > 0`) and `env.side` — never any piece ranks.
        """
        if idx is None:
            idx = self._arange
        m = idx.numel()
        if m == 0:
            return
        idx = idx.to(self.device)

        for tensor in (
            self.start_sq, self.move_count, self.total_dist, self.max_dist,
            self.long_move_count, self.adv_moves, self.ret_moves, self.lat_moves,
            self.toward_flag_moves, self.ever_attacked, self.ever_adj_own,
            self.ever_app_own, self.ever_ret_own, self.ever_app_rev, self.ever_ret_rev,
            self.max_app_rev_rank, self.max_ret_rev_rank,
            self.app_rev_by_rank, self.ret_rev_by_rank,
            self.event_buf, self.event_ply, self.event_len,
        ):
            tensor[idx] = 0
        self.first_move_ply[idx] = -1
        self.last_move_ply[idx] = -1
        self.min_app_rev_rank[idx] = 13.0
        self.min_ret_rev_rank[idx] = 13.0

        own_occ = env.own[idx] > 0
        opp_occ = env.opp[idx] > 0
        neg_one = torch.full((m, NUM_SQUARES), -1, dtype=torch.long, device=self.device)
        own_id = torch.where(own_occ, own_occ.long().cumsum(dim=1) - 1, neg_one)
        opp_id = torch.where(opp_occ, opp_occ.long().cumsum(dim=1) - 1, neg_one)
        self.own_pid[idx] = own_id
        self.opp_pid[idx] = opp_id

        mover_s = (env.side[idx] < 0).long()
        other_s = 1 - mover_s

        g_own, sq_own = own_occ.nonzero(as_tuple=True)
        if g_own.numel():
            self.start_sq[idx[g_own], mover_s[g_own], own_id[g_own, sq_own]] = (NUM_SQUARES - 1) - sq_own
        g_opp, sq_opp = opp_occ.nonzero(as_tuple=True)
        if g_opp.numel():
            self.start_sq[idx[g_opp], other_s[g_opp], opp_id[g_opp, sq_opp]] = sq_opp

    @torch.no_grad()
    def before_step(self, env: VecStratego, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract strictly-public move & proximity features before `env.step(actions)` runs.

        At this moment `env.own` is the moving side M and `env.opp` is the waiting observer O.
        We NEVER read `env.own` (M's ranks). For O's side (`env.opp`), we read ONLY:
          - occupancy `env.opp > 0` (O's own pieces)
          - `env.opp == FLAG` (O's own Flag)
          - `torch.where(env.opp_rev, env.opp, 0)` (O's own pieces revealed to M)
        All three are 100% known to observer O.
        """
        t = env.t
        ar = self._arange
        live = ~env.done
        a = torch.where(live, actions.to(self.device), torch.zeros_like(actions, device=self.device))
        src_m = t.act_from[a]
        dst_m = t.act_to[a].clamp(max=NUM_SQUARES - 1)
        is_long = t.act_long[a]

        mover_s = (env.side < 0).long()
        other_s = 1 - mover_s
        ply_before = env.ply.clone()

        src_pid = self.own_pid[ar, src_m].clone()
        def_pid = self.opp_pid[ar, dst_m].clone()
        was_battle = (env.opp[ar, dst_m] > 0) & live

        # Coordinates in observer O's canonical frame (180-degree rotation of M's frame)
        src_o = (NUM_SQUARES - 1) - src_m
        dst_o = (NUM_SQUARES - 1) - dst_m
        r1, c1 = (src_o // 10).float(), (src_o % 10).float()
        r2, c2 = (dst_o // 10).float(), (dst_o % 10).float()
        dr = r2 - r1
        dc = c2 - c1
        dist_f = dr.abs() + dc.abs()

        # Manhattan distances are invariant under 180-degree rotation
        d_before = t.manhattan[src_m]  # (N, 100)
        d_after = t.manhattan[dst_m]   # (N, 100)
        big = torch.tensor(99.0, device=self.device, dtype=torch.float32)

        o_mask = env.opp > 0
        d_own_b = torch.where(o_mask, d_before, big).amin(dim=1)
        d_own_a = torch.where(o_mask, d_after, big).amin(dim=1)
        has_o = d_own_b < big
        approach_own = (d_own_a < d_own_b) & has_o
        retreat_own = (d_own_a > d_own_b) & has_o
        adj_own_after = (d_own_a <= 1.0) & has_o

        flag_mask = env.opp == FLAG
        df_b = torch.where(flag_mask, d_before, big).amin(dim=1)
        df_a = torch.where(flag_mask, d_after, big).amin(dim=1)
        toward_flag = (df_a < df_b) & (df_b < big)

        # O's own pieces that have been revealed to M
        rev_ranks = torch.where(env.opp_rev, env.opp, torch.zeros_like(env.opp))
        rev_oh = F.one_hot(rev_ranks, NUM_RANKS + 1)[..., 1:].bool()  # (N, 100, 12)
        d_b_r = torch.where(rev_oh, d_before[:, :, None], big).amin(dim=1)  # (N, 12)
        d_a_r = torch.where(rev_oh, d_after[:, :, None], big).amin(dim=1)   # (N, 12)

        near_r = torch.minimum(d_b_r, d_a_r) <= APPROACH_RADIUS
        app_r = near_r & (d_a_r < d_b_r)
        ret_r = near_r & (d_a_r > d_b_r)
        zeros_r = torch.zeros_like(d_a_r)
        app_prox_r = torch.where(app_r, 1.0 / d_a_r.clamp_min(1.0), zeros_r)
        ret_prox_r = torch.where(ret_r, 1.0 / d_b_r.clamp_min(1.0), zeros_r)
        rev_interact_r = app_prox_r - ret_prox_r  # (N, 12) in [-1, 1]

        any_app_rev = app_r.any(dim=1)
        any_ret_rev = ret_r.any(dim=1)
        max_app_rank = torch.where(app_r, self._rank_vals, zeros_r).amax(dim=1)
        min_app_rank = torch.where(app_r, self._rank_vals, torch.full_like(d_a_r, 13.0)).amin(dim=1)
        max_ret_rank = torch.where(ret_r, self._rank_vals, zeros_r).amax(dim=1)
        min_ret_rank = torch.where(ret_r, self._rank_vals, torch.full_like(d_a_r, 13.0)).amin(dim=1)

        return dict(
            live=live,
            src_m=src_m,
            dst_m=dst_m,
            is_long=is_long,
            mover_s=mover_s,
            other_s=other_s,
            ply_before=ply_before,
            src_pid=src_pid,
            def_pid=def_pid,
            was_battle=was_battle,
            dr=dr,
            dc=dc,
            dist_f=dist_f,
            r2=r2,
            c2=c2,
            approach_own=approach_own,
            retreat_own=retreat_own,
            adj_own_after=adj_own_after,
            toward_flag=toward_flag,
            any_app_rev=any_app_rev,
            any_ret_rev=any_ret_rev,
            max_app_rank=max_app_rank,
            min_app_rank=min_app_rank,
            max_ret_rank=max_ret_rank,
            min_ret_rank=min_ret_rank,
            app_prox_r=app_prox_r,
            ret_prox_r=ret_prox_r,
            rev_interact_r=rev_interact_r,
        )

    @torch.no_grad()
    def after_step(self, env: VecStratego, pre: Dict[str, torch.Tensor]):
        """Update piece identities, summary accumulators, and event buffers after `env.step`."""
        ar = self._arange
        live = pre['live']
        src_m = pre['src_m']
        dst_m = pre['dst_m']
        mover_s = pre['mover_s']
        other_s = pre['other_s']
        src_pid = pre['src_pid']
        def_pid = pre['def_pid']
        was_battle = pre['was_battle']

        # After env.step flipped & swapped frames, the mover's square dst_m is now at (99 - dst_m)
        # in env.opp (mover's side) and env.own (defender's side).
        dst_rot = (NUM_SQUARES - 1) - dst_m
        mover_stays = live & (env.opp[ar, dst_rot] > 0)
        defender_stays = was_battle & (env.own[ar, dst_rot] > 0)
        gone = was_battle & ~defender_stays

        neg_one = torch.full_like(src_pid, -1)
        self.own_pid[ar, src_m] = torch.where(live, neg_one, self.own_pid[ar, src_m])
        self.own_pid[ar, dst_m] = torch.where(mover_stays, src_pid, self.own_pid[ar, dst_m])
        self.opp_pid[ar, dst_m] = torch.where(gone, neg_one, self.opp_pid[ar, dst_m])

        new_own_pid = torch.where(live[:, None], self.opp_pid.flip(1), self.own_pid)
        new_opp_pid = torch.where(live[:, None], self.own_pid.flip(1), self.opp_pid)
        self.own_pid, self.opp_pid = new_own_pid, new_opp_pid

        # Record battle participation for surviving defenders
        m_def = defender_stays & (def_pid >= 0)
        if m_def.any():
            self.ever_attacked[ar[m_def], other_s[m_def], def_pid[m_def]] = 1.0

        # Update history for surviving movers
        m_mov = mover_stays & (src_pid >= 0)
        if not m_mov.any():
            return

        g = ar[m_mov]
        s = mover_s[m_mov]
        p = src_pid[m_mov]

        ply_b = pre['ply_before'][m_mov]
        prev_count = self.move_count[g, s, p]
        new_count = prev_count + 1
        prev_ply = self.last_move_ply[g, s, p]
        first_ply = torch.where(prev_count == 0, ply_b, self.first_move_ply[g, s, p])
        delta_ply = torch.where(prev_count == 0, torch.zeros_like(ply_b), (ply_b - prev_ply).clamp_min(0))

        self.move_count[g, s, p] = new_count
        self.first_move_ply[g, s, p] = first_ply
        self.last_move_ply[g, s, p] = ply_b

        dr = pre['dr'][m_mov]
        dc = pre['dc'][m_mov]
        dist_f = pre['dist_f'][m_mov]
        is_long_f = pre['is_long'][m_mov].float()
        was_att_f = was_battle[m_mov].float()
        r2 = pre['r2'][m_mov]
        c2 = pre['c2'][m_mov]
        toward_flag_f = pre['toward_flag'][m_mov].float()
        app_own_f = pre['approach_own'][m_mov].float()
        ret_own_f = pre['retreat_own'][m_mov].float()
        adj_own_f = pre['adj_own_after'][m_mov].float()
        any_app_rev_f = pre['any_app_rev'][m_mov].float()
        any_ret_rev_f = pre['any_ret_rev'][m_mov].float()
        max_app_r = pre['max_app_rank'][m_mov]
        min_app_r = pre['min_app_rank'][m_mov]
        max_ret_r = pre['max_ret_rank'][m_mov]
        min_ret_r = pre['min_ret_rank'][m_mov]
        app_prox_r = pre['app_prox_r'][m_mov]
        ret_prox_r = pre['ret_prox_r'][m_mov]
        rev_interact_r = pre['rev_interact_r'][m_mov]

        self.total_dist[g, s, p] += dist_f
        self.max_dist[g, s, p] = torch.maximum(self.max_dist[g, s, p], dist_f)
        self.long_move_count[g, s, p] += is_long_f
        self.adv_moves[g, s, p] += (dr > 0).float()
        self.ret_moves[g, s, p] += (dr < 0).float()
        self.lat_moves[g, s, p] += (dc != 0).float()
        self.toward_flag_moves[g, s, p] += toward_flag_f

        self.ever_attacked[g, s, p] = torch.maximum(self.ever_attacked[g, s, p], was_att_f)
        self.ever_adj_own[g, s, p] = torch.maximum(self.ever_adj_own[g, s, p], adj_own_f)
        self.ever_app_own[g, s, p] = torch.maximum(self.ever_app_own[g, s, p], app_own_f)
        self.ever_ret_own[g, s, p] = torch.maximum(self.ever_ret_own[g, s, p], ret_own_f)
        self.ever_app_rev[g, s, p] = torch.maximum(self.ever_app_rev[g, s, p], any_app_rev_f)
        self.ever_ret_rev[g, s, p] = torch.maximum(self.ever_ret_rev[g, s, p], any_ret_rev_f)
        self.max_app_rev_rank[g, s, p] = torch.maximum(self.max_app_rev_rank[g, s, p], max_app_r)
        self.min_app_rev_rank[g, s, p] = torch.minimum(self.min_app_rev_rank[g, s, p], min_app_r)
        self.max_ret_rev_rank[g, s, p] = torch.maximum(self.max_ret_rev_rank[g, s, p], max_ret_r)
        self.min_ret_rev_rank[g, s, p] = torch.minimum(self.min_ret_rev_rank[g, s, p], min_ret_r)
        self.app_rev_by_rank[g, s, p] = torch.maximum(self.app_rev_by_rank[g, s, p], app_prox_r)
        self.ret_rev_by_rank[g, s, p] = torch.maximum(self.ret_rev_by_rank[g, s, p], ret_prox_r)

        # Build the 36-dim event feature vector (slot 15 = age_at_query is filled at snapshot/query time)
        M = g.numel()
        feat = torch.empty((M, EVENT_DIM), dtype=torch.float32, device=self.device)
        feat[:, 0] = dr / 9.0
        feat[:, 1] = dc / 9.0
        feat[:, 2] = dist_f / 9.0
        feat[:, 3] = (dr < 0).float()
        feat[:, 4] = (dr > 0).float()
        feat[:, 5] = (dc < 0).float()
        feat[:, 6] = (dc > 0).float()
        feat[:, 7] = is_long_f
        feat[:, 8] = was_att_f
        feat[:, 9] = r2 / 9.0
        feat[:, 10] = c2 / 9.0
        feat[:, 11] = (ply_b.float() / PLY_NORM_SCALE).clamp(max=1.0)
        feat[:, 12] = (first_ply.float() / PLY_NORM_SCALE).clamp(max=1.0)
        feat[:, 13] = (delta_ply.float() / DELTA_PLY_SCALE).clamp(max=1.0)
        feat[:, 14] = (new_count.float() / 20.0).clamp(max=1.0)
        feat[:, EVENT_AGE_IDX] = 0.0
        feat[:, 16] = toward_flag_f
        feat[:, 17] = app_own_f
        feat[:, 18] = ret_own_f
        feat[:, 19] = adj_own_f
        feat[:, 20] = any_app_rev_f
        feat[:, 21] = any_ret_rev_f
        feat[:, 22] = max_app_r / float(NUM_RANKS)
        feat[:, 23] = max_ret_r / float(NUM_RANKS)
        feat[:, 24:] = rev_interact_r

        # Append into chronological fixed-length buffer (shift left when full)
        cur_len = self.event_len[g, s, p]
        full = cur_len >= self.max_events
        if full.any():
            gf, sf, pf = g[full], s[full], p[full]
            self.event_buf[gf, sf, pf, :-1] = self.event_buf[gf, sf, pf, 1:].clone()
            self.event_ply[gf, sf, pf, :-1] = self.event_ply[gf, sf, pf, 1:].clone()
        write_pos = torch.where(full, torch.full_like(cur_len, self.max_events - 1), cur_len)
        self.event_buf[g, s, p, write_pos] = feat.to(torch.float16)
        self.event_ply[g, s, p, write_pos] = ply_b.clamp(max=32767).to(torch.int16)
        self.event_len[g, s, p] = (cur_len + 1).clamp(max=self.max_events)

    @torch.no_grad()
    def step(self, env: VecStratego, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper: pre-step public extraction -> `env.step(actions)` -> post-step update."""
        pre = self.before_step(env, actions)
        done_now, result = env.step(actions)
        self.after_step(env, pre)
        return done_now, result

    @torch.no_grad()
    def _piece_summary_vectors(
        self,
        g: torch.Tensor,
        enemy_s: torch.Tensor,
        p: torch.Tensor,
        sq: torch.Tensor,
        query_ply: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the (P, NUM_HISTORY_PLANES) summary vectors for moved enemy pieces `(g, enemy_s, p)`."""
        P = g.numel()
        out = torch.zeros((P, NUM_HISTORY_PLANES), dtype=torch.float32, device=self.device)
        if P == 0:
            return out

        mc = self.move_count[g, enemy_s, p].float().clamp_min(1.0)
        fply = self.first_move_ply[g, enemy_s, p].float().clamp_min(0.0)
        lply = self.last_move_ply[g, enemy_s, p].float().clamp_min(0.0)
        tdist = self.total_dist[g, enemy_s, p]
        mdist = self.max_dist[g, enemy_s, p]
        start = self.start_sq[g, enemy_s, p]
        sr, sc = (start // 10).float(), (start % 10).float()
        cr, cc = (sq // 10).float(), (sq % 10).float()

        ever_app = self.ever_app_rev[g, enemy_s, p]
        ever_ret = self.ever_ret_rev[g, enemy_s, p]
        min_app = torch.where(ever_app > 0.5, self.min_app_rev_rank[g, enemy_s, p], torch.zeros_like(ever_app))
        min_ret = torch.where(ever_ret > 0.5, self.min_ret_rev_rank[g, enemy_s, p], torch.zeros_like(ever_ret))

        out[:, 0] = torch.log1p(mc) / LOG_MOVES_DENOM
        out[:, 1] = (fply / PLY_NORM_SCALE).clamp(max=1.0)
        out[:, 2] = ((query_ply.float() - lply).clamp_min(0.0) / DWELL_NORM_SCALE).clamp(max=1.0)
        out[:, UNMOVED_DWELL_PLANE_IDX] = 0.0  # filled for unmoved enemy pieces on the board grid
        out[:, 4] = (tdist / 30.0).clamp(max=1.0)
        out[:, 5] = (mdist / 9.0).clamp(max=1.0)
        out[:, 6] = ((tdist / mc) / 9.0).clamp(max=1.0)
        out[:, 7] = (self.long_move_count[g, enemy_s, p] / 5.0).clamp(max=1.0)
        out[:, 8] = self.adv_moves[g, enemy_s, p] / mc
        out[:, 9] = self.ret_moves[g, enemy_s, p] / mc
        out[:, 10] = self.lat_moves[g, enemy_s, p] / mc
        out[:, 11] = self.toward_flag_moves[g, enemy_s, p] / mc
        out[:, 12] = self.ever_attacked[g, enemy_s, p]
        out[:, 13] = self.ever_adj_own[g, enemy_s, p]
        out[:, 14] = self.ever_app_own[g, enemy_s, p]
        out[:, 15] = self.ever_ret_own[g, enemy_s, p]
        out[:, 16] = ever_app
        out[:, 17] = self.max_app_rev_rank[g, enemy_s, p] / float(NUM_RANKS)
        out[:, 18] = min_app / float(NUM_RANKS)
        out[:, 19] = ever_ret
        out[:, 20] = self.max_ret_rev_rank[g, enemy_s, p] / float(NUM_RANKS)
        out[:, 21] = min_ret / float(NUM_RANKS)
        out[:, 22] = (cr - sr) / 9.0
        out[:, 23] = ((cr - sr).abs() + (cc - sc).abs()) / 18.0
        out[:, 24:36] = self.app_rev_by_rank[g, enemy_s, p]
        out[:, 36:48] = self.ret_rev_by_rank[g, enemy_s, p]
        return out

    @torch.no_grad()
    def extract_moved_enemy_snapshot(
        self, env: VecStratego, idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract compact CSR-packed moved-enemy summaries & event sequences for rows `idx`.

        Returns CPU or same-device tensors:
            piece_counts:  (B,) long          number of moved enemy pieces on each sampled board
            piece_sq:      (P,) uint8         current square 0..99 of each moved enemy piece
            piece_summary: (P, 48) float16    hand-made per-piece summary vector
            event_len:     (P,) uint8         number of recorded events (1..max_events)
            events:        (E, 36) float16    concatenated chronological event vectors with age_at_query filled
        """
        if idx is None:
            idx = self._arange
        idx = idx.to(self.device)
        B = idx.numel()
        if B == 0:
            return dict(
                piece_counts=torch.zeros(0, dtype=torch.long, device=self.device),
                piece_sq=torch.zeros(0, dtype=torch.uint8, device=self.device),
                piece_summary=torch.zeros((0, NUM_HISTORY_PLANES), dtype=torch.float16, device=self.device),
                event_len=torch.zeros(0, dtype=torch.uint8, device=self.device),
                events=torch.zeros((0, EVENT_DIM), dtype=torch.float16, device=self.device),
            )

        enemy_s_b = 1 - (env.side[idx] < 0).long()  # (B,)
        pids_grid = self.opp_pid[idx]               # (B, 100)
        present = pids_grid >= 0
        safe_pid = pids_grid.clamp_min(0)
        mc_grid = self.move_count[idx[:, None], enemy_s_b[:, None], safe_pid]
        moved_mask = present & (mc_grid > 0)

        piece_counts = moved_mask.sum(dim=1).long()  # (B,)
        b_i, sq_i = moved_mask.nonzero(as_tuple=True)
        P = b_i.numel()
        if P == 0:
            return dict(
                piece_counts=piece_counts,
                piece_sq=torch.zeros(0, dtype=torch.uint8, device=self.device),
                piece_summary=torch.zeros((0, NUM_HISTORY_PLANES), dtype=torch.float16, device=self.device),
                event_len=torch.zeros(0, dtype=torch.uint8, device=self.device),
                events=torch.zeros((0, EVENT_DIM), dtype=torch.float16, device=self.device),
            )

        g_i = idx[b_i]
        es_i = enemy_s_b[b_i]
        p_i = pids_grid[b_i, sq_i]
        qply_i = env.ply[g_i]

        summary = self._piece_summary_vectors(g_i, es_i, p_i, sq_i, qply_i).to(torch.float16)

        ev_len = self.event_len[g_i, es_i, p_i]  # (P,) in 1..max_events
        max_l = int(ev_len.max())
        ev_pad = self.event_buf[g_i, es_i, p_i, :max_l].float()       # (P, max_l, EVENT_DIM)
        ev_ply = self.event_ply[g_i, es_i, p_i, :max_l].float()       # (P, max_l)
        age = ((qply_i.float()[:, None] - ev_ply).clamp_min(0.0) / DWELL_NORM_SCALE).clamp(max=1.0)
        ev_pad[:, :, EVENT_AGE_IDX] = age

        valid = torch.arange(max_l, device=self.device)[None, :] < ev_len[:, None]
        events_flat = ev_pad[valid].to(torch.float16)  # (E, EVENT_DIM)

        return dict(
            piece_counts=piece_counts,
            piece_sq=sq_i.to(torch.uint8),
            piece_summary=summary,
            event_len=ev_len.to(torch.uint8),
            events=events_flat,
        )

    @torch.no_grad()
    def history_planes(self, env: VecStratego, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build `(B, NUM_HISTORY_PLANES, 10, 10)` float32 spatial history planes directly from tracker."""
        if idx is None:
            idx = self._arange
        idx = idx.to(self.device)
        snap = self.extract_moved_enemy_snapshot(env, idx)
        counts = snap['piece_counts']
        B = idx.numel()
        piece_ptr = torch.zeros(B + 1, dtype=torch.long, device=self.device)
        piece_ptr[1:] = counts.cumsum(0)
        return scatter_history_planes(
            piece_summary=snap['piece_summary'],
            piece_ptr=piece_ptr,
            piece_sq=snap['piece_sq'],
            opp_present=env.opp[idx] > 0,
            opp_mov=env.opp_mov[idx],
            ply=env.ply[idx],
            dtype=torch.float32,
        )


def scatter_history_planes(
    piece_summary: torch.Tensor,
    piece_ptr: torch.Tensor,
    piece_sq: torch.Tensor,
    opp_present: torch.Tensor,
    opp_mov: torch.Tensor,
    ply: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Reconstruct `(B, NUM_HISTORY_PLANES, 10, 10)` from CSR-packed dataset tensors.

    When `batch_idx` is None, assumes `piece_ptr` has length `B + 1` covering all rows of `opp_present`.
    When `batch_idx` is provided, gathers the selected rows cleanly without Python loops.
    """
    dev = opp_present.device if device is None else torch.device(device)
    if batch_idx is not None:
        b_idx = batch_idx.to(piece_ptr.device)
        B = b_idx.numel()
        p_starts = piece_ptr[b_idx]
        p_counts = piece_ptr[b_idx + 1] - p_starts
        total_p = int(p_counts.sum())
        grid = torch.zeros((B, NUM_HISTORY_PLANES, NUM_SQUARES), dtype=dtype, device=dev)
        if total_p > 0:
            ar_b = torch.arange(B, device=piece_ptr.device)
            p_batch = torch.repeat_interleave(ar_b, p_counts)
            cum = p_counts.cumsum(0) - p_counts
            offsets = torch.arange(total_p, device=piece_ptr.device) - torch.repeat_interleave(cum, p_counts)
            p_idx = torch.repeat_interleave(p_starts, p_counts) + offsets
            sq = piece_sq[p_idx].long().to(dev, non_blocking=True)
            summ = piece_summary[p_idx].to(device=dev, dtype=dtype, non_blocking=True)
            grid[p_batch.to(dev, non_blocking=True), :, sq] = summ
        op_idx = batch_idx.to(opp_present.device)
        op_b = opp_present[op_idx].to(dev, non_blocking=True)
        om_b = opp_mov[op_idx].to(dev, non_blocking=True)
        ply_b = ply[op_idx].to(dev, non_blocking=True)
    else:
        B = opp_present.size(0)
        grid = torch.zeros((B, NUM_HISTORY_PLANES, NUM_SQUARES), dtype=dtype, device=dev)
        p_counts = piece_ptr[1:] - piece_ptr[:-1]
        total_p = int(p_counts.sum())
        if total_p > 0:
            p_batch = torch.repeat_interleave(torch.arange(B, device=piece_ptr.device), p_counts).to(dev)
            sq = piece_sq[:total_p].long().to(dev)
            grid[p_batch, :, sq] = piece_summary[:total_p].to(device=dev, dtype=dtype)
        op_b = opp_present.to(dev)
        om_b = opp_mov.to(dev)
        ply_b = ply.to(dev)

    unmoved = op_b & ~om_b
    dwell = (ply_b.float() / DWELL_NORM_SCALE).clamp(max=1.0).to(dtype)
    grid[:, UNMOVED_DWELL_PLANE_IDX, :] = torch.where(unmoved, dwell[:, None], torch.zeros_like(grid[:, 0, :]))
    return grid.view(B, NUM_HISTORY_PLANES, 10, 10)


def extract_batch_sequences(
    piece_ptr: torch.Tensor,
    piece_sq: torch.Tensor,
    event_ptr: torch.Tensor,
    event_len: torch.Tensor,
    events: torch.Tensor,
    batch_idx: torch.Tensor,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather end-padded per-piece event sequences for a minibatch `batch_idx` (zero Python loops).

    Returns `(piece_batch, piece_sq_b, seq_padded, lengths)`:
        piece_batch: (P_b,) long            batch row 0..B-1 of each moved enemy piece
        piece_sq_b:  (P_b,) long            board square 0..99 of each moved enemy piece
        seq_padded:  (P_b, L_max, EVENT_DIM) float32 end-padded chronological event sequence
        lengths:     (P_b,) long            true length of each sequence (>= 1)
    """
    dev = events.device if device is None else torch.device(device)
    b_idx = batch_idx.to(piece_ptr.device)
    B = b_idx.numel()
    p_starts = piece_ptr[b_idx]
    p_counts = piece_ptr[b_idx + 1] - p_starts
    total_p = int(p_counts.sum())
    if total_p == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=dev),
            torch.zeros(0, dtype=torch.long, device=dev),
            torch.zeros((0, 1, EVENT_DIM), dtype=torch.float32, device=dev),
            torch.zeros(0, dtype=torch.long, device=dev),
        )

    ar_b = torch.arange(B, device=piece_ptr.device)
    piece_batch = torch.repeat_interleave(ar_b, p_counts)
    cum = p_counts.cumsum(0) - p_counts
    offsets = torch.arange(total_p, device=piece_ptr.device) - torch.repeat_interleave(cum, p_counts)
    p_idx = torch.repeat_interleave(p_starts, p_counts) + offsets

    sq_b = piece_sq[p_idx].long()
    lens = event_len[p_idx].long()
    e_starts = event_ptr[p_idx]
    max_l = int(lens.max())

    ar_l = torch.arange(max_l, device=event_ptr.device)
    valid = ar_l[None, :] < lens[:, None]
    gather_idx = (e_starts[:, None] + ar_l[None, :]).clamp(max=max(events.size(0) - 1, 0))
    seq = events[gather_idx].float()
    seq = seq * valid[:, :, None].float()

    return (
        piece_batch.to(dev, non_blocking=True),
        sq_b.to(dev, non_blocking=True),
        seq.to(dev, non_blocking=True),
        lens.to(dev, non_blocking=True),
    )
