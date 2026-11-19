"""
VecStratego: N games of Stratego as tensors on one device, stepped together.

Same rules as environment/environment.py (flag capture, no-moves loss, no-battle draw, turn
limit), same 41-plane observation as environment/observation.py and same 3600-action encoding as
environment/action_encoding.py; test/test_vec_engine.py replays games on both and compares them.

The board is always stored in the canonical frame of the side to move ("own" at the bottom,
"up" = towards the enemy). After every ply the frame is rotated by 180 degrees and the two sides
swap, so no rule needs to know which player is moving.

State, per game (N = number of games, squares row-major 0..99):
    own, opp            (N, 100) long   rank 1..12 (PieceType value) of the piece, 0 = none
    own_rev, opp_rev    (N, 100) bool   the other side knows this piece's rank
    own_mov, opp_mov    (N, 100) bool   the piece has moved
    own_long, opp_long  (N, 100) bool   the piece has made a move of more than one square
    side                (N,) long       +1 when Player 1 is to move, -1 for Player 2
    ply, nb             (N,) long       plies played, plies since the last battle
    done                (N,) bool       game over; winner (N,) long in Player-1 terms (+1/-1/0)
    end_type            (N,) long       0 running, 1 flag, 2 no moves, 3 no-battle draw, 4 turn limit
    legal               (N, 3600) bool  legal actions of the side to move
    last                (N, 4) long     squares of the last two moves in the mover's frame: the
                                        enemy's from / to, the own side's from / to (100 = none)
    own_hist, opp_hist  (N, 4) long     each side's last two quiet moves (src1, dst1, src2, dst2)
                                        in that side's own canonical frame (100 = none; cleared on battle)
"""

from typing import Optional

import torch

from environment.battle import BattleResolver
from environment.piece import PieceType

NUM_RANKS = 12
NUM_SQUARES = 100
NUM_ACTIONS = 3600
FLAG, SPY, SCOUT, MINER, BOMB = (PieceType.FLAG.value, PieceType.SPY.value, PieceType.SCOUT.value,
                                 PieceType.MINER.value, PieceType.BOMB.value)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right (action_encoding order)
LAKES = ((4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7))
END_FLAG, END_NO_MOVES, END_NO_BATTLE, END_TURN_LIMIT = 1, 2, 3, 4
BASE_OBS_PLANES = 41   # the thesis observation (environment/observation.py)
# Extra public information, appended after the thesis planes:
#   41-52  share of each enemy rank that is still hidden (hidden count / starting count): tells
#          which enemy pieces are still in the game, which the board alone does not show
#   53     own pieces that have moved (the enemy knows they are not Bombs or the Flag)
#   54     own pieces that have made a long move (the enemy knows they are Scouts)
#   55-56  the enemy's last move: from-square and to-square
#   57-58  the own side's last move: from-square and to-square
NUM_OBS_PLANES = 59

# The 40 pieces of one side by rank value 1..12 (index 0 unused)
INITIAL_COUNTS = [0, 1, 1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 6]


def _rays() -> torch.Tensor:
    """(4, 100, 9): square reached from each square in each direction at distance 1..9, or 100
    (a sentinel square) once the line has left the board."""
    idx = torch.full((4, NUM_SQUARES, 9), NUM_SQUARES, dtype=torch.long)
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for sq in range(NUM_SQUARES):
            r, c = divmod(sq, 10)
            for k in range(1, 10):
                rr, cc = r + dr * k, c + dc * k
                if not (0 <= rr < 10 and 0 <= cc < 10):
                    break
                idx[d, sq, k - 1] = rr * 10 + cc
    return idx


def _battle_table() -> torch.Tensor:
    """(13, 13) long: outcome when rank a attacks rank d (1 attacker wins, -1 defender, 0 both go)."""
    br = BattleResolver()
    t = torch.zeros(13, 13, dtype=torch.long)
    for a in range(1, 13):
        for d in range(1, 13):
            t[a, d] = br.resolve_battle(PieceType(a), PieceType(d))
    return t


class Tables:
    """Constant lookup tables on one device."""

    def __init__(self, device):
        self.device = device
        rays = _rays()
        self.rays = rays.to(device)                                    # (4, 100, 9)
        self.rays_flat = self.rays.reshape(-1)                         # (3600,) in (direction, square, distance) order
        # action index -> from square, to square (100 = leaves the board)
        a = torch.arange(NUM_ACTIONS)
        sq, rest = a // 36, a % 36
        d, k = rest // 9, rest % 9
        act_to_cpu = rays[d, sq, k]
        self.act_from = sq.to(device)
        self.act_to = act_to_cpu.to(device)
        self.act_long = (k > 0).to(device)
        self.act_dir = d.to(device)
        self.act_k = k.to(device)
        pair_to_action = torch.full((NUM_SQUARES + 1, NUM_SQUARES + 1), -1, dtype=torch.long)
        valid_act = act_to_cpu < NUM_SQUARES
        pair_to_action[sq[valid_act], act_to_cpu[valid_act]] = a[valid_act]
        self.pair_to_action = pair_to_action.to(device)
        lake = torch.zeros(NUM_SQUARES, dtype=torch.bool)
        for r, c in LAKES:
            lake[r * 10 + c] = True
        self.lake = lake.to(device)
        self.battle = _battle_table().to(device)
        win = (self.battle == 1).float()[1:, 1:].clone()               # (12, 12) attacker x defender
        lose = (self.battle == -1).float()[1:, 1:].clone()
        for imm in (FLAG - 1, BOMB - 1):                               # immobile pieces never attack
            win[imm] = 0.0
            lose[imm] = 0.0
        self.win, self.lose = win, lose
        zero = torch.zeros(1, NUM_RANKS, device=device)
        # by rank value 0..12 (0 = no piece): the row of win / lose for that attacker, and the
        # column of win for that defender (which attackers beat it)
        self.win_by_attacker = torch.cat([zero, win], 0)               # (13, 12)
        self.lose_by_attacker = torch.cat([zero, lose], 0)
        self.win_by_defender = torch.cat([zero, win.t()], 0)
        self.initial = torch.tensor(INITIAL_COUNTS[1:], dtype=torch.float32, device=device)
        movable = torch.ones(NUM_RANKS, device=device)
        movable[FLAG - 1] = 0.0
        movable[BOMB - 1] = 0.0
        self.movable = movable
        # neighbours of each square (100 = off the board)
        self.neigh = self.rays[:, :, 0].t().contiguous()               # (100, 4)
        rc = torch.stack([torch.arange(NUM_SQUARES) // 10, torch.arange(NUM_SQUARES) % 10], 1).float()
        self.manhattan = (rc[:, None, :] - rc[None, :, :]).abs().sum(-1).to(device)   # (100, 100)
        self.row = rc[:, 0].to(device)
        self.col = rc[:, 1].to(device)


_tables = {}


def tables(device) -> Tables:
    key = str(device)
    if key not in _tables:
        _tables[key] = Tables(torch.device(device))
    return _tables[key]


class VecStratego:
    def __init__(self, num_games: int, device, max_plies: int = 1500, no_battle_plies: int = 200):
        self.n = num_games
        self.device = torch.device(device)
        self.t = tables(self.device)
        self.max_plies = max_plies
        self.no_battle_plies = no_battle_plies
        z = lambda dtype: torch.zeros(num_games, NUM_SQUARES, dtype=dtype, device=self.device)
        self.own, self.opp = z(torch.long), z(torch.long)
        self.own_rev, self.opp_rev = z(torch.bool), z(torch.bool)
        self.own_mov, self.opp_mov = z(torch.bool), z(torch.bool)
        self.own_long, self.opp_long = z(torch.bool), z(torch.bool)
        v = lambda dtype: torch.zeros(num_games, dtype=dtype, device=self.device)
        self.side = torch.ones(num_games, dtype=torch.long, device=self.device)
        self.ply, self.nb = v(torch.long), v(torch.long)
        self.done = torch.ones(num_games, dtype=torch.bool, device=self.device)  # nothing dealt yet
        self.winner, self.end_type = v(torch.long), v(torch.long)
        self.legal = torch.zeros(num_games, NUM_ACTIONS, dtype=torch.bool, device=self.device)
        self.last = torch.full((num_games, 4), NUM_SQUARES, dtype=torch.long, device=self.device)
        self.own_hist = torch.full((num_games, 4), NUM_SQUARES, dtype=torch.long, device=self.device)
        self.opp_hist = torch.full((num_games, 4), NUM_SQUARES, dtype=torch.long, device=self.device)
        self._arange = torch.arange(num_games, device=self.device)

    # ------------------------------------------------------------------ setup
    def reset(self, idx: torch.Tensor, mover_setup: torch.Tensor, other_setup: torch.Tensor,
              start_side: Optional[torch.Tensor] = None):
        """Deal new games into rows `idx`. Setups are (len(idx), 40) rank values for squares
        60..99 of each side's own canonical frame (its four rows, back row last); the mover's goes
        to the bottom, the other side's is rotated to the top. start_side: +1/-1 per game (who
        the mover is, in Player-1 terms), default +1."""
        m = idx.numel()
        if m == 0:
            return
        own = torch.zeros(m, NUM_SQUARES, dtype=torch.long, device=self.device)
        opp = torch.zeros_like(own)
        own[:, 60:] = mover_setup.to(self.device)
        opp[:, :40] = other_setup.to(self.device).flip(1)   # square 99 - (60 + i) = 39 - i
        self.own[idx], self.opp[idx] = own, opp
        for f in (self.own_rev, self.opp_rev, self.own_mov, self.opp_mov, self.own_long, self.opp_long):
            f[idx] = False
        self.side[idx] = 1 if start_side is None else start_side.to(self.device)
        self.ply[idx] = 0
        self.nb[idx] = 0
        self.last[idx] = NUM_SQUARES
        self.own_hist[idx] = NUM_SQUARES
        self.opp_hist[idx] = NUM_SQUARES
        self.done[idx] = False
        self.winner[idx] = 0
        self.end_type[idx] = 0
        self.legal[idx] = self._legal_mask(self.own[idx], self.opp[idx], self.own_hist[idx])

    # ------------------------------------------------------------------ rules
    def _legal_mask(self, own: torch.Tensor, opp: torch.Tensor,
                    own_hist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, 3600) legal actions of the side whose pieces are `own`."""
        t = self.t
        B = own.size(0)
        hits, clear = ray_state(own, opp, t)
        blocked = torch.zeros(B, 1, dtype=torch.bool, device=own.device)        # sentinel: not a target
        free_for_own = torch.cat([~((own > 0) | t.lake), blocked], 1)
        ok = free_for_own.index_select(1, t.rays_flat).view(B, 4, NUM_SQUARES, 9) & clear
        scout = (own == SCOUT)[:, None, :, None]
        movable = ((own >= SPY) & (own <= PieceType.MARSHAL.value))[:, None, :, None]
        legal = ok & movable
        legal[..., 1:] &= scout                                                 # only Scouts move further
        legal_flat = legal.permute(0, 2, 1, 3).reshape(B, NUM_ACTIONS)
        if own_hist is not None:
            # Official two-square rule: if this side's previous two quiet moves reversed each other
            # (s2 -> d2 then s1 -> d1 with s1 == d2 and d1 == s2), repeating d1 -> s1 is illegal.
            s1, d1, s2, d2 = own_hist[:, 0], own_hist[:, 1], own_hist[:, 2], own_hist[:, 3]
            is_rep = (s1 < NUM_SQUARES) & (s2 < NUM_SQUARES) & (s1 == d2) & (d1 == s2)
            forb_act = t.pair_to_action[d1, s1]
            is_rep = is_rep & (forb_act >= 0)
            ar_b = torch.arange(B, device=own.device)
            legal_flat[ar_b, forb_act.clamp_min(0)] &= ~is_rep
        return legal_flat

    def step(self, actions: torch.Tensor):
        """Play `actions` (N,) for every game that is not done (entries of finished games are
        ignored). Afterwards the frame is the next mover's. Returns (done_now, result):
        done_now (N,) bool for the games that ended on this ply; result (N,) long from the point
        of view of the side that just moved (+1 it won, -1 it lost, 0 draw or still running)."""
        t = self.t
        live = ~self.done
        ar = self._arange
        a = torch.where(live, actions, torch.zeros_like(actions))
        src, dst, long_move = t.act_from[a], t.act_to[a], t.act_long[a]
        dst = dst.clamp(max=NUM_SQUARES - 1)       # finished games may carry a meaningless action
        att = self.own[ar, src]
        dfn = self.opp[ar, dst]
        battle = (dfn > 0) & live
        res = t.battle[att, dfn]
        mover_stays = live & (~battle | (res == 1))
        defender_stays = battle & (res == -1)
        flag_taken = battle & (dfn == FLAG)

        rev_src, long_src = self.own_rev[ar, src], self.own_long[ar, src]
        # the origin square empties
        self.own[ar, src] = torch.where(live, torch.zeros_like(att), att)
        self.own_rev[ar, src] &= ~live
        self.own_mov[ar, src] &= ~live
        self.own_long[ar, src] &= ~live
        # the mover arrives (a quiet move or a won battle)
        self.own[ar, dst] = torch.where(mover_stays, att, self.own[ar, dst])
        self.own_rev[ar, dst] = torch.where(mover_stays, rev_src | battle, self.own_rev[ar, dst])
        self.own_mov[ar, dst] = torch.where(mover_stays, torch.ones_like(mover_stays), self.own_mov[ar, dst])
        self.own_long[ar, dst] = torch.where(mover_stays, long_src | long_move, self.own_long[ar, dst])
        # the defender: stays revealed, or leaves
        gone = battle & ~defender_stays
        self.opp[ar, dst] = torch.where(gone, torch.zeros_like(dfn), self.opp[ar, dst])
        self.opp_rev[ar, dst] = torch.where(battle, defender_stays, self.opp_rev[ar, dst])
        self.opp_mov[ar, dst] &= ~gone
        self.opp_long[ar, dst] &= ~gone

        self.nb = torch.where(live, torch.where(battle, torch.zeros_like(self.nb), self.nb + 1), self.nb)
        self.ply = self.ply + live.long()

        # rotate the frame: the other side moves next
        flip = lambda x: x.flip(1)
        self.own, self.opp = torch.where(live[:, None], flip(self.opp), self.own), \
            torch.where(live[:, None], flip(self.own), self.opp)
        self.own_rev, self.opp_rev = torch.where(live[:, None], flip(self.opp_rev), self.own_rev), \
            torch.where(live[:, None], flip(self.own_rev), self.opp_rev)
        self.own_mov, self.opp_mov = torch.where(live[:, None], flip(self.opp_mov), self.own_mov), \
            torch.where(live[:, None], flip(self.own_mov), self.opp_mov)
        self.own_long, self.opp_long = torch.where(live[:, None], flip(self.opp_long), self.own_long), \
            torch.where(live[:, None], flip(self.own_long), self.opp_long)
        mover_side = self.side.clone()
        self.side = torch.where(live, -self.side, self.side)
        # the move just played becomes "the enemy's last move" in the new frame (squares rotate)
        rot = lambda sq: torch.where(sq < NUM_SQUARES, NUM_SQUARES - 1 - sq, sq)
        new_last = torch.stack([rot(src), rot(dst), rot(self.last[:, 0]), rot(self.last[:, 1])], 1)
        self.last = torch.where(live[:, None], new_last, self.last)

        # Two-square repetition history: each side's last two quiet moves in its own canonical
        # frame (src1, dst1, src2, dst2). Any battle resets both sides; quiet moves push onto the mover.
        empty_hist = torch.full_like(self.own_hist, NUM_SQUARES)
        pushed_own = torch.stack([src, dst, self.own_hist[:, 0], self.own_hist[:, 1]], 1)
        upd_own = torch.where(battle[:, None], empty_hist, pushed_own)
        upd_opp = torch.where(battle[:, None], empty_hist, self.opp_hist)
        self.own_hist, self.opp_hist = (
            torch.where(live[:, None], upd_opp, self.own_hist),
            torch.where(live[:, None], upd_own, self.opp_hist),
        )

        legal = self._legal_mask(self.own, self.opp, self.own_hist)
        self.legal = torch.where(live[:, None], legal, self.legal)
        no_moves = live & ~flag_taken & ~legal.any(1)
        no_battle = live & ~flag_taken & ~no_moves & (self.nb >= self.no_battle_plies) \
            if self.no_battle_plies else torch.zeros_like(live)
        turn_limit = live & ~flag_taken & ~no_moves & ~no_battle & (self.ply >= self.max_plies)
        mover_wins = flag_taken | no_moves
        done_now = mover_wins | no_battle | turn_limit
        result = mover_wins.long()
        self.winner = torch.where(mover_wins, mover_side, self.winner)
        self.end_type = torch.where(flag_taken, torch.full_like(self.end_type, END_FLAG), self.end_type)
        self.end_type = torch.where(no_moves, torch.full_like(self.end_type, END_NO_MOVES), self.end_type)
        self.end_type = torch.where(no_battle, torch.full_like(self.end_type, END_NO_BATTLE), self.end_type)
        self.end_type = torch.where(turn_limit, torch.full_like(self.end_type, END_TURN_LIMIT), self.end_type)
        self.done = self.done | done_now
        return done_now, result

    # ------------------------------------------------------------------ observation
    def hidden_pool(self) -> torch.Tensor:
        """(N, 12) rank counts of the enemy pieces the mover cannot see."""
        hidden_rank = torch.where(self.opp_rev, torch.zeros_like(self.opp), self.opp)   # 0 = none / revealed
        return torch.nn.functional.one_hot(hidden_rank, 13)[..., 1:].sum(1).float()

    def observe(self, idx: Optional[torch.Tensor] = None):
        """Observation of the side to move: (obs, dist) with obs (B, 41, 10, 10) float and dist
        (B, 12, 100), the rank distribution of the enemy piece on each square (0 where none)."""
        g = (lambda x: x) if idx is None else (lambda x: x[idx])
        return encode(g(self.own), g(self.opp), g(self.own_rev), g(self.opp_rev), g(self.opp_mov),
                      g(self.opp_long), g(self.nb), g(self.ply), self.no_battle_plies, self.max_plies, self.t,
                      own_mov=g(self.own_mov), own_long=g(self.own_long), last=g(self.last))


def ray_state(own, opp, t: Tables):
    """(hits, clear), both (B, 4, 100, 9) bool: the square at distance k+1 in a direction is
    occupied (a piece, a lake or the edge of the board), and no square before it is."""
    B = own.size(0)
    occupied = (own > 0) | (opp > 0) | t.lake
    occ = torch.cat([occupied, torch.ones(B, 1, dtype=torch.bool, device=own.device)], 1)
    hits = occ.index_select(1, t.rays_flat).view(B, 4, NUM_SQUARES, 9)
    c = torch.ones(B, 4, NUM_SQUARES, dtype=torch.bool, device=own.device)
    cols = [c]
    for k in range(8):
        c = c & ~hits[..., k]
        cols.append(c)
    return hits, torch.stack(cols, -1)


def rank_distribution(opp, opp_rev, opp_mov, opp_long, t: Tables) -> torch.Tensor:
    """(B, 12, 100): rank distribution of the enemy piece on each square from the mover's view
    (battle_outlook.enemy_rank_distribution with the exact pool of hidden ranks)."""
    B = opp.size(0)
    present = opp > 0
    hidden = present & ~opp_rev
    known = torch.nn.functional.one_hot(torch.where(opp_rev, opp, torch.zeros_like(opp)), 13)[..., 1:].float()  # (B,100,12)
    pool = torch.nn.functional.one_hot(torch.where(hidden, opp, torch.zeros_like(opp)), 13)[..., 1:].sum(1).float()
    long_moved = hidden & opp_long
    moved = hidden & opp_mov & ~opp_long
    unmoved = hidden & ~opp_mov & ~opp_long
    n_long = long_moved.sum(1).float()
    n_moved = moved.sum(1).float()
    c = pool.clone()
    c[:, SCOUT - 1] = (c[:, SCOUT - 1] - n_long).clamp_min(0.0)
    mov = c * t.movable
    total = mov.sum(1, keepdim=True)
    d_moved = torch.where(total > 0, mov / total.clamp_min(1e-9), (t.movable / t.movable.sum()).expand(B, -1))
    rest = (c - n_moved.view(-1, 1) * d_moved).clamp_min(0.0)
    total = rest.sum(1, keepdim=True)
    d_unmoved = torch.where(total > 0, rest / total.clamp_min(1e-9), torch.full_like(rest, 1.0 / NUM_RANKS))
    dist = known + unmoved[..., None] * d_unmoved[:, None, :] + moved[..., None] * d_moved[:, None, :]
    dist[..., SCOUT - 1] += long_moved.float()
    return dist.transpose(1, 2).contiguous()                                    # (B, 12, 100)


def encode(own, opp, own_rev, opp_rev, opp_mov, opp_long, nb, ply, no_battle_plies, max_plies, t: Tables,
           own_mov=None, own_long=None, last=None):
    B, dev = own.size(0), own.device
    obs = torch.zeros(B, NUM_OBS_PLANES, NUM_SQUARES, dtype=torch.float32, device=dev)
    hidden = (opp > 0) & ~opp_rev
    known = opp_rev & (opp > 0)
    obs.scatter_(1, (own - 1).clamp_min(0)[:, None, :], (own > 0).float()[:, None, :])
    obs.scatter_(1, (opp + 11).clamp(12, 23)[:, None, :], known.float()[:, None, :])
    obs[:, 24] = hidden.float()
    obs[:, 25] = t.lake.float()
    obs[:, 26] = (own_rev & (own > 0)).float()
    if no_battle_plies:
        obs[:, 28] = (nb.float() / no_battle_plies).clamp(max=1.0)[:, None]
    if max_plies:
        obs[:, 29] = (ply.float() / max_plies).clamp(max=1.0)[:, None]
    obs[:, 30] = (hidden & opp_mov).float()
    obs[:, 31] = (hidden & opp_long).float()

    dist = rank_distribution(opp, opp_rev, opp_mov, opp_long, t)                             # (B, 12, 100)
    # battle outlook (battle_outlook.outlook_planes): the target of each own piece in each
    # direction is the adjacent square, or for a Scout the first occupied square along the line
    hits, clear = ray_state(own, opp, t)
    first = hits & clear                                                                     # (B, 4, 100, 9)
    k_first = first.float().argmax(-1, keepdim=True)
    hit_sq = t.rays.expand(B, -1, -1, -1).gather(-1, k_first).squeeze(-1)                    # (B, 4, 100)
    hit_sq = torch.where(first.any(-1), hit_sq, torch.full_like(hit_sq, NUM_SQUARES))
    tgt = torch.cat([dist.transpose(1, 2), torch.zeros(B, 1, NUM_RANKS, device=dev)], 1)     # (B, 101, 12)
    adjacent = tgt.index_select(1, t.rays[:, :, 0].reshape(-1)).view(B, 4, NUM_SQUARES, NUM_RANKS)
    ray = tgt.gather(1, hit_sq.view(B, -1, 1).expand(-1, -1, NUM_RANKS)).view(B, 4, NUM_SQUARES, NUM_RANKS)
    reach = torch.where((own == SCOUT)[:, None, :, None], ray, adjacent)
    obs[:, 32:36] = (reach * t.win_by_attacker[own][:, None]).sum(-1)
    obs[:, 36:40] = (reach * t.lose_by_attacker[own][:, None]).sum(-1)
    obs[:, 40] = (adjacent * t.win_by_defender[own][:, None]).sum(-1).amax(1)
    # extra planes (see NUM_OBS_PLANES)
    pool = torch.nn.functional.one_hot(torch.where(hidden, opp, torch.zeros_like(opp)), 13)[..., 1:].sum(1).float()
    obs[:, 41:53] = (pool / t.initial)[:, :, None]
    if own_mov is not None:
        obs[:, 53] = (own_mov & (own > 0)).float()
        obs[:, 54] = (own_long & (own > 0)).float()
    if last is not None:
        obs[:, 55:59] = torch.nn.functional.one_hot(last, NUM_SQUARES + 1)[..., :NUM_SQUARES].float()
    return obs.view(B, NUM_OBS_PLANES, 10, 10), dist
