"""
Supervised belief probe models comparing:
  (a)  a_baseline : 59 observation planes only (same trunk/head capacity as PolicyValueNet)
  (b)  b_history  : 59 planes + 48 hand-made per-square history planes
  (c1) c1_aaren   : 59 planes + per-piece AAREN temporal encoder (ported cleanly from network/aaren/)
  (c2) c2_gru     : 59 planes + per-piece GRU temporal encoder control

All four models share the exact same ResNet belief trunk (64ch, 6 residual blocks, 12-class
per-square belief head) so differences isolate the contribution of memory representation.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vec.engine import NUM_OBS_PLANES, NUM_RANKS, NUM_SQUARES
from vec.net import Block
from vec.memory.features import EVENT_DIM, NUM_HISTORY_PLANES


class ResNetBeliefTrunk(nn.Module):
    """Residual tower + 12-rank per-square belief head matching `vec.net.PolicyValueNet`."""

    def __init__(self, in_planes: int, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.in_planes = in_planes
        self.channels = channels
        self.num_blocks = blocks
        coords = torch.stack(
            torch.meshgrid(torch.linspace(-1.0, 1.0, 10), torch.linspace(-1.0, 1.0, 10), indexing='ij')
        )
        self.register_buffer('coords', coords[None])  # (1, 2, 10, 10)
        self.stem = nn.Conv2d(in_planes + 2, channels, 3, padding=1)
        self.blocks = nn.ModuleList([Block(channels) for _ in range(blocks)])
        self.belief = nn.Conv2d(channels, NUM_RANKS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self.coords.expand(x.size(0), -1, -1, -1).to(dtype=x.dtype, device=x.device)
        h = self.stem(torch.cat([x, coords], dim=1))
        for block in self.blocks:
            h = block(h)
        h = F.relu(h)
        return self.belief(h).flatten(2)  # (B, 12, 100)


def aaren_prefix_scan(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal Hillis-Steele parallel prefix scan for AAREN (strictly float32 for stability).

    Args:
        k: (P, L, H) keys
        v: (P, L, H) values
        q: (H,) learned query vector
        scale: attention logit scale (e.g. 1 / sqrt(H))

    Returns:
        (u, w, m): weighted value sum (P, L, H), normalizer (P, L, 1), running max (P, L, 1)
    """
    k32 = k.float()
    v32 = v.float()
    q32 = q.float()
    batch_size, seq_len, hidden_size = k32.shape

    s = (k32 * q32).sum(dim=2, keepdim=True) * scale  # (P, L, 1)
    curr_m = s
    curr_u = v32
    curr_w = torch.ones_like(s)

    if seq_len <= 1:
        return curr_u, curr_w, curr_m

    pad_m_val = -1e20
    num_steps = int(math.ceil(math.log2(float(seq_len))))
    for i in range(num_steps):
        offset = 1 << i
        if offset >= seq_len:
            break
        pad_m = torch.full((batch_size, offset, 1), pad_m_val, device=curr_m.device, dtype=torch.float32)
        prev_m = torch.cat((pad_m, curr_m[:, :-offset, :]), dim=1)

        pad_u = torch.zeros((batch_size, offset, hidden_size), device=curr_u.device, dtype=torch.float32)
        prev_u = torch.cat((pad_u, curr_u[:, :-offset, :]), dim=1)

        pad_w = torch.zeros((batch_size, offset, 1), device=curr_w.device, dtype=torch.float32)
        prev_w = torch.cat((pad_w, curr_w[:, :-offset, :]), dim=1)

        m_new = torch.maximum(prev_m, curr_m)
        exp_prev = torch.exp(prev_m - m_new)
        exp_curr = torch.exp(curr_m - m_new)

        curr_u = prev_u * exp_prev + curr_u * exp_curr
        curr_w = prev_w * exp_prev + curr_w * exp_curr
        curr_m = m_new

    return curr_u, curr_w, curr_m


class AarenLayer(nn.Module):
    """Single AAREN layer supporting both causal parallel prefix scan and O(1) recurrent step."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5
        self.q = nn.Parameter(torch.randn(hidden_size) * self.scale)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x.float()
            k = self.W_k(x32)
            v = self.W_v(x32)
            u, w, _ = aaren_prefix_scan(k, v, self.q, self.scale)
            return self.norm(u / (w + 1e-8))

    def step(
        self,
        x_t: torch.Tensor,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x_t.float()
            k_t = self.W_k(x32)
            v_t = self.W_v(x32)
            s_t = (k_t * self.q).sum(dim=-1, keepdim=True) * self.scale
            if prev_state is None:
                m_t = s_t
                u_t = v_t
                w_t = torch.ones_like(s_t)
            else:
                u_prev, w_prev, m_prev = prev_state
                m_t = torch.maximum(m_prev, s_t)
                exp_prev = torch.exp(m_prev - m_t)
                exp_curr = torch.exp(s_t - m_t)
                u_t = u_prev * exp_prev + v_t * exp_curr
                w_t = w_prev * exp_prev + exp_curr
            out = self.norm(u_t / (w_t + 1e-8))
            return out, (u_t, w_t, m_t)


class PieceAarenEncoder(nn.Module):
    """Per-piece AAREN encoder modeled on `network/aaren/network.py` with causal variable-length support."""

    def __init__(self, event_dim: int = EVENT_DIM, hidden_size: int = 48, num_layers: int = 2):
        super().__init__()
        self.event_dim = event_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_proj = nn.Sequential(nn.Linear(event_dim, hidden_size), nn.GELU())
        self.layers = nn.ModuleList([AarenLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode end-padded sequences `(P, L, event_dim)` using exact real `lengths` `(P,)`."""
        if seq.size(0) == 0:
            return torch.zeros((0, self.hidden_size), dtype=torch.float32, device=seq.device)
        with torch.amp.autocast('cuda', enabled=False):
            h = self.input_proj(seq.float())
            for layer in self.layers:
                h = layer.forward_parallel(h)
            idx = (lengths.long().clamp(min=1, max=h.size(1)) - 1)
            rows = torch.arange(h.size(0), device=h.device)
            return h[rows, idx, :]

    def step(
        self,
        x_t: torch.Tensor,
        prev_states: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """O(1) sequential update matching `forward()` on the accumulated sequence."""
        with torch.amp.autocast('cuda', enabled=False):
            h = self.input_proj(x_t.float())
            new_states = []
            for i, layer in enumerate(self.layers):
                st = None if prev_states is None else prev_states[i]
                h, new_st = layer.step(h, st)
                new_states.append(new_st)
            return h, new_states


class PieceGruEncoder(nn.Module):
    """Standard GRU control over the exact same per-piece event sequences."""

    def __init__(self, event_dim: int = EVENT_DIM, hidden_size: int = 48, num_layers: int = 2):
        super().__init__()
        self.event_dim = event_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_proj = nn.Sequential(nn.Linear(event_dim, hidden_size), nn.GELU())
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if seq.size(0) == 0:
            return torch.zeros((0, self.hidden_size), dtype=torch.float32, device=seq.device)
        with torch.amp.autocast('cuda', enabled=False):
            h0 = self.input_proj(seq.float())
            lens_cpu = lengths.long().clamp(min=1, max=h0.size(1)).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h0, lens_cpu, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
            return self.out_norm(h_n[-1])


class BaselineBeliefNet(nn.Module):
    """Model (a): 59 observation planes only."""

    kind = 'a_baseline'

    def __init__(self, channels: int = 64, blocks: int = 6, in_planes: int = NUM_OBS_PLANES):
        super().__init__()
        self.config = dict(kind=self.kind, channels=channels, blocks=blocks, in_planes=in_planes)
        self.trunk = ResNetBeliefTrunk(in_planes=in_planes, channels=channels, blocks=blocks)

    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.trunk(obs)


class HistoryPlanesBeliefNet(nn.Module):
    """Model (b): 59 observation planes + 48 hand-made spatial history planes."""

    kind = 'b_history'

    def __init__(
        self,
        channels: int = 64,
        blocks: int = 6,
        in_planes: int = NUM_OBS_PLANES,
        history_planes: int = NUM_HISTORY_PLANES,
    ):
        super().__init__()
        self.config = dict(
            kind=self.kind,
            channels=channels,
            blocks=blocks,
            in_planes=in_planes,
            history_planes=history_planes,
        )
        self.trunk = ResNetBeliefTrunk(in_planes=in_planes + history_planes, channels=channels, blocks=blocks)

    def forward(self, obs: torch.Tensor, hist_planes: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.cat([obs, hist_planes.to(dtype=obs.dtype, device=obs.device)], dim=1)
        return self.trunk(x)


class SequenceBeliefNet(nn.Module):
    """Model (c1/c2): 59 observation planes + per-piece learned sequence encoder scattered to `(r, c)`."""

    def __init__(
        self,
        encoder_type: str = 'aaren',
        channels: int = 64,
        blocks: int = 6,
        in_planes: int = NUM_OBS_PLANES,
        event_dim: int = EVENT_DIM,
        embed_dim: int = NUM_HISTORY_PLANES,
        seq_layers: int = 2,
    ):
        super().__init__()
        if encoder_type == 'aaren':
            self.kind = 'c1_aaren'
            self.encoder = PieceAarenEncoder(event_dim=event_dim, hidden_size=embed_dim, num_layers=seq_layers)
        elif encoder_type == 'gru':
            self.kind = 'c2_gru'
            self.encoder = PieceGruEncoder(event_dim=event_dim, hidden_size=embed_dim, num_layers=seq_layers)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.embed_dim = embed_dim
        self.config = dict(
            kind=self.kind,
            encoder_type=encoder_type,
            channels=channels,
            blocks=blocks,
            in_planes=in_planes,
            event_dim=event_dim,
            embed_dim=embed_dim,
            seq_layers=seq_layers,
        )
        self.trunk = ResNetBeliefTrunk(in_planes=in_planes + embed_dim, channels=channels, blocks=blocks)

    def forward(
        self,
        obs: torch.Tensor,
        piece_batch: torch.Tensor,
        piece_sq: torch.Tensor,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        B = obs.size(0)
        grid = torch.zeros((B, self.embed_dim, NUM_SQUARES), dtype=obs.dtype, device=obs.device)
        if piece_batch.numel() > 0:
            emb = self.encoder(seq, lengths).to(dtype=obs.dtype, device=obs.device)
            grid[piece_batch, :, piece_sq] = emb
        grid = grid.view(B, self.embed_dim, 10, 10)
        return self.trunk(torch.cat([obs, grid], dim=1))


MODEL_KINDS = ('a_baseline', 'b_history', 'c1_aaren', 'c2_gru')


def build_model(
    kind: str,
    channels: int = 64,
    blocks: int = 6,
    embed_dim: int = NUM_HISTORY_PLANES,
    seq_layers: int = 2,
) -> nn.Module:
    if kind == 'a_baseline':
        return BaselineBeliefNet(channels=channels, blocks=blocks)
    if kind == 'b_history':
        return HistoryPlanesBeliefNet(channels=channels, blocks=blocks)
    if kind == 'c1_aaren':
        return SequenceBeliefNet(encoder_type='aaren', channels=channels, blocks=blocks,
                                 embed_dim=embed_dim, seq_layers=seq_layers)
    if kind == 'c2_gru':
        return SequenceBeliefNet(encoder_type='gru', channels=channels, blocks=blocks,
                                 embed_dim=embed_dim, seq_layers=seq_layers)
    raise ValueError(f"Unknown model kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
