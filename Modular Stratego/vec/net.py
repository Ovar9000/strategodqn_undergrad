"""
Policy-value network for the batched engine.

Input: the (B, 41, 10, 10) observation of the side to move. Output: (B, 3600) policy logits in
the action encoding (square * 36 + direction * 9 + distance - 1) and a (B,) value in [-1, 1] from
the mover's point of view. The policy is its own head, so the choice of move does not depend on
small differences between action values (the failure of the Q-learner in the thesis runs).

The trunk is a residual tower without normalisation layers; two coordinate planes are appended
to the input so the convolutional policy head can tell the rows and columns apart.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vec.engine import NUM_OBS_PLANES, NUM_ACTIONS, VecStratego


class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.c2.weight)      # each block starts as the identity
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6, in_planes: int = NUM_OBS_PLANES, value_hidden: int = 256):
        super().__init__()
        self.config = dict(channels=channels, blocks=blocks, in_planes=in_planes, value_hidden=value_hidden)
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), indexing='ij'))
        self.register_buffer('coords', coords[None])                       # (1, 2, 10, 10)
        self.stem = nn.Conv2d(in_planes + 2, channels, 3, padding=1)
        self.blocks = nn.ModuleList([Block(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(), nn.Conv2d(channels, 36, 1))
        self.value_conv = nn.Conv2d(channels, 16, 1)
        self.value_fc = nn.Sequential(nn.Linear(16 * 100 + 2 * channels, value_hidden), nn.ReLU(),
                                      nn.Linear(value_hidden, 1))
        # auxiliary head: the rank of the enemy piece on each square (trained on the hidden ones
        # against the true board, which self-play knows). It makes the trunk carry a belief about
        # the hidden pieces, the job the thesis gave to AAREN.
        self.belief = nn.Conv2d(channels, 12, 1)

    def forward(self, obs, with_belief: bool = False):
        x = torch.cat([obs, self.coords.expand(obs.size(0), -1, -1, -1).to(obs.dtype)], 1)
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = F.relu(x)
        logits = self.policy(x).permute(0, 2, 3, 1).reshape(obs.size(0), NUM_ACTIONS)
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        v = torch.cat([F.relu(self.value_conv(x)).flatten(1), pooled], 1)
        value = torch.tanh(self.value_fc(v)).squeeze(1)
        if with_belief:
            return logits, value, self.belief(x).flatten(2)             # (B, 12, 100) rank logits
        return logits, value


def belief_loss(belief_logits: torch.Tensor, opp: torch.Tensor, hidden: torch.Tensor):
    """Cross-entropy of the predicted rank on the hidden enemy squares, and the accuracy there.
    belief_logits (B, 12, 100); opp (B, 100) true rank values; hidden (B, 100) bool."""
    target = (opp - 1).clamp_min(0)
    ce = F.cross_entropy(belief_logits.float(), target, reduction='none')    # (B, 100)
    n = hidden.sum().clamp_min(1)
    acc = ((belief_logits.argmax(1) == target) & hidden).sum() / n
    return (ce * hidden).sum() / n, acc


def masked_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~legal, float('-inf'))


class NetPolicy:
    """A network as an arena policy: greedy, or sampling at `temperature`."""

    def __init__(self, net: PolicyValueNet, temperature: float = 0.0, name: str = 'net', amp: bool = True):
        self.net, self.temperature, self.name, self.amp = net, temperature, name, amp

    @torch.no_grad()
    def act(self, env: VecStratego, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        obs, _ = env.observe(idx)
        legal = env.legal if idx is None else env.legal[idx]
        self.net.eval()
        with torch.autocast('cuda', enabled=self.amp and obs.is_cuda):
            logits, _ = self.net(obs.contiguous(memory_format=torch.channels_last))
        logits = masked_logits(logits, legal)
        if self.temperature > 0:
            return torch.multinomial(torch.softmax(logits / self.temperature, 1), 1).squeeze(1)
        return logits.argmax(1)


def save(net: PolicyValueNet, path: str, **extra):
    torch.save({'config': net.config, 'state_dict': net.state_dict(), **extra}, path)


def load(path: str, device) -> PolicyValueNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = PolicyValueNet(**ck['config']).to(device).to(memory_format=torch.channels_last)
    net.load_state_dict(ck['state_dict'])
    return net
