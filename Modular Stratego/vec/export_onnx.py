"""
Export a policy-value network (vec/net.py) to ONNX for the browser player (web/play.html).

    python -m vec.export_onnx <net.pt> [out.onnx]        (from Modular Stratego/)

Input "obs" (B, 59, 10, 10) float32: the mover's observation (vec/engine.py; the page builds the
same planes). Output "policy" (B, 3600): move logits in the action encoding; the page masks the
illegal moves and turns the rest into probabilities. A second output "value" (B,) is the mover's
expected result in [-1, 1].
"""
import json
import os
import sys

import torch

from vec.net import load


class Exported(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        logits, value = self.net(obs)
        return logits, value


def export(path: str, out: str) -> str:
    net = load(path, torch.device('cpu')).to(memory_format=torch.contiguous_format).float().eval()
    planes = net.config['in_planes']
    with torch.no_grad():
        torch.onnx.export(Exported(net), (torch.zeros(1, planes, 10, 10),), out, input_names=['obs'],
                          output_names=['policy', 'value'], opset_version=17, dynamo=False,
                          dynamic_axes={'obs': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}})
    meta = {'input_planes': planes, 'actions': 3600, 'kind': 'policy-value', 'network': net.config,
            'encoding': 'canonical-frame obs (41 thesis planes + 18 public-information planes) / '
                        'index = sq*36 + dir*9 + dist-1', 'source': os.path.basename(path)}
    with open(os.path.splitext(out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + '.onnx'
    print(f"exported {export(src, dst)}")
