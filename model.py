"""Neural network for the AlphaZero-style chess agent (option 1: CNN ResNet + MCTS).

A single `ChessNet` class: ResNet body (conv + BN + ReLU, then residual blocks)
followed by a spatial policy head and a value head.

Sprint B: the policy head is a TRUE spatial 1x1 Conv2d over the board grid
with one output channel per action plane (73), flattened in NHWC order so the
flat logit index is exactly ``from_square * 73 + plane`` (the 4672-action
AlphaZero map).  The value head is a small 1x1 conv + MLP squashed to [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessNet(torch.nn.Module):
    """ResNet encoder + spatial policy/value heads, sized by config.Config.

    forward(x): x is (batch, num_input_planes, board_size, board_size) float
    returns (policy_logits, value):
        policy_logits: (batch, policy_size) raw logits; flat index = from_square
                       * policy_planes + plane (NHWC flatten of the 73-plane
                       spatial head output)
        value:         (batch, 1) in [-1, 1]
    """

    def __init__(self, cfg):
        super().__init__()
        n_in = cfg.num_input_planes
        n_res = cfg.num_res_blocks
        n_f = cfg.num_filters
        n_policy_planes = cfg.policy_planes   # 73
        b = cfg.board_size
        flat_size = 32 * b * b

        # ---- body: initial 3x3 conv + BN + ReLU ----
        self.conv_in = nn.Conv2d(n_in, n_f, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(n_f)

        # ---- residual blocks: 3x3 conv + BN + ReLU + 3x3 conv + BN, skip, ReLU ----
        self.res_convs = nn.ModuleList()
        self.res_bns = nn.ModuleList()
        for _ in range(n_res):
            self.res_convs.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(n_f, n_f, kernel_size=3, padding=1),
                        nn.Conv2d(n_f, n_f, kernel_size=3, padding=1),
                    ]
                )
            )
            self.res_bns.append(
                nn.ModuleList(
                    [nn.BatchNorm2d(n_f), nn.BatchNorm2d(n_f)]
                )
            )

        # ---- policy head: TRUE spatial 1x1 conv, one channel per action plane ----
        # NHWC flatten (square-major, plane-minor) makes the flat logit index
        # from_square * 73 + plane, exactly matching encoding.move_to_index().
        self.policy_conv = nn.Conv2d(n_f, n_policy_planes, kernel_size=1)

        # ---- value head: 1x1 conv -> 32, ReLU, flatten, linear -> n_f, ReLU, linear -> 1, tanh ----
        self.value_conv = nn.Conv2d(n_f, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(flat_size, n_f)
        self.value_fc2 = nn.Linear(n_f, 1)

    def body(self, x):
        """ResNet trunk: input planes -> (batch, num_filters, board, board)."""
        x = F.relu(self.bn_in(self.conv_in(x)))
        for convs, bns in zip(self.res_convs, self.res_bns):
            residual = x
            x = F.relu(bns[0](convs[0](x)))
            x = bns[1](convs[1](x))
            x = F.relu(x + residual)
        return x

    def forward(self, x):
        x = self.body(x)

        # policy head: spatial conv, then NHWC flatten so
        # flat index = from_square * policy_planes + plane
        p = self.policy_conv(x)
        p = p.permute(0, 2, 3, 1).contiguous().view(p.size(0), -1)
        policy_logits = p

        # value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
