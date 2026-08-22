"""Neural network for the AlphaZero-style chess agent (option 1: CNN ResNet + MCTS).

A single `ChessNet` class: ResNet body (conv + BN + ReLU, then residual blocks)
followed by a policy head (logits over 4096 from->to moves) and a value head
(win/loss/draw estimate squashed to [-1, 1] with tanh).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessNet(torch.nn.Module):
    """ResNet encoder + policy/value heads, sized by config.Config.

    forward(x): x is (batch, num_input_planes, board_size, board_size) float
    returns (policy_logits, value):
        policy_logits: (batch, policy_size) raw logits
        value:         (batch, 1) in [-1, 1]
    """

    def __init__(self, cfg):
        super().__init__()
        n_in = cfg.num_input_planes
        n_res = cfg.num_res_blocks
        n_f = cfg.num_filters
        n_policy = cfg.policy_size
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

        # ---- policy head: 1x1 conv -> 32, ReLU, flatten, linear -> policy_size ----
        self.policy_conv = nn.Conv2d(n_f, 32, kernel_size=1)
        self.policy_fc = nn.Linear(flat_size, n_policy)

        # ---- value head: 1x1 conv -> 32, ReLU, flatten, linear -> n_f, ReLU, linear -> 1, tanh ----
        self.value_conv = nn.Conv2d(n_f, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(flat_size, n_f)
        self.value_fc2 = nn.Linear(n_f, 1)

    def forward(self, x):
        # body
        x = F.relu(self.bn_in(self.conv_in(x)))
        for convs, bns in zip(self.res_convs, self.res_bns):
            residual = x
            x = F.relu(bns[0](convs[0](x)))
            x = bns[1](convs[1](x))
            x = F.relu(x + residual)

        # policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
