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

from config import ARCHITECTURES


# --------------------------------------------------------------------------- #
#  architecture_id versioning (Task 9)                                        #
# --------------------------------------------------------------------------- #
# The model is fully parameterized by (num_res_blocks, num_filters); the
# architecture_id is the canonical label of a body so checkpoints carry their
# identity and a mismatched body can NEVER be silently loaded.

def resolve_architecture_id(num_res_blocks, num_filters):
    """Canonical architecture id for an actual (blocks, filters) body.

    Returns the registry id when the body matches a known architecture, else a
    ``custom-{n_res}x{n_f}`` label (used by tests and small experiments).
    """
    num_res_blocks = int(num_res_blocks)
    num_filters = int(num_filters)
    for arch_id, (n_res, n_f) in ARCHITECTURES.items():
        if n_res == num_res_blocks and n_f == num_filters:
            return arch_id
    return f"custom-{num_res_blocks}x{num_filters}"


def infer_state_dict_architecture_id(state_dict):
    """Best-effort body identity of a RAW state dict (no metadata).

    Used to validate legacy checkpoints that predate architecture_id:
    filter count from ``conv_in.weight``, res-block count from the
    ``res_convs.*`` keys, input planes from the first conv's input dim.
    """
    conv_in_w = state_dict.get("conv_in.weight")
    if conv_in_w is None:
        raise ValueError(
            "state_dict has no conv_in.weight — not a ChessNet body"
        )
    n_f = int(conv_in_w.shape[0])
    n_in = int(conv_in_w.shape[1])
    res_keys = {int(k.split(".")[1]) for k in state_dict
                if k.startswith("res_convs.")}
    n_res = len(res_keys)
    if n_in == 104:  # default 104-plane encoder; only then can a registry id apply
        for arch_id, (r, f) in ARCHITECTURES.items():
            if (r, f) == (n_res, n_f):
                return arch_id
    return f"custom-{n_res}x{n_f}"


def validate_state_dict_body(architecture_id, state_dict):
    """Raise unless ``state_dict`` was produced by a body with the given
    architecture_id (cross-body loads are rejected, never silently accepted).
    """
    inferred = infer_state_dict_architecture_id(state_dict)
    if inferred != architecture_id:
        raise ValueError(
            f"architecture mismatch: state_dict body is {inferred!r} but "
            f"expected {architecture_id!r}; refusing to load tensors into "
            "a different body"
        )
    return True


def build_arch_cfg(architecture_id, remove_conv_bias=False):
    """A minimal Config-sized object for the given architecture id."""
    from config import Config
    if architecture_id not in ARCHITECTURES:
        raise KeyError(
            f"unknown architecture_id {architecture_id!r}; known: "
            f"{sorted(ARCHITECTURES)}"
        )
    cfg = Config()
    cfg.architecture_id = architecture_id
    cfg.num_res_blocks, cfg.num_filters = ARCHITECTURES[architecture_id]
    cfg.remove_conv_bias = bool(remove_conv_bias)
    return cfg


def count_architecture_parameters(architecture_id, remove_conv_bias=False):
    """Total trainable parameter count (weights + biases + BN) for a body."""
    net = ChessNet(build_arch_cfg(architecture_id, remove_conv_bias))
    return sum(p.numel() for p in net.parameters())


class ChessNet(torch.nn.Module):
    """ResNet encoder + spatial policy/value heads, sized by config.Config.

    forward(x): x is (batch, num_input_planes, board_size, board_size) float
    returns (policy_logits, value):
        policy_logits: (batch, policy_size) raw logits; flat index = from_square
                       * policy_planes + plane (NHWC flatten of the 73-plane
                       spatial head output)
        value:         (batch, 1) in [-1, 1]

    Task 9: the instance records ``self.architecture_id`` — the canonical body
    identity (registry id, or ``custom-{n_res}x{n_f}`` when the body was
    overridden) — so checkpoints carry the body they were trained with.
    """

    def __init__(self, cfg):
        super().__init__()
        n_in = cfg.num_input_planes
        n_res = int(getattr(cfg, "num_res_blocks", 6))
        n_f = int(getattr(cfg, "num_filters", 128))
        n_policy_planes = cfg.policy_planes   # 73
        b = cfg.board_size
        flat_size = 32 * b * b

        # ---- canonical body identity ----
        arch_id = getattr(cfg, "architecture_id", None)
        if arch_id and arch_id in ARCHITECTURES:
            reg_res, reg_f = ARCHITECTURES[arch_id]
            if (reg_res, reg_f) != (n_res, n_f):
                arch_id = None  # explicit body override wins over stale label
        self.architecture_id = (
            arch_id if arch_id else resolve_architecture_id(n_res, n_f)
        )
        self.num_res_blocks = n_res
        self.num_filters = n_f

        # ---- v3-only conv-bias removal (never mutates v2 silently) ----
        remove_conv_bias = bool(getattr(cfg, "remove_conv_bias", False))
        if remove_conv_bias and not self.architecture_id.startswith("v3-"):
            raise ValueError(
                f"remove_conv_bias=True requires an explicit v3 architecture "
                f"id; got {self.architecture_id!r}. v2 bodies keep their conv "
                "biases so legacy v2 checkpoints stay byte-compatible."
            )
        conv_bias = not remove_conv_bias

        # ---- body: initial 3x3 conv + BN + ReLU ----
        self.conv_in = nn.Conv2d(n_in, n_f, kernel_size=3, padding=1,
                                 bias=conv_bias)
        self.bn_in = nn.BatchNorm2d(n_f)

        # ---- residual blocks: 3x3 conv + BN + ReLU + 3x3 conv + BN, skip, ReLU ----
        self.res_convs = nn.ModuleList()
        self.res_bns = nn.ModuleList()
        for _ in range(n_res):
            self.res_convs.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(n_f, n_f, kernel_size=3, padding=1,
                                  bias=conv_bias),
                        nn.Conv2d(n_f, n_f, kernel_size=3, padding=1,
                                  bias=conv_bias),
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
        self.policy_conv = nn.Conv2d(n_f, n_policy_planes, kernel_size=1,
                                     bias=conv_bias)

        # ---- value head: 1x1 conv -> 32, ReLU, flatten, linear -> n_f, ReLU, linear -> 1, tanh ----
        self.value_conv = nn.Conv2d(n_f, 32, kernel_size=1, bias=conv_bias)
        self.value_fc1 = nn.Linear(flat_size, n_f)
        self.value_fc2 = nn.Linear(n_f, 1)

        # ---- optional moves-left auxiliary head (KataGo-style) ----
        # Predicts how many plies remain in the game.  It is an auxiliary
        # TARGET, not an output the search uses: its job is to regularise the
        # shared trunk and speed up value convergence, because "who is winning"
        # and "how long until it is over" are learned from the same features.
        #
        # OFF by default.  Turning it on adds parameters and therefore changes
        # the state_dict, so a checkpoint written with the head cannot load
        # into a net without it (and vice versa).  `heads_id` records the head
        # set so the checkpoint layer can say so plainly instead of surfacing
        # a raw missing/unexpected-keys error.
        self.has_moves_left = bool(getattr(cfg, "moves_left_head", False))
        if self.has_moves_left:
            self.moves_left_conv = nn.Conv2d(n_f, 32, kernel_size=1,
                                             bias=conv_bias)
            self.moves_left_fc1 = nn.Linear(flat_size, n_f)
            self.moves_left_fc2 = nn.Linear(n_f, 1)

    @property
    def heads_id(self):
        """Canonical label for the HEAD set, orthogonal to architecture_id.

        architecture_id names the body (blocks x filters); two nets with the
        same body but different heads have different state_dicts, so the head
        set needs its own identity.
        """
        return "pv+ml" if self.has_moves_left else "pv"

    def parameter_count(self):
        """Total trainable parameter count (master weights, always FP32)."""
        return sum(p.numel() for p in self.parameters())

    def body(self, x):
        """ResNet trunk: input planes -> (batch, num_filters, board, board)."""
        x = F.relu(self.bn_in(self.conv_in(x)))
        for convs, bns in zip(self.res_convs, self.res_bns):
            residual = x
            x = F.relu(bns[0](convs[0](x)))
            x = bns[1](convs[1](x))
            x = F.relu(x + residual)
        return x

    def forward(self, x, with_moves_left=False):
        """(policy_logits, value), or (policy_logits, value, moves_left).

        The third output is returned ONLY when explicitly requested, so every
        existing two-tuple call site is unaffected whether or not the head
        exists.
        """
        x = self.body(x)

        # policy head: spatial conv, then NHWC flatten so
        # flat index = from_square * policy_planes + plane
        p = self.policy_conv(x)
        p = p.permute(0, 2, 3, 1).contiguous().view(p.size(0), -1)
        policy_logits = p

        # value head
        v = F.relu(self.value_conv(x))
        # reshape, not view: under channels_last the conv output is NHWC-strided
        # and `view` raises "view size is not compatible with input tensor's
        # size and stride".  reshape falls back to a copy exactly when the
        # tensor is non-contiguous and is a no-op otherwise.
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        if not with_moves_left:
            return policy_logits, value
        if not self.has_moves_left:
            raise RuntimeError(
                "with_moves_left=True but this net has no moves-left head; "
                "build it with cfg.moves_left_head = True"
            )
        # softplus keeps the prediction non-negative without saturating, which
        # a ReLU output would do for every position it ever got wrong low.
        ml = F.relu(self.moves_left_conv(x))
        ml = ml.reshape(ml.size(0), -1)
        ml = F.relu(self.moves_left_fc1(ml))
        moves_left = F.softplus(self.moves_left_fc2(ml))
        return policy_logits, value, moves_left
