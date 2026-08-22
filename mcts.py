"""Monte Carlo Tree Search with PUCT (AlphaZero style) for chess.

A `Node` holds visit count N, total action-value W, prior probability P, and a
dict of children keyed by python-chess move.  The `MCTS` class runs PUCT-guided
simulations from a root node, using the network to evaluate leaf positions, and
returns a move distribution.

Two performance optimisations over the naive AlphaZero loop, both critical on a
small CNN where the GPU is idle most of the time:

1. Board-copy-free descent.  Instead of materialising a full chess.Board for
   every child (python-chess copies the whole move stack), we keep ONE mutable
   board and push/pop moves as we descend each simulation.  Children are pure
   bookkeeping (move, P, N, W) and know nothing about the board.  Only the
   leaf positions in a batch are materialised (one copy each).

2. Batched leaf evaluation + virtual loss.  Within each batch of `batch_size`
   simulations, all leaf positions are encoded and run through the network in a
   SINGLE forward pass (batch-32 is only ~1.4 ms vs ~1.2 ms for batch-1, so
   this collapses ~100 round-trips into ~4).  Virtual loss is added to nodes on
   each in-flight path so concurrent simulations explore different parts of the
   tree instead of all piling onto the same move.

Convention: every stored value (W, and the network's value output) is from the
perspective of the side to move at that node.  Backpropagation adds the value
and flips its sign at every level, so each node's W is automatically in that
node's own perspective.  Selection therefore negates a child's Q before
comparing (the child's side to move is the opponent's), i.e.

    score(child) = -W_child / N_child
                   + c_puct * P_child * sqrt(N_parent) / (1 + N_child)

Terminal positions (checkmate -> -1 for the side to move, draws -> 0) are
recognised via python-chess and backed up without a network call.
"""

import math

import chess
import numpy as np
import torch

import encoding


class Node:
    """A node of the search tree. Pure bookkeeping — no board is stored here.

    Attributes:
        parent:         parent Node (None for the root).
        move:           the move that led from parent to this node (None for root).
        P:              prior probability from the policy network.
        N:              visit count.
        W:              total action value, from this node's side-to-move perspective.
        children:       dict {chess.Move: Node}.
        is_terminal:    True once this position is known to be game over.
        terminal_value: game value (-1 / 0 / +1) once is_terminal is set.
    """

    __slots__ = ("parent", "move", "P", "N", "W", "children",
                 "is_terminal", "terminal_value")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.is_terminal = False
        self.terminal_value = None


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search with PUCT (batched)."""

    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        self.net.eval()  # BatchNorm must use running statistics during search
        self.root = None
        self._board = None  # the single mutable board; at root between sims

    # ------------------------------------------------------------ public API

    def search(self, board, temperature=0.0, num_sims=None):
        """Run a search from `board` and return {chess.Move: probability}.

        num_sims defaults to cfg.num_simulations.  temperature == 0 returns a
        one-hot distribution over the most-visited move.
        """
        if num_sims is None:
            num_sims = self.cfg.num_simulations

        self._board = board.copy()  # own copy; the caller keeps theirs
        self.root = Node()

        if self._board.is_game_over():
            return self._get_policy(temperature)

        self._expand_root()
        self._apply_dirichlet_noise(self.root)

        batch_size = max(1, int(getattr(self.cfg, "batch_size", 1)))
        virtual_loss = float(getattr(self.cfg, "virtual_loss", 3.0))
        remaining = num_sims
        while remaining > 0:
            n = min(batch_size, remaining)
            remaining -= n
            self._run_batch(n, virtual_loss)

        return self._get_policy(temperature)

    # ------------------------------------------------------------- internals

    def _expand_root(self):
        """Evaluate the root position and create its children (one forward pass)."""
        x = torch.from_numpy(encoding.encode_batch([self._board])).to(self.cfg.device)
        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits[0].float()
        moves = list(self._board.legal_moves)
        mask = torch.from_numpy(encoding.moves_to_mask(moves)).to(self.cfg.device)
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        probs = torch.softmax(logits, dim=0)
        self._make_children(self.root, self._board, probs, moves)

    def _make_children(self, node, board, probs, moves=None):
        """Create one child per legal move from an already-computed prior row.

        `moves` is the already-generated legal-move list; reusing it here means
        python-chess legal-move generation runs ONCE per expanded node instead
        of twice (it is the dominant CPU cost of the search).
        """
        if moves is None:
            moves = board.legal_moves
        for move in moves:
            child = Node(parent=node, move=move,
                         prior=float(probs[encoding.move_to_index(move)]))
            node.children[move] = child

        # Re-normalise: the 4096 from->to encoding cannot distinguish
        # promotion pieces, so a promotion position would otherwise carry
        # duplicate priors that sum to more than one.
        total = sum(c.P for c in node.children.values())
        if total > 0.0:
            for child in node.children.values():
                child.P /= total

    def _run_batch(self, n, virtual_loss):
        """Run `n` simulations with virtual-loss parallelisation and one batched
        forward pass over all leaf evaluations."""
        pending = []  # (node, board, path) — leaves awaiting network eval

        for _ in range(n):
            # ---- descend to a leaf (push moves onto the mutable board) ----
            node = self.root
            path = [node]
            while node.children and not node.is_terminal:
                node = self._select(node)
                path.append(node)
                self._board.push(node.move)

            # ---- resolve the leaf: terminal / game-over / needs-eval ----
            if node.is_terminal:
                leaf_kind, leaf_val = "terminal", node.terminal_value
            elif self._board.is_game_over():
                node.is_terminal = True
                node.terminal_value = self._terminal_value(self._board)
                leaf_kind, leaf_val = "terminal", node.terminal_value
            else:
                leaf_kind, leaf_val = "pending", self._board.copy()

            # ---- unwind the mutable board back to the root ----
            for _ in range(len(path) - 1):
                self._board.pop()

            # ---- apply virtual loss so concurrent sims diverge ----
            # (child-perspective W: a "win" for the child = a loss for the
            #  selecting parent, so +vl makes the node less attractive)
            for p in path:
                p.N += virtual_loss
                p.W += virtual_loss

            if leaf_kind == "terminal":
                self._backprop(path, leaf_val, virtual_loss)
            else:
                pending.append((node, leaf_val, path))

        # ---- one batched forward pass over all pending leaves ----
        if pending:
            boards = [b for _, b, _ in pending]
            moves_lists = [list(b.legal_moves) for b in boards]  # generate ONCE per node
            xs = torch.from_numpy(encoding.encode_batch(boards)).to(self.cfg.device)
            masks = torch.from_numpy(
                np.stack([encoding.moves_to_mask(m) for m in moves_lists])
            ).to(self.cfg.device)
            with torch.no_grad():
                logits, values = self.net(xs)
            logits = logits.float().masked_fill(masks < 0.5, float("-inf"))
            probs = torch.softmax(logits, dim=1)
            for i, (node, board, path) in enumerate(pending):
                self._make_children(node, board, probs[i], moves_lists[i])
                self._backprop(path, float(values[i, 0]), virtual_loss)

    def _backprop(self, path, value, virtual_loss):
        """Remove virtual loss and add the real value up the path, flipping sign
        at every level so each node's W ends up in its own perspective."""
        for node in reversed(path):
            node.N -= virtual_loss   # undo virtual visits
            node.W -= virtual_loss   # undo virtual value
            node.N += 1              # real visit
            node.W += value          # real value
            value = -value

    def _select(self, node):
        """Pick the child maximising Q + c_puct * P * sqrt(Np) / (1 + Nc)."""
        c_puct = self.cfg.c_puct
        sqrt_np = math.sqrt(node.N)
        best_child, best_score = None, float("-inf")
        for child in node.children.values():
            # W is stored from the child's own (opponent's) perspective, so
            # negate it to get the value from `node`'s perspective.
            q = 0.0 if child.N == 0 else -child.W / child.N
            score = q + c_puct * child.P * sqrt_np / (1.0 + child.N)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _apply_dirichlet_noise(self, node):
        """Add Dirichlet noise to the root's child priors (root only)."""
        if not node.children:
            return
        eps = self.cfg.dirichlet_epsilon
        alpha = self.cfg.dirichlet_alpha
        noise = np.random.default_rng().dirichlet([alpha] * len(node.children))
        for child, eta in zip(node.children.values(), noise):
            child.P = (1.0 - eps) * child.P + eps * float(eta)

    @staticmethod
    def _terminal_value(board):
        """Game value from the side-to-move's perspective at a finished game."""
        if board.is_checkmate():
            return -1.0  # side to move has been mated
        return 0.0       # stalemate / insufficient material / 50-move / repetition

    def _get_policy(self, temperature):
        """Visit-count distribution over legal moves, temperature-adjusted."""
        moves = list(self._board.legal_moves)
        if not moves:
            return {}

        counts = np.array(
            [self.root.children[m].N if m in self.root.children else 0 for m in moves],
            dtype=np.float64,
        )

        if temperature == 0.0:
            best = moves[int(np.argmax(counts))]
            return {best: 1.0}

        if counts.sum() <= 0.0:  # no visits at all (e.g. num_sims == 0)
            return {m: 1.0 / len(moves) for m in moves}

        probs = np.power(counts, 1.0 / temperature)
        probs = probs / probs.sum()
        return {m: float(p) for m, p in zip(moves, probs)}
