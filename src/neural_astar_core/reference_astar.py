"""The reference A* search: the reference model for the hardware port.

Why this file exists
--------------------
The upstream repository ships *two* A* implementations and they are not
interchangeable:

``DifferentiableAstar``
    A dense tensor formulation.  Every expansion step runs a softmax over the
    entire H*W map plus a 3x3 convolution, so one step costs O(H*W) and a full
    search costs O((H*W)^2).  It exists so that gradients can flow during
    training.  It is not a sensible thing to build in hardware.

``pq_astar``
    An ordinary priority-queue A*.  This is what actually runs at inference
    time on large maps, and it is what maps onto an FPGA.

This module is a clean re-implementation of ``pq_astar`` semantics, written to
be read alongside an HLS translation.  Differences from upstream are limited to
presentation, plus one deliberate change described below.

Deliberate difference: explicit g
---------------------------------
Upstream never stores g.  It keeps f in the priority queue and rolls the update
forward algebraically::

    f_new = f_sel - (1-r)*h(sel) + r*cost[nei] + (1-r)*h(nei)

This implementation stores g per node and compute ``f = r*g + (1-r)*h`` directly.  The two
are equal up to the constant ``(1-r)*h(start)``, which shifts every f by the
same amount and therefore cannot change the pop order.  ``tests`` asserts this
equivalence against a literal transcription of the upstream formula.  Explicit g
is the right choice for hardware: the accumulated cost has a known, bounded
range that a fixed-point format can be sized around, whereas the rolled-up form
mixes a running sum with a subtraction and is harder to reason about for
overflow.

Semantics worth knowing before porting
--------------------------------------
* **Closed nodes are never reopened.**  Once a node is popped it is final.
  With the learned cost map the heuristic is not guaranteed consistent, so this
  can in principle yield a suboptimal path.  That is upstream behaviour and it
  preserve it -- but it means "Neural A* found a longer path" is sometimes the
  algorithm, not a bug here.
* **Step cost does not depend on direction.**  Moving diagonally costs the same
  as moving straight: ``g_ratio * cost_map[neighbour]``.  No sqrt(2) anywhere.
* **Termination is on *closing* the goal**, not on generating it.
* **The cost map is consumed, not produced, here.**  For vanilla A* it is all
  ones.  For Neural A* it is the CNN's guidance map, computed once up front.
  The network is never consulted inside this loop.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .grid import GridProblem, build_neighbour_table
from .heuristic import DEFAULT_TB_FACTOR, heuristic_map

DEFAULT_G_RATIO = 0.5


@dataclass
class SearchResult:
    """Everything one search produced, including the numbers needed for hardware sizing."""

    success: bool
    path: list[int] = field(default_factory=list)
    """Flat indices from start to goal inclusive.  Empty if the search failed."""

    num_expansions: int = 0
    """Nodes popped from the open list and closed.  This is the primary
    'how much work did the search do' metric, and what Neural A* is trained to
    reduce."""

    peak_open_size: int = 0
    """Largest number of live entries in the open list at any point.  This sizes
    the hardware priority queue -- it is the single most important number for
    deciding between a register-array PQ and a bucket queue."""

    closed: np.ndarray | None = None
    """Boolean ``(H*W,)`` mask of closed nodes -- the 'history' map upstream plots.
    These are the nodes actually *expanded*: popped from the open list and had
    their neighbours generated."""

    generated: np.ndarray | None = None
    """Boolean ``(H*W,)`` mask of every node ever *pushed* onto the open list.

    Always a superset of ``closed``.  The difference between the two is the
    frontier the search built but never got round to expanding, because it
    reached the goal first.  Upstream does not record this and it is easy to
    forget it exists -- which makes A* look like it is magically walking
    straight to the goal on easy maps.  It is not: it evaluates all 8
    neighbours of every expanded node, it just never pops most of them.

    For hardware this is the more honest work metric.  Every generated node
    costs a cost-map read, a heuristic read, an f computation and a queue
    insertion, whether or not it is ever expanded.""" 
    g: np.ndarray | None = None
    """Final cost-to-come per node (inf where never reached)."""

    parent: np.ndarray | None = None
    """Parent pointer per node (-1 where unset).  Backtracking follows this."""

    max_f: float = 0.0
    """Largest f value seen.  Feeds the fixed-point range analysis."""

    def path_cost(self, cost_map_flat: np.ndarray) -> float:
        """Sum of the cost map over the path, excluding the start cell.

        Matches how g accumulates during the search, so this equals ``g[goal]``
        for a successful search.
        """
        if not self.path:
            return float("inf")
        return float(sum(cost_map_flat[idx] for idx in self.path[1:]))

    def path_length(self) -> int:
        """Number of cells on the path, matching upstream's ``paths.sum()`` metric."""
        return len(self.path)

    @property
    def num_generated(self) -> int:
        """Nodes ever pushed onto the open list.  Always >= num_expansions."""
        return 0 if self.generated is None else int(self.generated.sum())

    def closed_map(self, height: int, width: int) -> np.ndarray:
        """Closed (expanded) mask reshaped to the grid, for plotting."""
        if self.closed is None:
            return np.zeros((height, width), dtype=bool)
        return self.closed.reshape(height, width)

    def generated_map(self, height: int, width: int) -> np.ndarray:
        """Generated mask reshaped to the grid, for plotting."""
        if self.generated is None:
            return np.zeros((height, width), dtype=bool)
        return self.generated.reshape(height, width)

    def frontier_map(self, height: int, width: int) -> np.ndarray:
        """Generated but never expanded -- the frontier left on the table.

        Plot this separately from ``closed_map``.  Painting the path over the
        closed set alone makes A* look like it never explored, which on an open
        map is visually indistinguishable from a bug.
        """
        return self.generated_map(height, width) & ~self.closed_map(height, width)

    def path_map(self, height: int, width: int) -> np.ndarray:
        """Path as a binary grid, for plotting."""
        out = np.zeros(height * width, dtype=bool)
        out[self.path] = True
        return out.reshape(height, width)


def astar_search(
    problem: GridProblem,
    cost_map: np.ndarray,
    g_ratio: float = DEFAULT_G_RATIO,
    tb_factor: float = DEFAULT_TB_FACTOR,
    neighbour_table: tuple[np.ndarray, np.ndarray] | None = None,
    max_expansions: int | None = None,
) -> SearchResult:
    """Run A* over ``problem`` using ``cost_map`` as the per-cell step cost.

    Args:
        problem: the grid instance.
        cost_map: ``(H, W)`` or ``(H*W,)`` per-cell cost.  Entering cell ``v``
            costs ``g_ratio * cost_map[v]``.  All ones reproduces vanilla A*;
            the CNN guidance map gives Neural A*.
        g_ratio: weight between cost-to-come and heuristic.  0.5 is upstream's
            default; 0.0 degenerates to greedy best-first search.
        tb_factor: Euclidean tie-break weight in the heuristic.
        neighbour_table: optional precomputed ``(neighbours, valid)`` pair from
            :func:`build_neighbour_table`.  Pass it in when solving many
            problems on the same grid size -- it is pure overhead to rebuild.
        max_expansions: abort after this many pops.  ``None`` means H*W, which
            is enough to close every reachable cell.

    Returns:
        A :class:`SearchResult`.

    The loop below is written to be transcribed.  Each numbered block maps to
    one stage of the intended hardware pipeline.
    """
    height, width = problem.height, problem.width
    num_nodes = problem.num_nodes

    cost_flat = np.asarray(cost_map, dtype=np.float64).reshape(-1)
    if cost_flat.size != num_nodes:
        raise ValueError(
            f"cost_map has {cost_flat.size} cells, grid has {num_nodes}"
        )

    passable = np.asarray(problem.obstacle_map, dtype=np.float64).reshape(-1) != 0.0
    h_flat = heuristic_map(problem.goal, height, width, tb_factor)

    if neighbour_table is None:
        neighbour_table = build_neighbour_table(height, width)
    neighbours, neighbour_valid = neighbour_table

    if max_expansions is None:
        max_expansions = num_nodes

    # --- Search state.  All of this lives in BRAM in the hardware version. ---
    g = np.full(num_nodes, np.inf, dtype=np.float64)  # cost-to-come
    parent = np.full(num_nodes, -1, dtype=np.int32)  # backtrack pointers
    closed = np.zeros(num_nodes, dtype=bool)  # 1 bit/node: the "visited" set
    in_open = np.zeros(num_nodes, dtype=bool)  # 1 bit/node: currently queued
    generated = np.zeros(num_nodes, dtype=bool)  # ever queued (instrumentation)

    # The open list.  heapq has no decrease-key, so a fresh entry is pushed on
    # every improvement and discard stale ones at pop time ("lazy deletion").
    # Entries are (f, idx); the idx makes tie-breaking deterministic, which
    # upstream's pqdict does not guarantee.  Ties are common because of the
    # integer Chebyshev term, so without this two runs could explore in
    # different orders.
    open_heap: list[tuple[float, int]] = []

    g[problem.start] = 0.0
    f_start = g_ratio * 0.0 + (1.0 - g_ratio) * h_flat[problem.start]
    heapq.heappush(open_heap, (f_start, problem.start))
    in_open[problem.start] = True
    generated[problem.start] = True

    result = SearchResult(success=False)
    live_open = 1
    result.peak_open_size = 1
    result.max_f = f_start

    while open_heap:
        # --- 1. POP: take the lowest-f live entry. -----------------------
        # Hardware: one cycle from a bucket queue (occupancy bitmap +
        # leading-zero detect) or a register-array compare tree.
        f_sel, sel = heapq.heappop(open_heap)
        if closed[sel]:
            continue  # stale duplicate left behind by lazy deletion
        if f_sel > g_ratio * g[sel] + (1.0 - g_ratio) * h_flat[sel] + 1e-12:
            continue  # superseded by a better entry for the same node

        # --- 2. CLOSE: mark visited. -------------------------------------
        # Hardware: a single bit written to the visited bitmap.  On a grid the
        # node id *is* the address, so this is exact and costs one BRAM port.
        # This is precisely the operation a Bloom filter would replace -- and
        # why replacing it cannot pay off here.
        closed[sel] = True
        in_open[sel] = False
        live_open -= 1
        result.num_expansions += 1
        result.max_f = max(result.max_f, f_sel)

        # --- 3. GOAL TEST: on close, not on generate. --------------------
        if sel == problem.goal:
            result.success = True
            break

        if result.num_expansions >= max_expansions:
            break

        # --- 4. EXPAND: all 8 Moore neighbours, independently. -----------
        # Hardware: 8 parallel lanes.  Each lane reads g[nei] and the two
        # bitmaps, computes one f, and proposes an update.  The lanes only
        # interact if two of them target the same node, which cannot happen
        # within a single expansion.
        for k in range(neighbours.shape[1]):
            if not neighbour_valid[sel, k]:
                continue
            nei = int(neighbours[sel, k])

            if not passable[nei]:
                continue
            if closed[nei]:
                continue  # never reopened -- see module docstring

            # --- 5. COST: uniform step cost, no diagonal penalty. --------
            g_new = g[sel] + cost_flat[nei]
            if g_new >= g[nei]:
                continue  # not an improvement (also covers the inf case)

            f_new = g_ratio * g_new + (1.0 - g_ratio) * h_flat[nei]

            # --- 6. RELAX: write back and (re)queue. ---------------------
            g[nei] = g_new
            parent[nei] = sel
            if not in_open[nei]:
                in_open[nei] = True
                live_open += 1
            generated[nei] = True
            heapq.heappush(open_heap, (f_new, nei))
            result.max_f = max(result.max_f, f_new)

        result.peak_open_size = max(result.peak_open_size, live_open)

    result.closed = closed
    result.generated = generated
    result.g = g
    result.parent = parent
    if result.success:
        result.path = backtrack(parent, problem.start, problem.goal)

    return result


def backtrack(parent: np.ndarray, start: int, goal: int) -> list[int]:
    """Walk parent pointers from goal back to start, returning start-to-goal order.

    Raises:
        RuntimeError: if the pointers form a cycle or dead-end before reaching
            start.  This should be impossible; it firing means the relax step
            corrupted state, which is exactly the failure a fixed-point port is
            likely to introduce.
    """
    path = [int(goal)]
    current = int(goal)
    guard = len(parent) + 1
    while current != start:
        current = int(parent[current])
        if current < 0:
            raise RuntimeError("backtrack hit an unset parent pointer")
        path.append(current)
        guard -= 1
        if guard <= 0:
            raise RuntimeError("backtrack did not terminate -- cycle in parents")
    path.reverse()
    return path


def upstream_pq_astar(
    problem: GridProblem,
    cost_map: np.ndarray,
    g_ratio: float = DEFAULT_G_RATIO,
    tb_factor: float = DEFAULT_TB_FACTOR,
) -> SearchResult:
    """Literal transcription of upstream ``pq_astar``, kept only for cross-checking.

    This is the rolled-up-f formulation with no stored g.  It is slower and
    harder to read; :func:`astar_search` is the one developed against.  The test
    suite asserts the two agree on path cost and expansion count so that any
    future change to the readable version is caught immediately.
    """
    height, width = problem.height, problem.width
    cost_flat = np.asarray(cost_map, dtype=np.float64).reshape(-1)
    passable = np.asarray(problem.obstacle_map, dtype=np.float64).reshape(-1) != 0.0
    h_flat = heuristic_map(problem.goal, height, width, tb_factor)
    neighbours, neighbour_valid = build_neighbour_table(height, width)

    open_f: dict[int, float] = {problem.start: 0.0}
    closed_f: dict[int, float] = {}
    parent = np.full(problem.num_nodes, -1, dtype=np.int32)
    heap: list[tuple[float, int]] = [(0.0, problem.start)]

    result = SearchResult(success=False)

    while heap:
        f_sel, sel = heapq.heappop(heap)
        if sel in closed_f or open_f.get(sel) != f_sel:
            continue
        del open_f[sel]
        closed_f[sel] = f_sel
        result.num_expansions += 1

        if sel == problem.goal:
            result.success = True
            break

        for k in range(neighbours.shape[1]):
            if not neighbour_valid[sel, k]:
                continue
            nei = int(neighbours[sel, k])
            if not passable[nei]:
                continue

            f_new = (
                f_sel
                - (1.0 - g_ratio) * h_flat[sel]
                + g_ratio * cost_flat[nei]
                + (1.0 - g_ratio) * h_flat[nei]
            )

            fresh = (nei not in open_f) and (nei not in closed_f)
            improved = nei in open_f and open_f[nei] > f_new
            if fresh or improved:
                open_f[nei] = f_new
                parent[nei] = sel
                heapq.heappush(heap, (f_new, nei))

    result.parent = parent
    closed_mask = np.zeros(problem.num_nodes, dtype=bool)
    closed_mask[list(closed_f.keys())] = True
    result.closed = closed_mask
    if result.success:
        result.path = backtrack(parent, problem.start, problem.goal)
    return result
