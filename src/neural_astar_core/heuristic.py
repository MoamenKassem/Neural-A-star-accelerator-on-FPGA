"""The A* heuristic used by Neural A*.

Neural A* uses Chebyshev distance plus a very small multiple of Euclidean
distance::

    h(v) = chebyshev(v, goal) + tb_factor * euclidean(v, goal)

Chebyshev is the exact cost-to-go on an 8-connected grid with unit move cost,
so on its own it is admissible and consistent.  The Euclidean term is a
*tie-breaker*: without it, huge plateaus of cells share the same h, the open
list fills with ties, and the search fans out in a diamond instead of driving
toward the goal.  ``tb_factor = 0.001`` is small enough not to break
admissibility in practice while collapsing most of those ties.

Hardware note
-------------
That 0.001 factor is the reason fixed-point width matters here.  If quantised
f too coarsely the tie-break term rounds away entirely and the hardware search
explores a visibly different (larger) region than the float reference, even
though both return valid paths.  ``scripts``-level experiments should sweep the
fractional bit width and watch expansion counts, not just path cost.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TB_FACTOR = 0.001


def heuristic_map(
    goal: int,
    height: int,
    width: int,
    tb_factor: float = DEFAULT_TB_FACTOR,
) -> np.ndarray:
    """Compute h(v) for every cell, as a flat ``(H*W,)`` float64 array.

    Args:
        goal: flat index of the goal cell.
        height: grid height.
        width: grid width.
        tb_factor: weight of the Euclidean tie-break term.

    Returns:
        Flat array of heuristic values, indexed by flat cell index.

    Computing the whole map up front costs one pass over the grid and turns
    every later ``h(v)`` into an array read.  The hardware will do the same
    thing -- precompute h into BRAM while the CNN is still producing the
    guidance map -- so the reference mirrors the intended dataflow.
    """
    rows, cols = np.divmod(np.arange(height * width), width)
    goal_row, goal_col = divmod(int(goal), width)

    d_row = np.abs(rows - goal_row).astype(np.float64)
    d_col = np.abs(cols - goal_col).astype(np.float64)

    # Chebyshev = max(|dx|, |dy|), written as sum - min to match upstream's
    # formulation exactly (identical value, but keeps the diff reviewable).
    chebyshev = (d_row + d_col) - np.minimum(d_row, d_col)
    euclidean = np.sqrt(d_row**2 + d_col**2)

    return chebyshev + tb_factor * euclidean
