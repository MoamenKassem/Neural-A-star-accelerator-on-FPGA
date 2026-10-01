"""Fixed-point emulation, for deciding how many bits the hardware search needs.

An FPGA will not carry float64.  Before writing any HLS it is necessary to know the
narrowest fixed-point format that still reproduces the float search, because
that number drives BRAM usage, comparator width and achievable clock.

What is at risk
---------------
Two things, and they fail differently:

* **The heuristic tie-break.**  ``h`` is Chebyshev (an integer) plus
  ``0.001 * euclidean``.  Quantise too coarsely and that second term rounds to
  zero, every plateau of equal-h cells becomes a tie again, and the search fans
  out.  The failure shows up as *more expansions*, not a wrong answer, so a
  correctness-only test will not catch it.
* **The accumulated cost.**  ``g`` sums sigmoid outputs in (0, 1) over the path.
  Rounding error accumulates along the path, so a coarse format can reorder two
  nearly-equal frontier nodes and return a slightly different path.

Measure both.  A format that keeps path cost identical but doubles expansions
has quietly thrown away the entire benefit of Neural A*.

Format
------
``Q(integer_bits).(fractional_bits)`` unsigned.  :func:`suggest_format` reads
the integer width off measured data rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import GridProblem
from .heuristic import DEFAULT_TB_FACTOR
from .reference_astar import DEFAULT_G_RATIO, SearchResult, astar_search


def quantise(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    """Round to the nearest multiple of ``2**-fractional_bits``.

    Round-to-nearest matches what an HLS ``ap_fixed`` with ``AP_RND`` does.
    Truncation (the ``ap_fixed`` default) biases every value downward and makes
    the tie-break term vanish sooner, so if the hardware ends up truncating,
    re-run this analysis with ``np.floor``.
    """
    scale = float(2**fractional_bits)
    return np.round(np.asarray(values, dtype=np.float64) * scale) / scale


@dataclass
class PrecisionResult:
    """Outcome of running one search at a given fractional width."""

    fractional_bits: int
    success: bool
    num_expansions: int
    path_length: int
    path_cost: float
    expansion_ratio: float
    """Expansions relative to the float64 reference.  1.0 is perfect; above 1.0
    means precision loss is costing us search efficiency."""

    path_matches: bool
    """Whether the path length matches the float64 reference."""


def sweep_precision(
    problem: GridProblem,
    cost_map: np.ndarray,
    fractional_bits: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20),
    g_ratio: float = DEFAULT_G_RATIO,
    tb_factor: float = DEFAULT_TB_FACTOR,
) -> list[PrecisionResult]:
    """Run the same problem at several fractional widths and compare to float64.

    Only the *cost map* is quantised here, not the internal accumulator.  That
    makes this an optimistic bound: real hardware also rounds ``g`` at every
    addition.  Treat the width this reports as a floor, and add a couple of
    bits of margin.
    """
    reference = astar_search(problem, cost_map, g_ratio=g_ratio, tb_factor=tb_factor)
    if not reference.success:
        return []

    cost_flat = np.asarray(cost_map, dtype=np.float64).reshape(-1)
    ref_expansions = max(reference.num_expansions, 1)
    ref_length = reference.path_length()

    results: list[PrecisionResult] = []
    for bits in fractional_bits:
        quantised = quantise(cost_map, bits)
        out = astar_search(problem, quantised, g_ratio=g_ratio, tb_factor=tb_factor)
        results.append(
            PrecisionResult(
                fractional_bits=bits,
                success=out.success,
                num_expansions=out.num_expansions,
                path_length=out.path_length(),
                path_cost=out.path_cost(cost_flat) if out.success else float("inf"),
                expansion_ratio=out.num_expansions / ref_expansions,
                path_matches=out.success and out.path_length() == ref_length,
            )
        )
    return results


def suggest_format(
    max_f_observed: float,
    max_expansions: int,
    fractional_bits: int,
    safety_margin_bits: int = 2,
) -> dict[str, int]:
    """Propose a fixed-point format from measured ranges.

    Args:
        max_f_observed: largest f seen across the evaluation set.
        max_expansions: largest expansion count seen -- bounds how much
            rounding error can accumulate along one path.
        fractional_bits: the width chosen from :func:`sweep_precision`.
        safety_margin_bits: extra integer headroom.  The evaluation set is a
            sample, not a proof of the worst case, and an f overflow in
            hardware is a silent wraparound that corrupts the pop order.

    Returns:
        Dict with ``integer_bits``, ``fractional_bits``, ``total_bits`` and the
        resulting ``resolution``.
    """
    integer_bits = int(np.ceil(np.log2(max(max_f_observed, 1.0) + 1.0)))
    integer_bits += safety_margin_bits

    return {
        "integer_bits": integer_bits,
        "fractional_bits": fractional_bits,
        "total_bits": integer_bits + fractional_bits,
        "max_representable": int(2**integer_bits) - 1,
        "resolution_micro": int(round(1e6 / (2**fractional_bits))),
        "max_expansions_seen": int(max_expansions),
    }


def open_list_bits(peak_open: int, total_bits: int, num_nodes: int) -> dict[str, int]:
    """Storage for a priority queue holding ``peak_open`` entries.

    Each entry is an f value plus a node index.  This is what decides between a
    register-array PQ (fast, LUT-hungry, fine at a few hundred entries) and a
    bucket queue (cheap, needs f to be bounded and quantised -- which it is).
    """
    index_bits = int(np.ceil(np.log2(max(num_nodes, 2))))
    entry_bits = total_bits + index_bits
    return {
        "entries": int(peak_open),
        "index_bits": index_bits,
        "entry_bits": entry_bits,
        "total_bits": int(peak_open) * entry_bits,
        "total_bytes": int(np.ceil(peak_open * entry_bits / 8)),
    }
