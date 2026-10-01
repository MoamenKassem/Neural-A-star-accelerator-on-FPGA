"""Metrics for comparing planners, plus the statistics that size the hardware.

Paper metrics
-------------
The Neural A* paper reports three numbers, reproduced here so the results
are directly comparable:

``p_opt``
    Fraction of problems where the returned path is optimal.  Upstream
    approximates this by checking whether the path has the same *length* as
    vanilla A*'s.  The datasets ship true optimal distances, so this module also computes
    the exact version -- and the two do disagree, which is worth knowing before
    quoting a number in a report.

``p_exp``
    Reduction in node expansions relative to vanilla A*, clipped at zero:
    ``max((exp_vanilla - exp_neural) / exp_vanilla, 0)``.  This is what Neural
    A* is actually trained to improve.

``h_mean``
    Harmonic mean of ``p_opt`` and ``p_exp``.  Harmonic rather than arithmetic
    because a planner that is fast but wrong, or correct but slow, should not
    score well.

Hardware sizing statistics
--------------------------
:func:`hardware_stats` reports the numbers that decide the architecture --
peak open-list occupancy (which sizes the priority queue) and the observed
f range (which sizes the fixed-point format).  These are not in the paper; they
need them and nobody else collected them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import GridProblem
from .planner import PlanResult


@dataclass
class ComparisonRow:
    """One problem, solved by both planners."""

    map_index: int
    solved_baseline: bool
    solved_neural: bool
    expansions_baseline: int
    expansions_neural: int
    path_length_baseline: int
    path_length_neural: int
    true_optimal_cost: float | None
    encode_seconds: float
    search_seconds: float
    baseline_search_seconds: float
    peak_open_baseline: int
    peak_open_neural: int
    max_f_neural: float


@dataclass
class ComparisonReport:
    """Aggregate of many :class:`ComparisonRow` s."""

    rows: list[ComparisonRow] = field(default_factory=list)

    def _solved(self) -> list[ComparisonRow]:
        """Rows where both planners succeeded -- the only fair comparison set."""
        return [r for r in self.rows if r.solved_baseline and r.solved_neural]

    @property
    def num_problems(self) -> int:
        return len(self.rows)

    @property
    def p_opt(self) -> float:
        """Fraction of Neural A* paths matching vanilla A*'s path length."""
        solved = self._solved()
        if not solved:
            return 0.0
        return float(
            np.mean(
                [r.path_length_neural == r.path_length_baseline for r in solved]
            )
        )

    @property
    def p_exp(self) -> float:
        """Mean reduction in expansions versus vanilla A*, clipped at zero."""
        solved = self._solved()
        if not solved:
            return 0.0
        reductions = [
            max(
                (r.expansions_baseline - r.expansions_neural)
                / max(r.expansions_baseline, 1),
                0.0,
            )
            for r in solved
        ]
        return float(np.mean(reductions))

    @property
    def h_mean(self) -> float:
        """Harmonic mean of p_opt and p_exp."""
        p_o, p_e = self.p_opt, self.p_exp
        if p_o <= 0 or p_e <= 0:
            return 0.0
        return float(2.0 / (1.0 / p_o + 1.0 / p_e))

    @property
    def success_rate_neural(self) -> float:
        if not self.rows:
            return 0.0
        return float(np.mean([r.solved_neural for r in self.rows]))

    @property
    def success_rate_baseline(self) -> float:
        if not self.rows:
            return 0.0
        return float(np.mean([r.solved_baseline for r in self.rows]))

    def timing_split(self) -> dict[str, float]:
        """Mean per-problem timing, and the CNN's share of Neural A* wall-clock.

        ``encode_fraction`` is the headline: it says which stage to accelerate.
        """
        if not self.rows:
            return {}
        encode = float(np.mean([r.encode_seconds for r in self.rows]))
        search = float(np.mean([r.search_seconds for r in self.rows]))
        baseline = float(np.mean([r.baseline_search_seconds for r in self.rows]))
        total = encode + search
        return {
            "encode_seconds": encode,
            "search_seconds": search,
            "total_seconds": total,
            "encode_fraction": encode / total if total > 0 else 0.0,
            "baseline_search_seconds": baseline,
            "search_speedup_vs_vanilla": baseline / search if search > 0 else 0.0,
        }

    def hardware_stats(self) -> dict[str, float]:
        """Numbers that constrain the FPGA design.

        ``peak_open_*`` sizes the priority queue: it is the maximum number of
        entries that must be simultaneously resident and sortable.
        ``max_f`` sizes the integer part of the fixed-point format.
        """
        if not self.rows:
            return {}
        return {
            "peak_open_neural_mean": float(
                np.mean([r.peak_open_neural for r in self.rows])
            ),
            "peak_open_neural_max": float(
                np.max([r.peak_open_neural for r in self.rows])
            ),
            "peak_open_baseline_max": float(
                np.max([r.peak_open_baseline for r in self.rows])
            ),
            "expansions_neural_max": float(
                np.max([r.expansions_neural for r in self.rows])
            ),
            "max_f_observed": float(np.max([r.max_f_neural for r in self.rows])),
        }

    def summary(self) -> str:
        t = self.timing_split()
        hw = self.hardware_stats()
        lines = [
            f"problems evaluated      : {self.num_problems}",
            f"success (vanilla / NA*) : {self.success_rate_baseline:.1%} / "
            f"{self.success_rate_neural:.1%}",
            f"p_opt                   : {self.p_opt:.3f}",
            f"p_exp                   : {self.p_exp:.3f}",
            f"h_mean                  : {self.h_mean:.3f}",
            "",
            f"mean encode time        : {t.get('encode_seconds', 0)*1e3:.2f} ms",
            f"mean search time        : {t.get('search_seconds', 0)*1e3:.2f} ms",
            f"CNN share of runtime    : {t.get('encode_fraction', 0):.1%}",
            "",
            f"peak open list (max)    : {hw.get('peak_open_neural_max', 0):.0f} entries",
            f"max f observed          : {hw.get('max_f_observed', 0):.2f}",
        ]
        return "\n".join(lines)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Column-wise view, convenient for plotting."""
        return {
            "expansions_baseline": np.array(
                [r.expansions_baseline for r in self.rows]
            ),
            "expansions_neural": np.array([r.expansions_neural for r in self.rows]),
            "path_length_baseline": np.array(
                [r.path_length_baseline for r in self.rows]
            ),
            "path_length_neural": np.array([r.path_length_neural for r in self.rows]),
            "peak_open_neural": np.array([r.peak_open_neural for r in self.rows]),
            "encode_seconds": np.array([r.encode_seconds for r in self.rows]),
            "search_seconds": np.array([r.search_seconds for r in self.rows]),
        }


def make_row(
    problem: GridProblem,
    map_index: int,
    baseline: PlanResult,
    neural: PlanResult,
) -> ComparisonRow:
    """Build one comparison row from a matched pair of results."""
    return ComparisonRow(
        map_index=map_index,
        solved_baseline=baseline.search.success,
        solved_neural=neural.search.success,
        expansions_baseline=baseline.search.num_expansions,
        expansions_neural=neural.search.num_expansions,
        path_length_baseline=baseline.search.path_length(),
        path_length_neural=neural.search.path_length(),
        true_optimal_cost=problem.optimal_cost(),
        encode_seconds=neural.encode_seconds,
        search_seconds=neural.search_seconds,
        baseline_search_seconds=baseline.search_seconds,
        peak_open_baseline=baseline.search.peak_open_size,
        peak_open_neural=neural.search.peak_open_size,
        max_f_neural=neural.search.max_f,
    )


def evaluate(
    problem_set,
    baseline_planner,
    neural_planner,
) -> ComparisonReport:
    """Run both planners over a problem set and collect the comparison.

    Args:
        problem_set: a :class:`~neural_astar_core.data.ProblemSet`.
        baseline_planner: usually a ``VanillaAStarPlanner``.
        neural_planner: usually a ``NeuralAStarPlanner``.

    Returns:
        A :class:`ComparisonReport`.
    """
    report = ComparisonReport()
    for problem, map_index in zip(problem_set.problems, problem_set.map_indices):
        baseline = baseline_planner.plan(problem)
        neural = neural_planner.plan(problem)
        report.rows.append(make_row(problem, map_index, baseline, neural))
    return report
