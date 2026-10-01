"""neural_astar_core -- a readable Neural A* reference, built for a hardware port.

This package is a reorganisation of omron-sinicx/neural-astar, keeping the same
algorithm and the same trained checkpoint but restructured so that:

* the search targeted for the FPGA lives in one file
  (:mod:`neural_astar_core.reference_astar`) and reads like code one would
  transcribe into HLS C++;
* the CNN and the search are cleanly separated and independently timed, so the
  question "which stage is the bottleneck?" has a measured answer;
* problem instances are randomised under explicit seed control, so runs are
  both varied and replayable.

Quick start::

    from neural_astar_core import (
        sample_problems, load_encoder_from_checkpoint,
        VanillaAStarPlanner, NeuralAStarPlanner, evaluate,
    )

    problems = sample_problems("mazes_032_moore_c8.npz", "test", 32, seed=None)
    encoder = load_encoder_from_checkpoint("model/mazes_032_moore_c8/lightning_logs/")

    report = evaluate(problems, VanillaAStarPlanner(),
                      NeuralAStarPlanner(encoder))
    print(report.summary())

Torch is imported lazily: the grid, heuristic, search, data and metrics modules
work with numpy alone, so the golden model can run anywhere -- including inside
an HLS test harness -- without a deep learning stack.
"""

from __future__ import annotations

from .data import MazeDataset, ProblemSet, sample_problems
from .grid import (
    MOORE_OFFSETS,
    GridProblem,
    build_neighbour_table,
    index_to_one_hot,
    one_hot_to_index,
)
from .heuristic import heuristic_map
from .metrics import ComparisonReport, ComparisonRow, evaluate
from .reference_astar import (
    SearchResult,
    astar_search,
    backtrack,
    upstream_pq_astar,
)

__version__ = "0.1.0"

__all__ = [
    "GridProblem",
    "MOORE_OFFSETS",
    "build_neighbour_table",
    "one_hot_to_index",
    "index_to_one_hot",
    "heuristic_map",
    "astar_search",
    "upstream_pq_astar",
    "backtrack",
    "SearchResult",
    "MazeDataset",
    "ProblemSet",
    "sample_problems",
    "evaluate",
    "ComparisonReport",
    "ComparisonRow",
    "VanillaAStarPlanner",
    "NeuralAStarPlanner",
    "PlanResult",
    "GuidanceEncoder",
    "load_encoder_from_checkpoint",
    "encoder_input",
]


def __getattr__(name: str):
    """Lazily import the torch-dependent symbols.

    Keeps ``import neural_astar_core`` cheap and lets the numpy-only half of the
    package run in environments without torch.
    """
    if name in ("VanillaAStarPlanner", "NeuralAStarPlanner", "PlanResult"):
        from . import planner

        return getattr(planner, name)
    if name in ("GuidanceEncoder", "load_encoder_from_checkpoint", "encoder_input"):
        from . import encoder

        return getattr(encoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
