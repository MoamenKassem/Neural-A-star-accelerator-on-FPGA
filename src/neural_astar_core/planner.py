"""Planners: vanilla A* and Neural A*, both over the same reference search.

The only difference between the two is where the cost map comes from.  Vanilla
A* uses an all-ones cost map -- every passable cell costs the same to enter.
Neural A* asks the CNN for a guidance map first, then runs the *identical*
search over it.

Every plan returns a :class:`PlanResult` that keeps the encode and search times
separately.  That split is the number the whole hardware plan hinges on: if the
CNN dominates, this is a FINN project; if the search dominates, it is an HLS
project.  Guessing is not required -- the instrumentation is right here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .grid import GridProblem, build_neighbour_table, index_to_one_hot
from .heuristic import DEFAULT_TB_FACTOR
from .reference_astar import DEFAULT_G_RATIO, SearchResult, astar_search


@dataclass
class PlanResult:
    """One solved problem, with the encode/search timing split kept apart."""

    search: SearchResult
    cost_map: np.ndarray
    encode_seconds: float
    search_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.encode_seconds + self.search_seconds

    @property
    def encode_fraction(self) -> float:
        """Share of wall-clock spent in the CNN.  Near 1.0 means accelerate the CNN."""
        total = self.total_seconds
        return self.encode_seconds / total if total > 0 else 0.0


class VanillaAStarPlanner:
    """Classical A* -- uniform step cost, Chebyshev heuristic."""

    name = "Vanilla A*"

    def __init__(
        self,
        g_ratio: float = DEFAULT_G_RATIO,
        tb_factor: float = DEFAULT_TB_FACTOR,
    ):
        self.g_ratio = g_ratio
        self.tb_factor = tb_factor
        self._neighbour_cache: dict[tuple[int, int], tuple] = {}

    def _neighbour_table(self, height: int, width: int):
        key = (height, width)
        if key not in self._neighbour_cache:
            self._neighbour_cache[key] = build_neighbour_table(height, width)
        return self._neighbour_cache[key]

    def cost_map(self, problem: GridProblem) -> tuple[np.ndarray, float]:
        """All-ones cost map.  Timed for symmetry with the neural planner."""
        t0 = time.perf_counter()
        cost = np.ones((problem.height, problem.width), dtype=np.float64)
        return cost, time.perf_counter() - t0

    def plan(self, problem: GridProblem) -> PlanResult:
        cost, encode_seconds = self.cost_map(problem)

        t0 = time.perf_counter()
        result = astar_search(
            problem,
            cost,
            g_ratio=self.g_ratio,
            tb_factor=self.tb_factor,
            neighbour_table=self._neighbour_table(problem.height, problem.width),
        )
        search_seconds = time.perf_counter() - t0

        return PlanResult(result, cost, encode_seconds, search_seconds)


class NeuralAStarPlanner(VanillaAStarPlanner):
    """Neural A* -- the CNN supplies the cost map, then the same search runs.

    Args:
        encoder: a :class:`~neural_astar_core.encoder.GuidanceEncoder`.
        device: torch device for the forward pass.
        g_ratio: weight between cost-to-come and heuristic.
        tb_factor: Euclidean tie-break weight.
        sync_cuda: when running on GPU, synchronise before stopping the timer.
            Without this the encode time is measured as ~0 because the kernel
            launch returns immediately, and the timing split becomes nonsense.
    """

    name = "Neural A*"

    def __init__(
        self,
        encoder,
        device: str = "cpu",
        g_ratio: float = DEFAULT_G_RATIO,
        tb_factor: float = DEFAULT_TB_FACTOR,
        sync_cuda: bool = True,
    ):
        super().__init__(g_ratio=g_ratio, tb_factor=tb_factor)
        self.encoder = encoder
        self.device = device
        self.sync_cuda = sync_cuda

    def cost_map(self, problem: GridProblem) -> tuple[np.ndarray, float]:
        import torch

        from .encoder import encoder_input

        start_map = index_to_one_hot(problem.start, problem.height, problem.width)
        goal_map = index_to_one_hot(problem.goal, problem.height, problem.width)

        t0 = time.perf_counter()
        with torch.no_grad():
            x = encoder_input(problem.obstacle_map, start_map, goal_map)
            x = x.to(self.device)
            y = self.encoder(x)
            if self.sync_cuda and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            cost = y.squeeze().detach().cpu().numpy().astype(np.float64)
        encode_seconds = time.perf_counter() - t0

        return cost, encode_seconds

    def encode_batch(self, problems: list[GridProblem]) -> tuple[np.ndarray, float]:
        """Encode many problems in one forward pass.

        Batching is how a GPU looks good.  It is also unrepresentative of the
        deployment case of interest -- a robot plans one path at a time -- so
        report batched and unbatched numbers separately rather than picking
        whichever flatters the conclusion.
        """
        import torch

        from .encoder import encoder_input

        tensors = []
        for p in problems:
            start_map = index_to_one_hot(p.start, p.height, p.width)
            goal_map = index_to_one_hot(p.goal, p.height, p.width)
            tensors.append(encoder_input(p.obstacle_map, start_map, goal_map))
        batch = torch.cat(tensors, dim=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            batch = batch.to(self.device)
            out = self.encoder(batch)
            if self.sync_cuda and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            cost = out.squeeze(1).detach().cpu().numpy().astype(np.float64)
        return cost, time.perf_counter() - t0
