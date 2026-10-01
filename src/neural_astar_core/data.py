"""Loading the Neural A* datasets and sampling randomised problem instances.

Dataset format (from omron-sinicx/planning-datasets)
----------------------------------------------------
Each ``.npz`` holds twelve arrays -- four per split, in the order
train, valid, test::

    arr_0  arr_4  arr_8   map designs      (N, H, W)      1.0 = passable
    arr_1  arr_5  arr_9   goal maps        (N, 1, H, W)   one-hot
    arr_2  arr_6  arr_10  optimal policy   (N, 8, 1, H, W) one-hot direction
    arr_3  arr_7  arr_11  optimal dists    (N, 1, H, W)

Two conventions catch people out:

* ``map_design`` is **1 for free space**, 0 for walls.  Not the other way round.
* ``opt_dist`` is **negative**: the goal is ``-0.0`` and cells grow more
  negative with distance.  Unreachable cells (including every wall) are
  ``-1024.0``.  True optimal cost from a cell is therefore ``-opt_dist[cell]``.

Where the start comes from
--------------------------
The datasets do not ship start positions.  Upstream generates one at random
inside ``__getitem__``, so every epoch sees different problems on the same maps.
That behaviour is kept but put the randomness under explicit control, because
"the benchmark moved under us" is not a debugging experience anyone wants
halfway through a two-week project.

Pass ``seed=None`` for a fresh draw each run (for cases where each run should
be an independent sample), or an integer for a reproducible set.  Either way
:class:`ProblemSet` records the seed it actually used, so a surprising result
can always be replayed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import GridProblem, one_hot_to_index

SPLIT_OFFSETS = {"train": 0, "valid": 4, "test": 8}

# Distance percentile bands used to pick start cells, matching upstream's
# MazeDataset.  Starts are drawn from the far 55% of reachable cells so the
# problems are non-trivial -- a start next to the goal tells us nothing about
# search efficiency.
START_PERCENTILES = np.array([0.55, 0.70, 0.85, 1.0])

UNREACHABLE = -1024.0


@dataclass
class ProblemSet:
    """A concrete, replayable collection of problem instances."""

    problems: list[GridProblem]
    map_indices: list[int]
    """Which dataset map each problem came from, so results can be traced back."""

    seed: int
    """The seed actually used.  Always populated, even when the caller passed None."""

    split: str
    source: str

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self):
        return iter(self.problems)

    def __getitem__(self, i: int) -> GridProblem:
        return self.problems[i]

    def describe(self) -> str:
        if not self.problems:
            return "empty problem set"
        p = self.problems[0]
        return (
            f"{len(self.problems)} problems | {p.height}x{p.width} grid | "
            f"split={self.split} | seed={self.seed} | source={self.source}"
        )


class MazeDataset:
    """Thin reader over one split of a planning-datasets ``.npz``."""

    def __init__(self, path: str, split: str = "test"):
        if split not in SPLIT_OFFSETS:
            raise ValueError(f"split must be one of {sorted(SPLIT_OFFSETS)}")
        self.path = str(path)
        self.split = split

        offset = SPLIT_OFFSETS[split]
        with np.load(self.path) as f:
            self.map_designs = f[f"arr_{offset}"].astype(np.float32)
            self.goal_maps = f[f"arr_{offset + 1}"].astype(np.float32)
            self.opt_dists = f[f"arr_{offset + 3}"].astype(np.float32)

        self.num_maps = int(self.map_designs.shape[0])
        self.height = int(self.map_designs.shape[-2])
        self.width = int(self.map_designs.shape[-1])

    def __len__(self) -> int:
        return self.num_maps

    def goal_index(self, i: int) -> int:
        return one_hot_to_index(self.goal_maps[i][0])

    def sample_start(self, i: int, rng: np.random.Generator) -> int:
        """Draw a start cell for map ``i`` from the far-distance percentile bands.

        Reproduces upstream's sampling: pick one of three distance bands
        uniformly, then a uniform cell inside it.  Cells that cannot reach the
        goal are excluded.
        """
        dist_flat = self.opt_dists[i][0].flatten()
        reachable = dist_flat[dist_flat > dist_flat.min()]
        if reachable.size == 0:
            raise ValueError(f"map {i} has no reachable cells")

        thresholds = np.percentile(reachable, 100.0 * (1.0 - START_PERCENTILES))
        band = int(rng.integers(0, len(thresholds) - 1))
        candidates = np.flatnonzero(
            (dist_flat >= thresholds[band + 1]) & (dist_flat <= thresholds[band])
        )
        if candidates.size == 0:
            # Degenerate map (very few reachable cells): fall back to any
            # reachable cell that is not the goal.
            candidates = np.flatnonzero(dist_flat > dist_flat.min())
        return int(rng.choice(candidates))

    def build_problem(self, i: int, rng: np.random.Generator) -> GridProblem:
        return GridProblem(
            obstacle_map=self.map_designs[i].astype(np.float64),
            start=self.sample_start(i, rng),
            goal=self.goal_index(i),
            opt_dist=-self.opt_dists[i][0].astype(np.float64),  # flip to positive
        )


def sample_problems(
    path: str,
    split: str = "test",
    num_problems: int = 32,
    seed: int | None = None,
    starts_per_map: int = 1,
    replace: bool = False,
) -> ProblemSet:
    """Draw a randomised set of problem instances.

    Args:
        path: path to the ``.npz`` dataset.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        num_problems: how many instances to produce.
        seed: ``None`` for a different draw every run (the seed is still
            recorded so the draw can be replayed); an integer for reproducibility.
        starts_per_map: draw this many distinct starts from each selected map.
            Useful for isolating "does the guidance map help on *this* map"
            from map-to-map variance.
        replace: allow the same map to be selected more than once.  Required if
            ``num_problems`` exceeds the number of maps in the split.

    Returns:
        A :class:`ProblemSet` carrying the problems and the seed used.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)

    dataset = MazeDataset(path, split)
    num_maps_needed = int(np.ceil(num_problems / max(starts_per_map, 1)))

    if not replace and num_maps_needed > len(dataset):
        raise ValueError(
            f"asked for {num_problems} problems at {starts_per_map} start(s) per map, "
            f"which needs {num_maps_needed} maps, but split '{split}' has only "
            f"{len(dataset)}. Pass replace=True or lower num_problems."
        )

    chosen = rng.choice(len(dataset), size=num_maps_needed, replace=replace)

    problems: list[GridProblem] = []
    map_indices: list[int] = []
    for map_idx in chosen:
        for _ in range(starts_per_map):
            if len(problems) >= num_problems:
                break
            problems.append(dataset.build_problem(int(map_idx), rng))
            map_indices.append(int(map_idx))

    return ProblemSet(
        problems=problems,
        map_indices=map_indices,
        seed=seed,
        split=split,
        source=path,
    )
