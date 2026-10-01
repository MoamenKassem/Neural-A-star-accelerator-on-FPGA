"""Grid representation shared by the software reference and the future hardware model.

Everything in this project addresses grid cells by a single flat integer index::

    idx = row * width + col

That is deliberate.  On an FPGA a node identifier wants to be a small dense
integer so it can be used directly as a BRAM address, which is exactly what a
flat grid index already is.  Keeping the software reference in the same
representation means the hardware port is a transcription rather than a
redesign.

Conventions inherited from the Neural A* datasets (omron-sinicx/planning-datasets):

* ``obstacle_map`` is 1.0 for *passable* cells and 0.0 for walls.  Note the
  polarity: 1 means a cell *is* traversable.
* Connectivity is Moore (8-neighbour), matching the ``--mechanism moore`` flag
  used to generate every dataset used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (d_row, d_col) for the 8 Moore neighbours, ordered so that the four
# straight moves come first.  The order only affects tie-breaking between
# equal-f nodes, but it must match between software and hardware or the two
# will explore in different orders and produce different (equally valid) paths.
MOORE_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # N
    (0, +1),  # E
    (0, -1),  # W
    (+1, 0),  # S
    (-1, +1),  # NE
    (-1, -1),  # NW
    (+1, +1),  # SE
    (+1, -1),  # SW
)


@dataclass(frozen=True)
class GridProblem:
    """A single shortest-path problem instance.

    Attributes:
        obstacle_map: ``(H, W)`` float array, 1.0 = passable, 0.0 = wall.
        start: flat index of the start cell.
        goal: flat index of the goal cell.
        opt_dist: optional ``(H, W)`` array of true shortest distances to the
            goal, as shipped with the datasets.  When present it lets us score
            true path optimality instead of only comparing against vanilla A*.
    """

    obstacle_map: np.ndarray
    start: int
    goal: int
    opt_dist: np.ndarray | None = None

    @property
    def height(self) -> int:
        return int(self.obstacle_map.shape[0])

    @property
    def width(self) -> int:
        return int(self.obstacle_map.shape[1])

    @property
    def num_nodes(self) -> int:
        return self.height * self.width

    def to_rc(self, idx: int) -> tuple[int, int]:
        """Flat index -> (row, col)."""
        return divmod(int(idx), self.width)

    def to_idx(self, row: int, col: int) -> int:
        """(row, col) -> flat index."""
        return int(row) * self.width + int(col)

    def optimal_cost(self) -> float | None:
        """True shortest-path cost from start to goal, if the dataset gave us one."""
        if self.opt_dist is None:
            return None
        return float(self.opt_dist.flatten()[self.start])


def build_neighbour_table(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the 8 neighbours of every cell.

    Returns:
        neighbours: ``(H*W, 8)`` int32 array of flat neighbour indices.  Entries
            that would fall outside the grid are set to -1.
        valid: ``(H*W, 8)`` bool array, True where the neighbour is on the grid.

    Why precompute?  Because this table is what the hardware will hold too:
    border handling becomes a lookup instead of four comparisons per neighbour
    per expansion.  Building it once here also keeps the search loop itself
    free of index arithmetic, which makes the loop far easier to read and to
    compare against an HLS implementation.
    """
    rows, cols = np.divmod(np.arange(height * width), width)
    neighbours = np.full((height * width, len(MOORE_OFFSETS)), -1, dtype=np.int32)
    valid = np.zeros((height * width, len(MOORE_OFFSETS)), dtype=bool)

    for k, (d_row, d_col) in enumerate(MOORE_OFFSETS):
        nr, nc = rows + d_row, cols + d_col
        on_grid = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        neighbours[on_grid, k] = (nr[on_grid] * width + nc[on_grid]).astype(np.int32)
        valid[:, k] = on_grid

    return neighbours, valid


def one_hot_to_index(one_hot: np.ndarray) -> int:
    """Convert a one-hot ``(H, W)`` map to a flat index.

    The datasets and the upstream model both pass start/goal around as one-hot
    maps; internally the integer form is preferred.
    """
    flat = np.asarray(one_hot).flatten()
    nonzero = np.flatnonzero(flat)
    if len(nonzero) != 1:
        raise ValueError(f"expected exactly one hot cell, found {len(nonzero)}")
    return int(nonzero[0])


def index_to_one_hot(idx: int, height: int, width: int) -> np.ndarray:
    """Inverse of :func:`one_hot_to_index`."""
    out = np.zeros(height * width, dtype=np.float32)
    out[int(idx)] = 1.0
    return out.reshape(height, width)
