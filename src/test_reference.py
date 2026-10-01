"""Tests for the reference search.

Run with ``pytest test_reference.py -v``.

The point of these is not general code hygiene. It is that
``reference_astar.astar_search`` is the specification the FPGA implementation
will be checked against, so it has to keep matching upstream even as the code is refactored
it for readability. If these fail, the hardware is being verified against the
wrong thing.

Only the last two tests need torch; the rest run on numpy alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_astar_core import (
    GridProblem,
    astar_search,
    build_neighbour_table,
    heuristic_map,
    sample_problems,
    upstream_pq_astar,
)
from neural_astar_core.fixedpoint import quantise, sweep_precision

DATA_32 = "mazes_032_moore_c8.npz"
CKPT_DIR = "model/mazes_032_moore_c8/lightning_logs/"

# Roughly half these tests need the dataset and a couple need the checkpoint.
# Neither is in the repo -- they are fetched separately -- so a plain
# FileNotFoundError here would look like a code bug when it is really a setup
# step that has not been run yet. Skip with an actionable message instead.
needs_data = pytest.mark.skipif(
    not Path(DATA_32).exists(),
    reason=f"{DATA_32} not found -- run `python fetch_data.py` first",
)

needs_checkpoint = pytest.mark.skipif(
    not list(Path(CKPT_DIR).glob("**/*.ckpt")) if Path(CKPT_DIR).exists() else True,
    reason=(
        f"no .ckpt under {CKPT_DIR} -- copy model/ from the neural-astar repo "
        "into this directory"
    ),
)


# --------------------------------------------------------------------------
# Grid mechanics
# --------------------------------------------------------------------------

def test_neighbour_table_corner_and_interior():
    neighbours, valid = build_neighbour_table(4, 4)
    assert valid[0].sum() == 3, "top-left corner has 3 Moore neighbours"
    assert valid[5].sum() == 8, "interior cell has 8"
    assert valid[15].sum() == 3, "bottom-right corner has 3"
    assert (neighbours[~valid] == -1).all(), "off-grid entries must be -1"


def test_neighbours_are_symmetric():
    """If b is a neighbour of a, then a is a neighbour of b."""
    neighbours, valid = build_neighbour_table(5, 6)
    for a in range(30):
        for k in range(8):
            if not valid[a, k]:
                continue
            b = neighbours[a, k]
            assert a in neighbours[b][valid[b]], f"{a}->{b} not reciprocated"


def test_heuristic_is_zero_at_goal_and_admissible():
    h = heuristic_map(goal=0, height=8, width=8)
    assert h[0] == 0.0
    # Chebyshev is the exact cost-to-go under unit-cost 8-connected moves, so h
    # may exceed it only by the tie-break term.
    for idx in range(64):
        row, col = divmod(idx, 8)
        assert h[idx] >= max(row, col) - 1e-9
        assert h[idx] <= max(row, col) + 0.001 * np.hypot(row, col) + 1e-9


# --------------------------------------------------------------------------
# Search correctness on hand-built cases
# --------------------------------------------------------------------------

def test_straight_line_on_empty_grid():
    problem = GridProblem(np.ones((5, 5)), start=0, goal=24)
    result = astar_search(problem, np.ones((5, 5)))
    assert result.success
    # 8-connected, so the diagonal is 4 moves = 5 cells
    assert result.path_length() == 5
    assert result.path[0] == 0 and result.path[-1] == 24


def test_wall_forces_detour():
    obstacles = np.ones((5, 5))
    obstacles[1:4, 2] = 0.0  # vertical wall with gaps top and bottom
    problem = GridProblem(obstacles, start=10, goal=14)
    result = astar_search(problem, np.ones((5, 5)))
    assert result.success
    for idx in result.path:
        assert obstacles.flatten()[idx] == 1.0, "path went through a wall"


def test_unreachable_goal_reports_failure():
    obstacles = np.ones((5, 5))
    obstacles[:, 2] = 0.0  # full wall, no gap
    problem = GridProblem(obstacles, start=0, goal=4)
    result = astar_search(problem, np.ones((5, 5)))
    assert not result.success
    assert result.path == []


def test_guidance_map_picks_the_cheaper_of_two_equal_routes():
    """The whole premise of Neural A*: cost steers the path.

    Start and goal are on the middle row with the centre cell blocked, so the
    detour above and the detour below are the same length. Only the cost map
    can break the tie. If this fails, the guidance map is not reaching the
    search and every downstream result is meaningless.
    """
    obstacles = np.ones((5, 5))
    obstacles[2, 2] = 0.0
    cost = np.ones((5, 5))
    cost[3, :] = 0.1  # make the lower detour cheap

    problem = GridProblem(obstacles, start=10, goal=14)  # (2,0) -> (2,4)
    result = astar_search(problem, cost)

    rows = [idx // 5 for idx in result.path]
    assert result.success
    assert 3 in rows, "took the expensive detour despite the cheap one"
    assert 1 not in rows, "wandered into the expensive detour"


# --------------------------------------------------------------------------
# Equivalence with upstream — the important one
# --------------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_upstream_pq_astar(seed):
    problems = sample_problems(DATA_32, "test", 15, seed=seed)
    ones = np.ones((32, 32))
    for p in problems:
        mine = astar_search(p, ones)
        theirs = upstream_pq_astar(p, ones)
        assert mine.success == theirs.success
        assert mine.path_length() == theirs.path_length()
        assert mine.num_expansions == theirs.num_expansions


@needs_data
def test_vanilla_astar_finds_true_optimum():
    """Cross-check against the dataset's own shortest distances, not just upstream."""
    problems = sample_problems(DATA_32, "test", 25, seed=11)
    ones = np.ones((32, 32))
    for p in problems:
        result = astar_search(p, ones)
        assert result.success
        assert abs(result.path_cost(ones.ravel()) - p.optimal_cost()) < 1e-9


@needs_data
def test_seed_controls_reproducibility():
    a = sample_problems(DATA_32, "test", 10, seed=99)
    b = sample_problems(DATA_32, "test", 10, seed=99)
    assert [p.start for p in a] == [p.start for p in b]
    assert a.map_indices == b.map_indices

    c = sample_problems(DATA_32, "test", 10, seed=100)
    assert [p.start for p in a] != [p.start for p in c]


@needs_data
def test_unseeded_sampling_records_its_seed():
    """seed=None must still be replayable after the fact."""
    a = sample_problems(DATA_32, "test", 8, seed=None)
    replay = sample_problems(DATA_32, "test", 8, seed=a.seed)
    assert [p.start for p in a] == [p.start for p in replay]


# --------------------------------------------------------------------------
# Hardware-facing properties
# --------------------------------------------------------------------------

@needs_data
def test_peak_open_size_is_tracked_and_bounded():
    problems = sample_problems(DATA_32, "test", 20, seed=5)
    ones = np.ones((32, 32))
    for p in problems:
        result = astar_search(p, ones)
        assert 0 < result.peak_open_size <= p.num_nodes


def test_quantise_is_exact_on_representable_values():
    values = np.array([0.5, 0.25, 0.125, 1.0])
    assert np.allclose(quantise(values, 3), values)


@needs_data
def test_ten_fractional_bits_preserve_the_search():
    """The precision claim the HLS design will rely on."""
    problems = sample_problems(DATA_32, "test", 10, seed=8)
    rng = np.random.default_rng(0)
    for p in problems:
        cost = rng.uniform(0.05, 1.0, size=(32, 32))
        results = {r.fractional_bits: r for r in sweep_precision(p, cost, (10, 16))}
        assert results[10].path_matches
        assert results[10].expansion_ratio == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------
# Torch-dependent
# --------------------------------------------------------------------------

@needs_data
@needs_checkpoint
def test_checkpoint_loads_with_expected_shape():
    torch = pytest.importorskip("torch")
    from neural_astar_core import load_encoder_from_checkpoint

    encoder = load_encoder_from_checkpoint(CKPT_DIR)
    assert encoder.num_parameters() == 391395
    out = encoder(torch.zeros(1, 2, 32, 32))
    assert out.shape == (1, 1, 32, 32)
    assert (out >= 0).all() and (out <= 1).all(), "sigmoid output must be in (0,1)"


@needs_data
@needs_checkpoint
def test_neural_astar_reduces_expansions_on_average():
    pytest.importorskip("torch")
    from neural_astar_core import (
        NeuralAStarPlanner,
        VanillaAStarPlanner,
        evaluate,
        load_encoder_from_checkpoint,
    )

    encoder = load_encoder_from_checkpoint(CKPT_DIR)
    problems = sample_problems(DATA_32, "test", 30, seed=42)
    report = evaluate(problems, VanillaAStarPlanner(), NeuralAStarPlanner(encoder))

    assert report.success_rate_neural == 1.0
    assert report.p_exp > 0.2, f"p_exp collapsed to {report.p_exp:.3f}"
    assert report.p_opt > 0.5, f"p_opt collapsed to {report.p_opt:.3f}"


@needs_data
def test_generated_is_superset_of_closed():
    """Every expanded node must have been generated first.

    This also guards the instrumentation itself: if `generated` ever stops
    being recorded, the visualisation silently goes back to looking like A*
    never explores anything.
    """
    problems = sample_problems(DATA_32, "test", 15, seed=17)
    ones = np.ones((32, 32))
    for p in problems:
        r = astar_search(p, ones)
        assert r.generated is not None
        assert (r.closed <= r.generated).all(), "closed node was never generated"
        assert r.num_generated >= r.num_expansions
        assert set(r.path) <= set(np.flatnonzero(r.closed))


@needs_data
def test_open_map_can_expand_exactly_the_path():
    """On an obstacle-free grid Chebyshev is exact, so A* should not detour.

    Documents the behaviour that looks like a bug in the plots: expanded ==
    path is correct here. But `generated` must still be strictly larger,
    because the neighbours were all evaluated.
    """
    problem = GridProblem(np.ones((15, 15)), start=0, goal=224)
    r = astar_search(problem, np.ones((15, 15)))
    assert r.success
    assert r.num_expansions == r.path_length(), "perfect heuristic should not detour"
    assert r.num_generated > r.num_expansions, "neighbours were not generated"
