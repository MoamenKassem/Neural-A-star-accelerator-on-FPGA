# Neural A* on FPGA — source

Moamen Kassem, 322672973
02360509 Advanced Topics in Hardware Accelerators for Deep Learning

Accompanies `report.pdf`. Section numbers below refer to it.

## What is mine and what is not

**Written by me**

- `neural_astar_core/` — the planning library: flat-index grid and Moore
  neighbour tables, Chebyshev heuristic, priority-queue A*, dataset loading,
  the paper's evaluation protocol, the p_opt / p_exp / p_gen metrics, and the
  fixed-point precision analysis (report Section 3).
- All five notebooks, including the FINN build flow, the folding planner, and
  the board-side driver code (Sections 7-10).
- `sweep.py` — the architecture and precision sweep behind Table 5.

**Reused**

- The Neural A* reference implementation (github.com/omron-sinicx/neural-astar)
  for the trained encoder checkpoint and as the behavioural reference my A* was
  validated against. `neural_astar_core/reference_astar.py` contains
  `upstream_pq_astar()`, a transcription of their search kept only as the thing
  to test against; everything else in that file is my own.
- The planning-datasets repository for the maze dataset.
- FINN and Brevitas as the quantization and synthesis toolchain. The build
  flow, the custom `step_name_nodes` step and the folding configuration are
  mine; the compiler is theirs.

## Layout

```
neural_astar_core/      planning library (numpy only, torch imported lazily)
01_baseline.ipynb       profiling, paper-protocol evaluation, metric analysis   (Sections 4, 5)
02_quantize.ipynb       precision sweep on the published encoder                (Section 7)
03_synthesis.ipynb      building the published encoder; fails DRC               (Section 8)
04_fast_track.ipynb     first working flow; produced build A                    (Section 10.1)
04b_build.ipynb         parameterised flow; produced builds B, C, D             (Sections 9, 10)
variant_d_cells.py      cells appended to 04b for build C (sigmoid on the host)
variant_e_cells.py      cells appended to 04b for build D (clipped targets)
05_board.ipynb          runs on the PYNQ-Z2: verification, latency, planning    (Section 10)
run_build.py            standalone FINN build, for running under sbatch
submit_build.sh         sbatch wrapper for long synthesis jobs
diagnose_build.py       extracts Vivado errors and utilisation from a failed build
sweep.py                architecture x precision sweep                          (Table 5)
sweep_results.json      its output, used for Table 5 and Figure 2
fetch_data.py           clones planning-datasets and copies the .npz files
test_reference.py       validates my A* against the reference implementation
```

Datasets, checkpoints, ONNX files and bitstreams are excluded from this archive
as instructed. `fetch_data.py` retrieves the dataset; the encoder checkpoint
comes from the reference repository.

## Reproducing

1. `python fetch_data.py` — dataset into the working directory.
2. `01_baseline.ipynb` — the profiling that redirected the project.
3. `02_quantize.ipynb` — W4A8 selection.
4. `04b_build.ipynb` — set `CHANNELS`, `MAC_BUDGET` and `TAG` in the first cell,
   then run. Produces a bitstream and deployment package. Needs FINN and Vivado.
5. Copy `tiny_output<TAG>/deploy`, the test vectors, the dataset and
   `neural_astar_core/` to the board, then run `05_board.ipynb` there.

The four builds reported in Section 10.1 differ only in the first cell of
`04b_build.ipynb`: build A is `MAC_BUDGET = 34`, build B is `54`, build C adds
the host-side sigmoid, and build D clips the distillation targets.
