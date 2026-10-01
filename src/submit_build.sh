#!/bin/bash
# Submit the FINN bitfile build as a Slurm batch job.
#
#   sbatch submit_build.sh
#
# Why batch and not `srun --pty`: an interactive allocation dies when its time
# limit expires and takes every process with it, tmux included. A batch job runs
# detached from the session entirely -- log out, drop the VPN, close the
# laptop; it keeps going until it finishes or hits --time.
#
# Watch it:
#   squeue -u $USER                  # queued / running
#   tail -f finn_astar_<jobid>.log   # live output (streams; see PYTHONUNBUFFERED)
#   grep "Running step" finn_astar_<jobid>.log | tail -3
#   scancel <jobid>                  # stop it
#
# Two things that matter here:
#
# --cpus-per-task=8   Vivado synthesis and place-and-route are multithreaded.
#                     An `srun` without this flag gets ONE core, which is why a
#                     build that should take ~90 minutes ran for over three
#                     hours. Raise to 16 if the queue allows.
#
# no --gres=gpu       Vivado and Vitis HLS never touch a GPU. The GPU was only
#                     needed for 02_quantize.ipynb; requesting one here just
#                     means a longer queue.
#
# --time=12:00:00     Generous on purpose. Unused time is not charged, so
#                     a job killed at 99% loses everything.
#                     Check the partition ceiling first:  sinfo -o "%P %l"

#SBATCH --job-name=finn_astar
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.log

set -euo pipefail

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "time limit: $(squeue -j "$SLURM_JOB_ID" -h -o %l 2>/dev/null || echo '?')"
echo

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fpga_hw2

# Vivado wants libtinfo.so.5; conda ships .so.6. Expose just that symlink
# rather than the whole conda lib/, which would shadow bash's libtinfo and
# flood the log with "no version information available".
mkdir -p "$HOME/vivado_libs"
ln -sf "$CONDA_PREFIX/lib/libtinfo.so.6" "$HOME/vivado_libs/libtinfo.so.5"
export LD_LIBRARY_PATH="$HOME/vivado_libs:${LD_LIBRARY_PATH:-}"

source /opt/Xilinx/Vivado/2022.1/settings64.sh
source /opt/Xilinx/Vitis_HLS/2022.1/settings64.sh

cd "$HOME/fpga_hw2"

# Vivado spawns many subshells; give it plenty of file handles.
ulimit -n 4096 || true

# -u / PYTHONUNBUFFERED: without these, Python buffers stdout in ~8KB blocks
# when it is not attached to a terminal, so the log sits empty for many minutes
# and a running build looks hung.
export PYTHONUNBUFFERED=1
python -u run_build.py

echo
echo "finished $(date)"
