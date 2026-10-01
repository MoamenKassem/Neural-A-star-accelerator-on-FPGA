#!/usr/bin/env python3
"""Run the FINN bitfile build detached from Jupyter.

A full build takes over an hour, and driving it from a notebook cell means a
dropped VPN tunnel can take the kernel with it. This is the same flow as
section 5 of 03_synthesis.ipynb, as a plain script that can be left running.

Usage, from inside tmux on the compute server:

    conda activate fpga_hw2
    source /opt/Xilinx/Vivado/2022.1/settings64.sh
    source /opt/Xilinx/Vitis_HLS/2022.1/settings64.sh

    cd ~/fpga_hw2
    nohup python run_build.py > build.log 2>&1 &
    tail -f build.log          # Ctrl-C stops watching, not the build

Progress:   grep "Running step" build.log
Still up?   ps aux | grep -E "vivado|run_build" | grep -v grep

The log is also filtered: the libtinfo.so.6 warnings Vivado's subshells emit are
harmless but drown everything else, so they are dropped here.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import sys
import time

# Everything this script prints should reach the log immediately -- under
# sbatch or nohup, stdout is a file rather than a terminal and Python would
# otherwise buffer it in ~8KB blocks, making a live build look hung.
print = functools.partial(print, flush=True)

WORKDIR = os.path.expanduser("~/fpga_hw2")
ONNX = os.path.join(WORKDIR, "neural_astar_encoder_w4a8.onnx")
FOLDING = os.path.join(WORKDIR, "folding_astar.json")
SPECIALIZE = os.path.join(WORKDIR, "specialize_astar.json")
OUTPUT = os.path.join(WORKDIR, "finn_output_astar")
TMP = os.path.join(WORKDIR, "finn_build_astar_tmp")
CLK_NS = 10.0

os.chdir(WORKDIR)
# Vivado needs libtinfo.so.5, which the conda env provides only as .so.6.
# Prepending the whole conda lib/ would work but also shadows bash's own
# libtinfo.so.6, which is where the thousands of "no version information
# available" warnings come from. Expose a directory holding just that one
# symlink instead.
_shim = os.path.expanduser("~/vivado_libs")
os.makedirs(_shim, exist_ok=True)
_target = os.path.join(sys.prefix, "lib", "libtinfo.so.6")
_link = os.path.join(_shim, "libtinfo.so.5")
if os.path.exists(_target) and not os.path.exists(_link):
    os.symlink(_target, _link)
os.environ["LD_LIBRARY_PATH"] = _shim + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["FINN_ROOT"] = os.path.expanduser("~/finn")
os.environ["XILINX_VIVADO"] = "/opt/Xilinx/Vivado/2022.1"
os.environ["XILINX_HLS"] = "/opt/Xilinx/Vitis_HLS/2022.1"
os.environ["FINN_BUILD_DIR"] = TMP

import finn.builder.build_dataflow as build              # noqa: E402
import finn.builder.build_dataflow_config as build_cfg   # noqa: E402
from qonnx.transformation.general import (                # noqa: E402
    GiveReadableTensorNames, GiveUniqueNodeNames)


def step_name_nodes(model, cfg):
    """Name every node. Brevitas exports unnamed nodes and specialize_layers
    creates more, which makes the folding config unaddressable."""
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def full_steps():
    steps = list(build_cfg.default_build_dataflow_steps)
    steps.insert(steps.index("step_specialize_layers") + 1, step_name_nodes)
    return steps


def preflight():
    """Fail in seconds rather than at minute 70."""
    problems = []
    for path, what in [(ONNX, "exported model"),
                       (FOLDING, "folding config"),
                       (SPECIALIZE, "specialize config")]:
        if not os.path.exists(path):
            problems.append(f"missing {what}: {path}")

    if os.path.exists(FOLDING):
        cfg = json.load(open(FOLDING))
        layers = [k for k in cfg if k != "Defaults"]
        if len(layers) < 5:
            problems.append(
                f"folding config has only {len(layers)} layer(s): {layers}. "
                "Expected 5 -- re-run the folding cell in 03_synthesis.ipynb.")
        for name, v in cfg.items():
            if name == "Defaults":
                continue
            if v.get("SIMD", 1) < 2 and "MVAU" in name:
                problems.append(f"{name} has SIMD={v.get('SIMD')} -- HLS "
                                "requires SIMD >= MW/1024")

    if shutil.which("vivado") is None:
        problems.append("vivado not on PATH -- source the settings64.sh files")

    if problems:
        print("PREFLIGHT FAILED")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    cfg = json.load(open(FOLDING))
    print("preflight OK")
    print(f"  model   : {os.path.basename(ONNX)}")
    print(f"  folding : {len(cfg)-1} layers")
    for name, v in cfg.items():
        if name != "Defaults":
            print(f"      {name:<24} SIMD={v['SIMD']:<4} PE={v['PE']}")


def main():
    preflight()

    for d in (OUTPUT, TMP):
        if os.path.exists(d):
            print(f"removing stale {os.path.basename(d)}")
            shutil.rmtree(d, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=OUTPUT,
        board="Pynq-Z2",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        synth_clk_period_ns=CLK_NS,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        folding_config_file=FOLDING,
        specialize_layers_config_file=SPECIALIZE,
        auto_fifo_depths=False,
        verify_save_rtlsim_waveforms=False,
        steps=full_steps(),
    )

    print(f"\nstarting build at {time.strftime('%H:%M:%S')}")
    print("expect 60-90 minutes on 8 cores; FINN prints each step as it goes\n")
    t0 = time.time()
    build.build_dataflow_cfg(ONNX, cfg)
    mins = (time.time() - t0) / 60

    print(f"\nBUILD COMPLETE in {mins:.0f} minutes")
    for name in ("post_synth_resources.json", "post_route_timing.json",
                 "estimate_layer_cycles.json"):
        path = os.path.join(OUTPUT, "report", name)
        if os.path.exists(path):
            print(f"\n=== {name} ===")
            print(json.dumps(json.load(open(path)), indent=2)[:1500])

    deploy = os.path.join(OUTPUT, "deploy")
    if os.path.isdir(deploy):
        print(f"\ndeployment package ready: {deploy}")
        print("copy this to the board for 04_board.ipynb")


if __name__ == "__main__":
    main()
