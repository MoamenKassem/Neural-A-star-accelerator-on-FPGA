#!/usr/bin/env python3
"""Find out why Vivado's implementation run failed, and salvage the numbers.

    python diagnose_build.py

FINN reports "no bitfile found" for every implementation failure, which tells
nothing. The real cause is in Vivado's own logs, and the important
utilisation figures are in the *synthesis* reports -- which usually completed
even when implementation did not.

That matters: if synthesis succeeded and implementation failed, the synthesis
utilisation report IS the result. A design that synthesises but cannot be
placed and routed is a design that does not fit, and the numbers say by how
much.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

TMP = os.path.expanduser("~/fpga_hw2/finn_build_astar_tmp")
OUT = os.path.expanduser("~/fpga_hw2/finn_output_astar")

PYNQ_Z2 = {"LUT": 53200, "FF": 106400, "BRAM": 140, "DSP": 220}


def header(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def find_proj():
    hits = sorted(glob.glob(os.path.join(TMP, "vivado_zynq_proj_*")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def check_disk():
    header("DISK")
    # "Unable to open" is very often a full filesystem or an exceeded quota.
    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    print(f"home filesystem: {free/2**30:.1f} GB free of {total/2**30:.1f} GB")
    if free < 5 * 2**30:
        print("  *** under 5 GB free -- Vivado needs room for temporaries ***")
    for cmd in (["quota", "-s"], ["df", "-h", os.path.expanduser("~")]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                print(f"\n$ {' '.join(cmd)}")
                print(r.stdout.strip())
        except Exception:
            pass


def show_errors(proj):
    header("VIVADO ERRORS")
    logs = []
    for pat in ("**/runme.log", "vivado.log", "**/vivado.log", "*.log"):
        logs += glob.glob(os.path.join(proj, pat), recursive=True)
    logs = sorted(set(logs), key=os.path.getmtime)
    if not logs:
        print(f"no logs under {proj}")
        return

    print(f"{len(logs)} log file(s); newest last\n")
    pat = re.compile(r"^(ERROR|CRITICAL WARNING)", re.I)
    for log in logs[-6:]:
        with open(log, errors="ignore") as f:
            lines = f.readlines()
        hits = [l.rstrip() for l in lines if pat.match(l.strip())]
        if not hits:
            continue
        print(f"--- {os.path.relpath(log, proj)} ---")
        for h in hits[:15]:
            print("   ", h[:160])
        print()


def show_utilisation(proj):
    header("UTILISATION (from synthesis -- the number that matters)")
    rpts = glob.glob(os.path.join(proj, "**", "*utilization*.rpt"), recursive=True)
    rpts += glob.glob(os.path.join(proj, "**", "*_utilization_*.rpt"), recursive=True)
    rpts = sorted(set(rpts), key=os.path.getmtime)
    if not rpts:
        print("no utilisation report found")
        return

    rpt = rpts[-1]
    print(f"from {os.path.relpath(rpt, proj)}\n")
    wanted = ("Slice LUTs", "CLB LUTs", "Slice Registers", "CLB Registers",
              "Block RAM Tile", "DSPs", "LUT as Logic", "LUT as Memory")
    with open(rpt, errors="ignore") as f:
        for line in f:
            if line.startswith("|") and any(w in line for w in wanted):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 5:
                    name, used, _, avail, util = cells[0], cells[1], cells[2], cells[3], cells[4]
                    flag = ""
                    try:
                        if float(util) > 100:
                            flag = "   *** OVER CAPACITY ***"
                        elif float(util) > 85:
                            flag = "   (very high)"
                    except ValueError:
                        pass
                    print(f"  {name:<20} {used:>10} / {avail:<10} {util:>6}%{flag}")


def show_timing(proj):
    header("TIMING")
    rpts = sorted(glob.glob(os.path.join(proj, "**", "*timing_summary*.rpt"),
                            recursive=True), key=os.path.getmtime)
    if not rpts:
        print("no timing report (implementation may not have got that far)")
        return
    with open(rpts[-1], errors="ignore") as f:
        text = f.read()
    for key in ("WNS(ns)", "Timing constraints are not met",
                "All user specified timing constraints are met"):
        if key in text:
            idx = text.index(key)
            print(text[max(0, idx - 200):idx + 300].strip()[:600])
            print()
            break


def main():
    proj = find_proj()
    if proj is None:
        print(f"no vivado_zynq_proj_* under {TMP}")
        print("The build may have been cleaned. Re-run and keep the directory.")
        sys.exit(1)

    print(f"project: {proj}")
    check_disk()
    show_errors(proj)
    show_utilisation(proj)
    show_timing(proj)

    header("WHAT THIS MEANS")
    print("""
If utilisation is over 100% (or LUTs above ~95%): the design does not fit the
Z-7020 at this folding. Halve TARGET_FOLDING in 03_synthesis.ipynb, regenerate
the folding config, and rebuild. Each halving roughly doubles the frame time,
so record the utilisation and throughput at each point -- that series IS the
resource/throughput frontier, and it is the strongest result this project can
produce.

If utilisation is comfortable but errors mention files or disk: clean the
temporaries and rerun.

If timing failed but placement succeeded: lower the clock. Change CLK_NS from
10.0 (100 MHz) to 15.0 (66 MHz) and rebuild -- slower, but it closes.
""".strip())


if __name__ == "__main__":
    main()
