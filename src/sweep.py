"""Sweep encoder width x weight precision, measuring both halves of the trade-off.

For each configuration this measures:
  * planning quality -- p_opt, p_exp, p_gen against vanilla A*
  * hardware cost    -- MACs/cycle, folding, cycles, projected LUTs

Results append to sweep_results.json so the sweep can run in chunks:

    python sweep.py 0 3      # configurations 0,1,2
    python sweep.py 3 8      # configurations 3..7

Produces the data behind Table 5 and Figure 2 of the report.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat

from neural_astar_core import (sample_problems, load_encoder_from_checkpoint,
                               VanillaAStarPlanner, NeuralAStarPlanner, evaluate)
from neural_astar_core.grid import index_to_one_hot
from neural_astar_core.encoder import encoder_input

DATA = "mazes_032_moore_c8.npz"
CKPT = "model/mazes_032_moore_c8/lightning_logs/"
RESULTS = "sweep_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Calibrated on the designs that reached synthesis; see report Section 9.
LUT_PER_MAC = 184839 / 342
LUT_AVAILABLE = 53200

CONFIGS = [
    # (channels, weight_bits, act_bits, mac_budget)
    ((4, 8, 8),    4, 8, 34),
    ((8, 16, 16),  4, 8, 34),
    ((8, 16, 16),  4, 8, 54),
    ((8, 16, 32),  4, 8, 54),
    ((16, 32, 32), 4, 8, 54),
    ((16, 32, 64), 4, 8, 54),
    ((8, 16, 16),  2, 8, 34),
    ((16, 32, 32), 2, 8, 54),
]


class QuantEncoder(nn.Module):
    """Quantized guidance-map encoder at arbitrary width.

    bias=False throughout: every convolution is followed by batch normalization,
    whose shift subsumes a bias, and it leaves FINN one less node to fold.
    """

    def __init__(self, weight_bits, act_bits, channels, in_dim=2):
        super().__init__()
        self.channels = tuple(channels)
        widths = [in_dim] + list(channels) + [1]
        self.input_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                         return_quant_tensor=True)
        blocks = []
        for i in range(len(widths) - 1):
            blocks.append(qnn.QuantConv2d(widths[i], widths[i + 1], 3, 1, 1,
                                          bias=False, weight_bit_width=weight_bits,
                                          return_quant_tensor=False))
            blocks.append(nn.BatchNorm2d(widths[i + 1]))
            if i != len(widths) - 2:
                blocks.append(qnn.QuantReLU(bit_width=act_bits,
                                            return_quant_tensor=True))
        self.model = nn.Sequential(*blocks)
        self.out_q = qnn.QuantIdentity(bit_width=act_bits, return_quant_tensor=False)

    def forward(self, x):
        return self.out_q(torch.sigmoid(self.model(self.input_q(x))))


def plan_folding(channels, budget, in_dim=2, pixels=1024):
    """Give parallelism to whichever layer is the current bottleneck.

    Respects the three hardware constraints: SIMD divides the layer's input
    channels, PE divides its output channels, and SIMD >= MW/1024.
    """
    widths = [in_dim] + list(channels) + [1]
    dims = [(widths[i] * 9, widths[i + 1]) for i in range(len(widths) - 1)]
    ins = [widths[i] for i in range(len(widths) - 1)]
    outs = [widths[i + 1] for i in range(len(widths) - 1)]
    fold = [[1, 1] for _ in dims]

    def cycles(i):
        mw, mh = dims[i]
        return (mw // fold[i][0]) * (mh // fold[i][1]) * pixels

    def used():
        return sum(s * p for s, p in fold)

    while True:
        i = max(range(len(dims)), key=cycles)
        mw, mh = dims[i]
        s, p = fold[i]
        chosen = None
        for ns, np_ in ((s * 2, p), (s, p * 2)):
            ok = (ns <= ins[i] and ins[i] % ns == 0 and mw % ns == 0
                  and np_ <= outs[i] and outs[i] % np_ == 0
                  and ns >= mw / pixels
                  and used() - s * p + ns * np_ <= budget)
            if ok:
                chosen = (ns, np_)
                break
        if chosen is None:
            break
        fold[i] = list(chosen)
    return dims, [tuple(f) for f in fold]


def distillation_set(float_encoder, n_instances, seed=7):
    ps = sample_problems(DATA, "train", n_instances, seed=seed,
                         starts_per_map=4, replace=True)
    X = torch.cat([encoder_input(p.obstacle_map,
                                 index_to_one_hot(p.start, p.height, p.width),
                                 index_to_one_hot(p.goal, p.height, p.width))
                   for p in ps], 0)
    float_encoder.eval()
    with torch.no_grad():
        Y = torch.cat([float_encoder(X[i:i + 256].to(DEVICE)).cpu()
                       for i in range(0, len(X), 256)], 0)
    return X, Y


def distil(channels, wbits, abits, X, Y, epochs, batch=64, lr=3e-3):
    model = QuantEncoder(wbits, abits, channels).to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.L1Loss()
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        total, nb = 0.0, 0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx].to(DEVICE)), Y[idx].to(DEVICE))
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        last = total / max(nb, 1)
    return model.eval(), last


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(CONFIGS)
    n_instances = int(os.environ.get("SWEEP_INSTANCES", 800))
    epochs = int(os.environ.get("SWEEP_EPOCHS", 60))
    n_eval = int(os.environ.get("SWEEP_EVAL", 200))

    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}

    float_encoder = load_encoder_from_checkpoint(CKPT, device=DEVICE)
    X, Y = distillation_set(float_encoder, n_instances)
    eval_set = sample_problems(DATA, "test", n_eval, seed=11, starts_per_map=2)
    vanilla = VanillaAStarPlanner()

    if "float" not in results:
        r = evaluate(eval_set, vanilla, NeuralAStarPlanner(float_encoder, device=DEVICE))
        results["float"] = dict(
            channels="32/64/128/256", wbits=32, abits=32,
            params=sum(p.numel() for p in float_encoder.parameters()),
            p_opt=r.p_opt, p_exp=r.p_exp, p_gen=r.p_gen,
            macs=399310848, mac_cycle=342, cycles=1179648, ms=11.8,
            lut=184839, lut_pct=100 * 184839 / LUT_AVAILABLE, l1=0.0,
            note="measured: implementation failed DRC, 3.47x over capacity")
        json.dump(results, open(RESULTS, "w"), indent=1)
        print(f"float baseline: p_opt {r.p_opt:.3f} p_exp {r.p_exp:.3f} "
              f"p_gen {r.p_gen:.3f}")

    for idx in range(lo, min(hi, len(CONFIGS))):
        ch, wb, ab, budget = CONFIGS[idx]
        key = f"{'_'.join(map(str, ch))}_w{wb}a{ab}_b{budget}"
        if key in results:
            print(f"[{idx}] {key}: already done, skipping")
            continue

        t0 = time.time()
        model, l1 = distil(ch, wb, ab, X, Y, epochs)
        r = evaluate(eval_set, vanilla, NeuralAStarPlanner(model, device=DEVICE))

        dims, fold = plan_folding(ch, budget)
        mac_cycle = sum(s * p for s, p in fold)
        cycles = max((mw // s) * (mh // p) * 1024
                     for (mw, mh), (s, p) in zip(dims, fold))
        macs = sum(mw * mh * 1024 for mw, mh in dims)
        lut = LUT_PER_MAC * mac_cycle

        results[key] = dict(
            channels="/".join(map(str, ch)), wbits=wb, abits=ab, budget=budget,
            params=sum(p.numel() for p in model.parameters()),
            p_opt=r.p_opt, p_exp=r.p_exp, p_gen=r.p_gen, l1=l1,
            macs=macs, mac_cycle=mac_cycle, cycles=cycles,
            ms=cycles / 1e5, lut=lut, lut_pct=100 * lut / LUT_AVAILABLE,
            folding=[list(f) for f in fold])
        json.dump(results, open(RESULTS, "w"), indent=1)

        print(f"[{idx}] {key:26} p_opt {r.p_opt:.3f} p_exp {r.p_exp:.3f} "
              f"p_gen {r.p_gen:.3f} | {mac_cycle:3d} MAC/cy "
              f"{cycles/1e5:6.2f}ms {100*lut/LUT_AVAILABLE:5.0f}%LUT "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
