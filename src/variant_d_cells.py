# =============================================================================
# Cells appended to 04b_build.ipynb
#
# Variant D: the sigmoid moves to the host.
#
# FINN left the terminal sigmoid unfolded, so the accelerator's output is a raw
# 24-bit accumulator. 24 bits is not a native word size, so the PYNQ driver
# unpacks 1024 of them per frame in interpreted Python - which is the whole of
# the 521.91 ms measured end-to-end, against 3.00 ms of actual accelerator work.
#
# Removing the sigmoid leaves FINN with BatchNorm -> Quant, which it folds into
# a MultiThreshold. The output becomes a plain 8-bit activation, numpy handles
# it natively, and the host applies sigmoid to 1024 values - free against a CNN
# that dominates runtime. This is standard FINN practice: classifiers return
# logits and the host does the softmax.
#
# Architecture and folding are unchanged from build B, so DIMS and FOLD are
# already correct and no replanning is needed.
# =============================================================================


# ---------------------------------------------------------------- CELL D1 ---
# Train the no-sigmoid variant, evaluate it, export. ~5 minutes on GPU.

TAG_D = "d"
ONNX_D  = os.path.join(WORKDIR, f"tiny_astar{TAG_D}_w{WEIGHT_BITS}a{ACT_BITS}.onnx")
EST_D   = os.path.join(WORKDIR, f"tiny_est{TAG_D}")
EST2_D  = os.path.join(WORKDIR, f"tiny_est_folded{TAG_D}")
BUILD_D = os.path.join(WORKDIR, f"tiny_output{TAG_D}")


class QuantEncoderNoSigmoid(QuantGuidanceEncoder):
    """Same network, sigmoid moved to the host.

    forward() returns the pre-sigmoid value. Apply torch.sigmoid() to whatever
    the accelerator returns to recover the guidance map.
    """

    def forward(self, x):
        return self.out_q(self.model(self.input_q(x)))


class HostSigmoid(nn.Module):
    """Wrapper so the A* metrics stay comparable with the other builds."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


tiny_d = QuantEncoderNoSigmoid(WEIGHT_BITS, ACT_BITS, channels=CHANNELS).to(DEVICE)
print(f"{sum(p.numel() for p in tiny_d.parameters()):,} params, {CHANNELS}, no sigmoid")

# Targets are now pre-sigmoid, so take them from the float encoder's body.
ps = sample_problems(DATA, "train", 3200, seed=7, starts_per_map=4, replace=True)
Xd = torch.cat([encoder_input(p.obstacle_map,
                              index_to_one_hot(p.start, p.height, p.width),
                              index_to_one_hot(p.goal, p.height, p.width))
                for p in ps], 0)
float_encoder.eval()
with torch.no_grad():
    Yd = torch.cat([float_encoder.model(Xd[i:i+256].to(DEVICE)).cpu()
                    for i in range(0, len(Xd), 256)], 0)
print(f"{len(Xd)} instances | target range [{Yd.min():.2f}, {Yd.max():.2f}] (pre-sigmoid)")

opt = torch.optim.Adam(tiny_d.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
loss_fn = nn.L1Loss()
t0 = time.time(); tiny_d.train()
for ep in range(120):
    perm = torch.randperm(len(Xd)); total, nb = 0.0, 0
    for i in range(0, len(Xd), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(tiny_d(Xd[idx].to(DEVICE)), Yd[idx].to(DEVICE))
        loss.backward(); opt.step()
        total += loss.item(); nb += 1
    sched.step()
    if ep % 20 == 0 or ep == 119:
        print(f"  epoch {ep+1:4d}/120  L1 {total/nb:.5f}")
tiny_d.eval()
torch.save(tiny_d.state_dict(), f"tiny_encoder_{ARCH}_nosigmoid.pt")
print(f"trained in {(time.time()-t0)/60:.1f} min")

rd = evaluate(eval_set, vanilla, NeuralAStarPlanner(HostSigmoid(tiny_d), device=DEVICE))
print(f"\n{'':24} {'p_opt':>7} {'p_exp':>7} {'p_gen':>7}")
print(f"{'build B (sigmoid on HW)':24} {0.750:7.3f} {0.407:7.3f} {0.221:7.3f}")
print(f"{'build D (sigmoid on host)':24} {rd.p_opt:7.3f} {rd.p_exp:7.3f} {rd.p_gen:7.3f}")
print("\nThese should be close. A large gap means the pre-sigmoid targets are")
print("harder to fit than the squashed ones -- raise epochs if so.")

# --- export ---
tiny_d.cpu().eval()
export_qonnx(tiny_d, input_t=torch.zeros(1, 2, 32, 32), export_path=ONNX_D)
m = onnx.load(ONNX_D)
bip = {n.input[1] for n in m.graph.node if n.op_type == "BipolarQuant"}
for init in m.graph.initializer:
    if init.name in bip:
        init.raw_data = np.ones(1, dtype=np.float32).tobytes()
        init.float_data[:] = []
onnx.save(m, ONNX_D)
tiny_d.to(DEVICE)

from collections import Counter
ops = Counter(n.op_type for n in m.graph.node)
print(f"\n{os.path.basename(ONNX_D)}  nodes: {dict(ops)}")
print("Sigmoid in graph:", ops.get("Sigmoid", 0), "(want 0)")

vecs = torch.cat([encoder_input(p.obstacle_map,
                                index_to_one_hot(p.start, p.height, p.width),
                                index_to_one_hot(p.goal, p.height, p.width))
                  for p in list(eval_set)[:64]], 0)
np.savez(f"tiny_test_vectors{TAG_D}.npz",
         x=vecs.numpy(),
         y=torch.sigmoid(tiny_d(vecs.to(DEVICE))).detach().cpu().numpy())
print(f"saved tiny_test_vectors{TAG_D}.npz (y is post-sigmoid, as before)")


# ---------------------------------------------------------------- CELL D2 ---
# Estimate build and folding. ~2 minutes. Check odt before going further.

fresh(EST_D, EST_D + "_tmp")
t0 = time.time()
build.build_dataflow_cfg(ONNX_D, make_cfg(EST_D, EST_D + "_tmp"))
print(f"estimate build in {time.time()-t0:.0f}s\n")

FOLDING_D = os.path.join(WORKDIR, f"folding{TAG_D}.json")
write_folding(EST_D, FOLDING_D)

fresh(EST2_D, EST2_D + "_tmp")
build.build_dataflow_cfg(ONNX_D, make_cfg(EST2_D, EST2_D + "_tmp", folding=FOLDING_D))

cyc = load_report(EST2_D, "estimate_layer_cycles.json")
if cyc:
    worst = max(cyc.values())
    print(f"\nthroughput   : {worst*CLK_NS/1e6:.2f} ms")
    print(f"single-frame : {worst*(557868/147456)*CLK_NS/1e6:.2f} ms")

# The payoff is visible here: the deployment driver reports the output type.
drv = glob.glob(os.path.join(EST2_D, "**", "driver.py"), recursive=True)
if drv:
    txt = open(drv[0]).read()
    for key in ("odt", "oshape_normal"):
        for line in txt.splitlines():
            if line.strip().startswith(key):
                print(f"  {line.strip()}")
print("\nWant odt to be UINT8/INT8 rather than INT24. If it still says INT24,")
print("FINN did not fold the output quantiser and this rebuild will not help.")


# ---------------------------------------------------------------- CELL D3 ---
# Full build. 30-60 min. Run only if D2 showed an 8-bit output type.

fresh(BUILD_D, BUILD_D + "_tmp")
print(f"full build started {time.strftime('%H:%M:%S')}\n")
t0 = time.time()
build.build_dataflow_cfg(ONNX_D, make_cfg(BUILD_D, BUILD_D + "_tmp",
                                          folding=FOLDING_D, bitfile=True))
print(f"\nBUILD COMPLETE in {(time.time()-t0)/60:.0f} min")

top = (load_report(BUILD_D, "post_synth_resources.json") or {}).get("(top)", {})
if top:
    print(f"\n{'Resource':<12} {'Used':>9} {'Available':>11} {'Util':>8}")
    print("-" * 44)
    for key, avail in PYNQ_Z2.items():
        used = top.get(key, top.get(key.replace("_36K", ""), 0))
        print(f"{key:<12} {used:9.0f} {avail:11d} {100*used/avail:7.1f}%")
    print(f"\nbuild B measured 33,820 LUT (63.6%) for comparison")

perf = load_report(BUILD_D, "estimate_network_performance.json")
if perf:
    print(f"\nthroughput   : {1000/perf['estimated_throughput_fps']:.2f} ms/frame")
    print(f"single-frame : {perf['estimated_latency_ns']/1e6:.2f} ms")
    print(f"vs CPU       : {CPU_MS/(perf['estimated_latency_ns']/1e6):.2f}x")

json.dump({"tag": TAG_D, "channels": list(CHANNELS), "budget": MAC_BUDGET,
           "params": sum(p.numel() for p in tiny_d.parameters()),
           "no_sigmoid": True, "reused_weights": False,
           "p_opt": rd.p_opt, "p_exp": rd.p_exp, "p_gen": rd.p_gen,
           "mac_cycle": mac_cycle, "cycles": bottleneck,
           "throughput_ms": bottleneck*CLK_NS/1e6,
           "latency_ms": bottleneck*(557868/147456)*CLK_NS/1e6,
           "est_lut": est_lut,
           "measured_lut": top.get("LUT", 0),
           "measured_lut_pct": 100*top.get("LUT", 0)/PYNQ_Z2["LUT"]},
          open(f"run{TAG_D}_summary.json", "w"), indent=1)

deploy = os.path.join(BUILD_D, "deploy")
if os.path.isdir(deploy):
    print(f"\nDEPLOYMENT PACKAGE: {deploy}")
    print("\nOn the board, section 3's calibration should now find a much")
    print("smaller scale factor, and section 4's latency should drop by two")
    print("orders of magnitude. Copy with:")
    print(f"  scp -r tiny_output{TAG_D}/deploy xilinx@<board-ip>:~/astar{TAG_D}")
    print(f"  scp tiny_test_vectors{TAG_D}.npz mazes_032_moore_c8.npz "
          f"xilinx@<board-ip>:~/astar{TAG_D}/")
    print(f"  scp -r neural_astar_core 05_board.ipynb xilinx@<board-ip>:~/astar{TAG_D}/")
