# =============================================================================
# Cells appended to 04b_build.ipynb
#
# Variant E: build D, with the distillation targets clamped.
#
# Build D moved the sigmoid to the host, which fixed the interface -- measured
# latency fell from 521.91 ms to 3.91 ms -- but cost path optimality, p_opt
# dropping from 0.750 to 0.655. The cause is visible in the training loss: D
# fits pre-sigmoid targets spanning [-9.13, 9.67] and converged to L1 0.427,
# against 0.009 when fitting the squashed (0, 1) output. Logit space is simply
# harder, and the error concentrates in the tails.
#
# Those tails carry almost no information. sigmoid(6) = 0.9975, so everything
# beyond +/-6 is saturated to within 0.3% and the network is spending its
# capacity fitting differences the search can never observe. Clamping the
# targets should cut the loss sharply and recover p_opt, at no cost to the
# guidance map the search actually sees.
#
# One line differs from variant D. Everything else is identical, including the
# folding, so DIMS and FOLD need no replanning.
# =============================================================================


# ---------------------------------------------------------------- CELL E1 ---
# Retrain with clamped targets and evaluate. ~3 minutes. Stop if p_opt
# does not improve -- there is no point building a worse version of D.

TAG_E = "e"
CLAMP = 6.0
ONNX_E  = os.path.join(WORKDIR, f"tiny_astar{TAG_E}_w{WEIGHT_BITS}a{ACT_BITS}.onnx")
EST_E   = os.path.join(WORKDIR, f"tiny_est{TAG_E}")
EST2_E  = os.path.join(WORKDIR, f"tiny_est_folded{TAG_E}")
BUILD_E = os.path.join(WORKDIR, f"tiny_output{TAG_E}")

tiny_e = QuantEncoderNoSigmoid(WEIGHT_BITS, ACT_BITS, channels=CHANNELS).to(DEVICE)

ps = sample_problems(DATA, "train", 3200, seed=7, starts_per_map=4, replace=True)
Xe = torch.cat([encoder_input(p.obstacle_map,
                              index_to_one_hot(p.start, p.height, p.width),
                              index_to_one_hot(p.goal, p.height, p.width))
                for p in ps], 0)
float_encoder.eval()
with torch.no_grad():
    Ye = torch.cat([float_encoder.model(Xe[i:i+256].to(DEVICE)).cpu()
                    for i in range(0, len(Xe), 256)], 0)

print(f"target range before clamp : [{Ye.min():.2f}, {Ye.max():.2f}]")
Ye = torch.clamp(Ye, -CLAMP, CLAMP)          # <-- the only change from variant D
print(f"target range after  clamp : [{Ye.min():.2f}, {Ye.max():.2f}]")
print(f"sigmoid({CLAMP}) = {1/(1+np.exp(-CLAMP)):.4f}, so the clipped tails are")
print("saturated and carry no information the search can use.\n")

opt = torch.optim.Adam(tiny_e.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
loss_fn = nn.L1Loss()
t0 = time.time(); tiny_e.train()
for ep in range(120):
    perm = torch.randperm(len(Xe)); total, nb = 0.0, 0
    for i in range(0, len(Xe), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(tiny_e(Xe[idx].to(DEVICE)), Ye[idx].to(DEVICE))
        loss.backward(); opt.step()
        total += loss.item(); nb += 1
    sched.step()
    if ep % 20 == 0 or ep == 119:
        print(f"  epoch {ep+1:4d}/120  L1 {total/nb:.5f}")
final_l1 = total / nb
tiny_e.eval()
torch.save(tiny_e.state_dict(), f"tiny_encoder_{ARCH}_nosigmoid_clamped.pt")
print(f"trained in {(time.time()-t0)/60:.1f} min")

re_ = evaluate(eval_set, vanilla, NeuralAStarPlanner(HostSigmoid(tiny_e), device=DEVICE))
print(f"\n{'':30} {'L1':>8} {'p_opt':>7} {'p_exp':>7} {'p_gen':>7}")
print(f"{'B  sigmoid on hardware':30} {0.009:8.3f} {0.750:7.3f} {0.407:7.3f} {0.221:7.3f}")
print(f"{'D  host sigmoid, unclamped':30} {0.427:8.3f} {0.655:7.3f} {0.412:7.3f} {0.228:7.3f}")
print(f"{'E  host sigmoid, clamped':30} {final_l1:8.3f} {re_.p_opt:7.3f} "
      f"{re_.p_exp:7.3f} {re_.p_gen:7.3f}")

if re_.p_opt > 0.72:
    print("\nRecovered. Build E gives D's interface with B's optimality --")
    print("continue to E2.")
elif re_.p_opt > 0.655:
    print("\nPartial recovery. Worth building if time allows, but the")
    print("trade-off described in the report still stands.")
else:
    print("\nNo recovery. Do not build: keep D and leave the report's account")
    print("of the trade-off as written. Try CLAMP = 4.0 first instead.")


# ---------------------------------------------------------------- CELL E2 ---
# Export, estimate, folding. ~3 minutes.

tiny_e.cpu().eval()
export_qonnx(tiny_e, input_t=torch.zeros(1, 2, 32, 32), export_path=ONNX_E)
m = onnx.load(ONNX_E)
bip = {n.input[1] for n in m.graph.node if n.op_type == "BipolarQuant"}
for init in m.graph.initializer:
    if init.name in bip:
        init.raw_data = np.ones(1, dtype=np.float32).tobytes()
        init.float_data[:] = []
onnx.save(m, ONNX_E)
tiny_e.to(DEVICE)

from collections import Counter
ops = Counter(n.op_type for n in m.graph.node)
print(f"{os.path.basename(ONNX_E)}  nodes: {dict(ops)}")
print("Sigmoid in graph:", ops.get("Sigmoid", 0), "(want 0)\n")

vecs = torch.cat([encoder_input(p.obstacle_map,
                                index_to_one_hot(p.start, p.height, p.width),
                                index_to_one_hot(p.goal, p.height, p.width))
                  for p in list(eval_set)[:64]], 0)
np.savez(f"tiny_test_vectors{TAG_E}.npz",
         x=vecs.numpy(),
         y=torch.sigmoid(tiny_e(vecs.to(DEVICE))).detach().cpu().numpy())
print(f"saved tiny_test_vectors{TAG_E}.npz")

fresh(EST_E, EST_E + "_tmp")
build.build_dataflow_cfg(ONNX_E, make_cfg(EST_E, EST_E + "_tmp"))
FOLDING_E = os.path.join(WORKDIR, f"folding{TAG_E}.json")
write_folding(EST_E, FOLDING_E)

fresh(EST2_E, EST2_E + "_tmp")
build.build_dataflow_cfg(ONNX_E, make_cfg(EST2_E, EST2_E + "_tmp", folding=FOLDING_E))
cyc = load_report(EST2_E, "estimate_layer_cycles.json")
if cyc:
    worst = max(cyc.values())
    print(f"\nthroughput   : {worst*CLK_NS/1e6:.2f} ms")
    print(f"single-frame : {worst*(557868/147456)*CLK_NS/1e6:.2f} ms")
    print("Should match build D exactly -- same architecture, same folding.")


# ---------------------------------------------------------------- CELL E3 ---
# Full build. ~60 min. Run only if E1 showed p_opt recovering.

fresh(BUILD_E, BUILD_E + "_tmp")
print(f"full build started {time.strftime('%H:%M:%S')}\n")
t0 = time.time()
build.build_dataflow_cfg(ONNX_E, make_cfg(BUILD_E, BUILD_E + "_tmp",
                                          folding=FOLDING_E, bitfile=True))
print(f"\nBUILD COMPLETE in {(time.time()-t0)/60:.0f} min")

top = (load_report(BUILD_E, "post_synth_resources.json") or {}).get("(top)", {})
if top:
    print(f"\n{'Resource':<12} {'Used':>9} {'Available':>11} {'Util':>8}")
    print("-" * 44)
    for key, avail in PYNQ_Z2.items():
        used = top.get(key, top.get(key.replace("_36K", ""), 0))
        print(f"{key:<12} {used:9.0f} {avail:11d} {100*used/avail:7.1f}%")
    print("\nbuild D measured 35,266 LUT (66.3%) -- E should be near identical,")
    print("since only the trained weight values differ.")

drv = glob.glob(os.path.join(BUILD_E, "**", "driver.py"), recursive=True)
if drv:
    for line in open(drv[0]):
        if line.strip().startswith(('"idt"', '"odt"')):
            print(" ", line.strip())
    print("want odt INT8, as in build D")

json.dump({"tag": TAG_E, "channels": list(CHANNELS), "budget": MAC_BUDGET,
           "params": sum(p.numel() for p in tiny_e.parameters()),
           "no_sigmoid": True, "clamp": CLAMP, "final_l1": final_l1,
           "p_opt": re_.p_opt, "p_exp": re_.p_exp, "p_gen": re_.p_gen,
           "mac_cycle": mac_cycle, "cycles": bottleneck,
           "throughput_ms": bottleneck*CLK_NS/1e6,
           "latency_ms": bottleneck*(557868/147456)*CLK_NS/1e6,
           "est_lut": est_lut, "measured_lut": top.get("LUT", 0),
           "measured_lut_pct": 100*top.get("LUT", 0)/PYNQ_Z2["LUT"]},
          open(f"run{TAG_E}_summary.json", "w"), indent=1)

deploy = os.path.join(BUILD_E, "deploy")
if os.path.isdir(deploy):
    print(f"\nDEPLOYMENT PACKAGE: {deploy}")
    print("\nOn the board, expect build D's latency with build B's optimality.")
    print(f"  scp -r tiny_output{TAG_E}/deploy xilinx@<board-ip>:~/astar{TAG_E}")
    print(f"  scp tiny_test_vectors{TAG_E}.npz xilinx@<board-ip>:~/astar{TAG_E}/")
    print(f"  ssh xilinx@<board-ip> 'cd ~/astar{TAG_E} && "
          f"ln -s ../astard/neural_astar_core . && "
          f"ln -s ../astard/mazes_032_moore_c8.npz . && "
          f"cp ../astard/05_board.ipynb .'")
