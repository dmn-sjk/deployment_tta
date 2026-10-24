# Max sustainable fps uses a strict one-frame latency bound

A frame rate is sustainable only if p99 end-to-end latency is at most one frame interval and, in spec mode, the adapter keeps up. This keeps the headline number an honest real-time edge figure: every frame is answered before the next one arrives.

## Considered Options

- **p99 ≤ k intervals (k ≥ 2):** would let the frame path pipeline several frames in flight and report much higher fps on the RTX 3080, at the cost of up to k frames of latency. Rejected for the headline. Pipelining is listed under Later.
- **Bounded queue only:** allows unbounded-looking latency spikes from adaptation steps to be averaged away. Rejected.

## Consequences

Max fps is capped at 1 / p99 latency, so throughput tricks that do not cut per-frame latency do not show up in the headline. Decoupled adaptation mainly improves p99, not total work per frame.

The replayer busy-waits between frames, yielding the GIL with `sleep(0)`, so the frame-path core never idles. Measured on the 3080 (Oct 21, 2026): with a sleeping wait, PyTorch FP32 eager frames slowed from 5.5 ms to 13 ms p50 (31 ms p99) at 30 fps because of CPU wake-up, while the GPU clock stayed at 1.8 GHz. Sustainability also stopped being monotonic in fps, which broke the max-fps bisection. Headline fps is therefore the best case for host cost and burns one CPU core (1 of 6 on Orin Nano). A camera loop that sleeps would see slower eager launches. TensorRT, a single enqueue, is barely affected. The frame path's `Event.synchronize()` already spins under CUDA's auto scheduling and releases the GIL, so it needs no change. A pure spin without `sleep(0)` delayed another Python thread by up to 5 ms (the GIL switch interval), which the adapter thread would pay.

## The adapter keeps up (Oct 22, 2026)

Checked per replay and per live run, on top of the p99 bound. *Arrived* is the adaptation batches the frame path completed, *stepped* the steps finished by the time the last frame is answered. The adapter keeps up iff stepped ≥ 0.98 × arrived − 1 (`KEEPS_UP_FRACTION`). The −1 allows the one batch that is normally in flight. A run where the frame path blocked on full adaptation slots is never sustainable. The adapter has 4 batch slots in spec mode (`SPEC_SLOTS`), the hard stop. Lag stays in the report, so it needs no bound of its own here. Deploy mode drops instead of falling behind, so the rule only applies to spec mode.

Replaces "at most 2 complete adaptation batches ever wait for the adapter" (Oct 21, `MAX_PENDING_BATCHES`). Also rejected: "pending ≤ 1 at the end of the replay", which lets the lag spike mid-run, and "no upward trend between the two halves of the replay", which is noisy on 2,000 frames.

## Whole-stream live run (Oct 22, 2026, amended Oct 23)

The confirmed rate is replayed once more over the whole dev stream (7,500 frames, seed 1). It must also be sustainable. Otherwise the rate steps down by 2% and the live run is repeated. The reported max fps is the rate whose live run held, and that run's lag trace is the one the simulator (a dev tool) replays. Applies to every max fps (`scripts/orchestrate.py` on `experiments/max_fps*.yaml`): every method and variant except deploy mode. Amended Oct 23, 2026: the live run's accuracy is report-only. The reported accuracy is the reference's, and timing does not depend on the images or the seed, so the live run no longer runs on the full subset. At most 5 live step-downs (`LIVE_MAX_STEP_DOWNS`), then the run fails loudly: a live run that still fails 10% below the confirmed rate points to a fault, not to the rate. A step-down that stops partway deletes the previous `max_fps.json` first, and the report and the simulator only accept a live run whose rate matches `max_fps.json` and that held. Reason: on the full dev stream, live spec runs at a rate confirmed on 2,000 frames failed p99 (B1_S4 at 492 fps: 2.44 vs 2.03 ms, twice), so a 2,000-frame confirmation alone overstates the headline.

fps runs need the GPU to themselves. Every measurement process started through `run_script` (`orchestrate.py`: every run of an experiment file) waits 5 s first (`COOLDOWN_SECONDS`, `eval/workers.py`), so the previous run's GPU work has drained. A script started by hand does not wait. The operator keeps other GPU work off the device: nothing checks for it. Replaces (Oct 23, 2026) the nvidia-smi idle check (utilisation ≤ 5% over 2 s, no compute process, from a process without a CUDA context), which ran in every measurement process and was dropped as redundant with the cooldown.

## Confirmed max fps (Oct 21, 2026)

The bisected rate is reported only once it passes 2 repeat replays in a row (`CONFIRM_REPLAYS`). Otherwise the rate steps down by 2% (`CONFIRM_STEP`) and is replayed again. Reason: the bisection ends on the edge of sustainability, and a repeat replay at that rate failed in 14 of 25 decoupled-adapter variants. Most missed p99 by under 2%. Adapter-bound variants could hold ≤ 2 pending batches on one 2,000-frame replay and run away to 4 pending on the next, because the adapter queue drifts when the adapter is the bottleneck. Rejected: bisecting on the full dev stream, about 4× slower.
