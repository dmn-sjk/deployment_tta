# Design

Terms are defined in [GLOSSARY.md](../GLOSSARY.md). Decisions with lasting consequences are in [docs/adr/](adr/).

## Goal

Deploy a ViT image classifier with test-time adaptation (TENT) for a single stream of frames on an NVIDIA GPU. There are two targets:

- **RTX 3080 (10 GB):** measured. The goal is the highest real frame rate possible on this card.
- **Jetson Orin family on JetPack 6:** design target, neither measured nor projected (ADR 0006). The design must work on the smallest member, **Orin Nano 8GB**, which then guarantees Orin NX and AGX Orin.

The project is about **deployment**, not TTA research: the adaptation method is used as published, and success means the deployed system computes the PyTorch reference's TENT at the highest frame rate possible.

Frame rate is traded against **adaptation frequency**: adapting on fewer frames costs fewer backward passes and allows a higher frame rate, at some accuracy cost. A typical camera runs at 30 fps. The central questions:

> Which methods sustain a 30 fps camera, at what latency, and what is each method's max sustainable fps? How do latency and max fps change with B and S?

Accuracy is context, not the result. TENT is unsupervised, so its updates are noisy and not necessarily correct. Each method's accuracy is the reference's at the same (B, S), since both TENT methods compute the reference's updates up to floating-point error and lag.

## Scope

**In scope:** source model (no adaptation) and TENT, classification, ImageNet-C continual stream, adaptation batch B and stride S, RTX 3080 measured, Orin as the design target.

**Out of scope:** other TTA methods, detection/segmentation, label-shift streams, episodic evaluation, on-device Jetson runs, serving frameworks, INT8, frame-path pipelining. See [Future work](#future-work).

**Known limitation:** the benchmark stream is i.i.d. within each corruption. A real camera stream is temporally correlated, and that is where TENT at small B is known to fail. Accuracy numbers are therefore an upper bound for a real single-camera deployment.

## Evaluation

Timing runs use the dev subset with seed 1, since timing does not depend on the images or their order. The reference runs on the full 5,000-per-corruption subset with 3 seeds and gives the reported accuracy. The dev subset gives about 0.6 pp standard error on the average, so dev accuracy is used only for correctness checks.

## Key decisions

| Decision | Why | Instead of |
| --- | --- | --- |
| torchvision ViT-B/16, `IMAGENET1K_V1`, the checkpoint used by ROID | Same model as the external reference, so the TENT result is directly comparable. LayerNorm has no batch statistics, so batch size 1 is well-defined. | timm AugReg (SAR's checkpoint), ResNet-50 with BatchNorm |
| Source + TENT only | No new hyperparameters to tune. TENT is the canonical backprop-based TTA baseline. | SAR, EATA, DeYO, ROID |
| TENT default hyperparameters, lr scaled by B | lr = 2.5e-4 × B / 64, SGD with momentum 0.9 and Nesterov, as in TENT's and ROID's ImageNet configs. Not tuned. | SAR's lr rule, tuning on the test stream |
| Own PyTorch implementation is the reference | Every method is compared with it on the same stream and (B, S). Paper numbers are a loose sanity check. | Matching published numbers |
| Reported accuracy is the reference's, timing comes from dev-stream live runs | The deployed methods compute the reference's updates, and TENT's unsupervised updates make accuracy context. Timing does not depend on the images or the seed. | Full-subset live runs per method and seed |
| Continual protocol only (ADR 0001) | Resets at corruption boundaries need oracle knowledge a deployed system never has. | Episodic + continual |
| RobustBench 5,000-image ImageNet-C subset | Standard fixed subset, used by ROID. 10× faster than the full set. | Full 50,000 images per corruption |
| Two adaptation knobs: B and S | B amortizes per-step overhead. Only S reduces adaptation compute per frame, so S trades accuracy for fps. | B only |
| Max sustainable fps under a strict one-interval p99 bound (ADR 0002) | Honest real-time edge figure. | Bounded queue only, p99 ≤ k intervals |
| Orin Nano 8GB as the design floor | Smallest compute and memory in the Orin family. | Designing for AGX Orin |
| Desktop TensorRT pinned to JetPack 6's version | Same TensorRT major.minor on both sides. | Latest TensorRT on desktop |
| bf16 autocast for adaptation compute, FP32 master copy of the adaptable parameters | Ampere tensor cores on both devices, no GradScaler. At lr ≈ 4e-6 a step changes γ by far less than one bf16 ulp near 1.0 (≈ 8e-3), so the parameters themselves must stay FP32 or updates vanish. | FP16 + GradScaler, bf16 parameters |
| TensorRT FP16 for the frame path, γ/β as FP32 inputs, LayerNorm in FP32 | FP16 resolution near 1.0 (≈ 1e-3) would hide hundreds of steps of updates and then apply them as one jump. | All-FP16 engine |
| Gradient accumulation for sync TENT | Exactly batch-B TENT, because LayerNorm does not mix samples. 1 forward + 1 backward per adaptation frame, constant memory, even latency. | Re-forward at batch B, retained graphs |
| Adaptation decoupled from the frame path | Frame latency stops depending on the backward pass. The cost (lag) is measured. | Only synchronous TTA |
| Adapter step captured as one CUDA graph | A step becomes one launch, so the adapter thread barely holds the GIL and does not stall the frame thread. | Eager adapter |
| No serving framework | One stream, one model, one device. | Triton |

## Target device

| | RTX 3080 (10 GB) | Orin Nano 8GB (Super mode) | AGX Orin 64GB (MAXN) |
| --- | --- | --- | --- |
| GPU architecture | Ampere (sm_86) | Ampere (sm_87) | Ampere (sm_87) |
| SMs | 68 | 8 (1024 CUDA cores) | 16 (2048 CUDA cores) |
| Max GPU clock | 1.71 GHz | 1.02 GHz | 1.3 GHz |
| SM × clock (compute proxy) | 116 | 8.2 (**≈ 14× below 3080**) | 20.8 (≈ 5.6× below) |
| Memory bandwidth | 760 GB/s | 102 GB/s | 204.8 GB/s |
| Memory | 10 GB discrete | 8 GB unified, shared with OS and CPU | 64 GB unified |
| CPU | desktop x86 | 6× Cortex-A78AE @ 1.7 GHz | 12× Cortex-A78AE @ 2.2 GHz |
| Tensor-core precisions | FP16, bf16, TF32, INT8 | same | same |

**Software target: JetPack 6.2** (Jetson Linux 36.4.3, Ubuntu 22.04, CUDA 12.6, cuDNN 9.3.0, TensorRT 10.3.0). Desktop NGC PyTorch 24.08 matches it: Ubuntu 22.04, CUDA 12.6, cuDNN 9.3.0.75, TensorRT 10.3.0.26, Python 3.10, PyTorch 2.5.0a0+872d972e41.

**No projection (ADR 0006).** Every reported number is measured on the 3080. Orin constrains the design (memory floor, portability), not the reported numbers. The SM × clock row above says only how far below the 3080 Orin sits.

**Portability constraints:**
- **TensorRT pinned to 10.3** via NGC PyTorch `24.08` (`nvcr.io/nvidia/pytorch:24.08-py3`). Native ONNX `LayerNormalization`, `enqueueV3` / `set_tensor_address` and `refit_cuda_engine_async` are available on both sides.
- **PyTorch versions differ.** Desktop 2.5.0a0, Orin JetPack 6.2 wheels 2.7.0a0 / 2.8.0a0. The adapter and the reference use only PyTorch 2.5 APIs. The model definition comes from torchvision, which needs a matching torchvision build on Orin.
- **Memory floor 8 GB unified.** Model, TensorRT engine, adapter activations and the OS must fit together. Peak memory is reported per configuration.
- No x86-only dependencies. Engines are rebuilt on the target, never shipped. CUDA graphs are portable.

## Setup

- **Model:** `torchvision.models.vit_b_16(weights="IMAGENET1K_V1")`, clean ImageNet top-1 81.07%. Same checkpoint as ROID (its repo's default `MODEL.WEIGHTS`).
- **Preprocessing:** `ToTensor` → `Normalize` with the mean/std of the weights' `transforms()` (ImageNet mean/std). A test asserts the values.
  - ImageNet-C images are already 224×224 (resize + center-crop happened before corruption), so no resize or crop, as in ROID.
- **Data:** ImageNet-C, severity 5, all 15 corruptions, 5,000 images per corruption from the RobustBench fixed list. 75,000 images in total.
  - The Zenodo tars are per corruption family and contain all 5 severities. Extraction streams each tar once and writes only `<corruption>/5/<class>/<file>` entries that appear in the RobustBench ID list.
  - A manifest (path, label, SHA-256) is committed so the subset is verifiable.
- **Dev subset:** a fixed seeded sample of 500 images per corruption (7,500 total) from the 5,000 list, committed as its own manifest.
- **Stream (continual protocol):** the 15 corruptions concatenated in the standard order, no reset. Images shuffled within each corruption.
  - The stream is one sequence, so adaptation batches cross corruption boundaries. ROID instead takes a smaller final step per corruption (5,000 = 78 × 64 + 8 at B = 64). A deployed system sees no boundaries either.
  - **Seeds:** shuffle seeds {1, 2, 3}, following ROID's seed scheme. Seed 1 is the default for single-seed runs.

## Methods

**Source:** frozen model, eval mode.

**TENT (reference definition):**
- Adaptable parameters: γ/β of every LayerNorm (`encoder.layers.*.ln_1`, `encoder.layers.*.ln_2`, final `encoder.ln`): 25 layers × 768 × 2 = 38,400 values. Everything else frozen.
- Loss: Shannon entropy of the softmax predictions, averaged over the adaptation batch.
- Model in `eval()` mode, as in ROID. This checkpoint has no active dropout and no BatchNorm, so `train()` and `eval()` give the same forward pass. A test confirms this.
- Optimizer: SGD, momentum 0.9, Nesterov, dampening 0, no weight decay, **lr = 2.5e-4 × B / 64**. B = 1 gives 3.9e-6, B = 64 gives 2.5e-4. This is TENT's ImageNet default (lr 2.5e-4 at batch 64, as in ROID's `cfgs/imagenet_c/tent.yaml` and `conf.py`), scaled linearly with B.
- One step per adaptation batch of B adaptation frames. Returned predictions come from the forward pass that computes the loss, i.e. from before the update.
- Precision: FP32, deterministic algorithms enabled (`torch.use_deterministic_algorithms(True)`, math SDPA backend).

**Stride S (the fps knob).** Every frame is predicted. Only frames with index ≡ 0 (mod S) are adaptation frames, grouped into adaptation batches of B in arrival order. S = 1 is published TENT. S > 1 is a deliberate deviation: same loss, adaptable parameters, optimizer and lr rule, but 1/S as many steps per corruption. The reference runs the same S values, so fidelity is always judged at matched (B, S).

**Single-stream forms of TENT.** LayerNorm has no cross-sample dependence, so a sample's prediction depends only on the current adaptable parameters, and batch-B TENT can be computed one frame at a time:

- **Gradient accumulation (sync form, `pytorch_tent`):** each adaptation frame runs one forward with grad, which gives its prediction, then one backward on loss / B into accumulated gradients. Every B-th adaptation frame triggers the step and clears the gradients. Non-adaptation frames run a no-grad forward. Cost per adaptation frame: 1 forward + 1 backward. Memory does not depend on B.
- **Re-forward at batch B (the `trt_tent` adapter's form):** predict every frame with a no-grad batch-1 forward. After every B-th adaptation frame, re-run forward + backward at batch B on the last B adaptation frames and step. Cost: one extra forward per adaptation frame, memory ∝ B, but batch-B kernels use the GPU better.

Batch-1 and batch-B kernels round differently, so these forms agree with the reference only up to floating-point error, which TENT then amplifies over thousands of steps (see [Fidelity](#fidelity)).

**External sanity check** (ROID, Marsden et al., WACV 2024, arXiv:2306.00650, Table 1: continual IN-C severity 5, RobustBench 5,000 subset, same checkpoint, TENT defaults at B = 64, 5 runs, error %):

| Source | TENT |
| --- | --- |
| 60.2 | 54.5 |

The reference at (B, S) = (64, 1) uses the same checkpoint, data, optimizer and lr, so it should land close to these numbers. Shuffle order and implementation details (for example batches crossing corruption boundaries, see Setup) can still differ, so the gap is reported.

## Fidelity

TENT is a chaotic system: a 1-ulp difference in one step changes every later parameter, so per-frame agreement decays along the stream even for a correct implementation. Fidelity is defined at two levels, always at matched (B, S).

- **Short-horizon equivalence (implementation correctness):** same initial parameters, same first 50 steps, FP32 + deterministic. Adaptable-parameter L∞ difference after each step < 1e-5 relative, and per-frame logits agree within tolerance. Catches wrong losses, wrong parameter sets, off-by-one batching or striding, wrong prediction timing. Covered by the test suite.
- **Deployed vs reference (report-only):** the dev-stream accuracy of each TENT method's 30 fps live run minus the dev reference at the same (B, S). A gap far above dev noise points to a bug.
  - **Collapsed configs** (reference average below source) are marked in the report, never dropped. Small B is expected to collapse without resets.
- **Precision delta:** the reference can also run in bf16 autocast, to tell precision effects from implementation bugs.

## Metrics

Recorded for every method, on the same stream, B and S:

- **Sustains 30 fps:** the 30 fps live run over the whole dev stream is sustainable (p99 end-to-end latency ≤ 33.3 ms and, in spec mode, the adapter keeps up).
- **Max sustainable fps:** the highest replay rate at which p99 end-to-end latency ≤ one frame interval and, in spec mode, the adapter keeps up (ADR 0002). See [Measuring max fps](#measuring-max-fps). Deploy mode is not given a max fps. It is measured at a given input fps and reports effective stride and drop rate.
- **Accuracy-vs-fps curve:** reference accuracy and measured max sustainable fps for each method and (B, S), plus source (inference-only) as the fps ceiling.
- **Accuracy:** the reference's online top-1 per corruption and averaged, on the full subset, as the mean and range over 3 seeds. Accuracy-over-time curve.
- **Latency:** per-frame p50 / p99 / p99.9, end to end (decoded image in host memory → prediction on host) and GPU-only, with host time and GPU time split.
- **Async modes:** adapter steps/s, lag (steps), queue depth, dropped adaptation batches, effective stride.
- **Resources:** peak GPU memory (and whether it fits Orin Nano's 8 GB with OS headroom), GPU utilisation, power (`nvidia-smi`).
- **Health:** rolling prediction entropy and rolling predicted-class histogram entropy, as collapse indicators.

## Measuring max fps

Applies to every max-fps number (`orchestrate.py` with `experiments/max_fps*.yaml`), for every method except deploy mode.

1. **Bisection** on the first 2,000 frames of the stream (ADR 0002, p99 ≤ one frame interval).
2. **Repeat replays:** the bisected rate must pass 2 repeat replays in a row. Otherwise multiply it by 0.98 and repeat.
3. **Whole-stream live run** at that rate, on the dev stream (7,500 frames, seed 1). It must also be sustainable. Otherwise multiply the rate by 0.98 and run it again. The reported max fps is the rate whose live run held.
4. **Repeats:** TensorRT TENT in spec mode runs steps 1 to 3 three times, each in new processes (`repeats: 3` in `experiments/max_fps.yaml`). Its contended frame latency varies by about 20% between processes. The reported max fps is the median repeat's, with the range over repeats.

**Spec mode: the adapter keeps up.** Over a replay or live run, let *arrived* be the adaptation batches the frame path completed and *stepped* the steps finished by the time the last frame is answered. The adapter keeps up iff stepped ≥ 0.98 × arrived − 1. The −1 allows the one batch that is normally in flight. A run where the frame path blocked on full adaptation slots is never sustainable. Lag is reported.

**Measurement hygiene:** every measurement process waits 5 s before it starts, so the previous run's GPU work has drained. GPU utilisation, power, SM clock and temperature are recorded per run (`nvidia-smi`).

## Architecture

```
frames ──► FramePath (high-priority stream): pinned staging ─► GPU preprocess ─► method.infer ─► logits to host
                                                                     │
        ┌──────────────────────┬──────────────────────┬──────────────┴────────────────────────┐
  pytorch_source         pytorch_tent             trt_source                           trt_tent
  eager FP32, no grad    eager FP32, grad-        TensorRT FP16 engine,               TensorRT frame path + decoupled adapter
                         accumulation TENT in     source γ/β                          (low-priority stream, CUDA-graph bf16 step,
                         the frame path                                               double-buffered γ/β handoff, spec | deploy)
```

- **Method** (`methods/method.py`): one `typing.Protocol` (`infer`, `settle`, `adaptation_progress`, `summary`, `traces`, `close`) implemented by plain classes, no inheritance: one `Source` over any frozen forward (eager PyTorch or the engine), `PytorchTent` and `TrtTent`. Configs are frozen dataclasses, `build_method` dispatches on them with `match` and loads the model and the built engine. Engine and adapter variants are configs of `trt_source` and `trt_tent`.
- **Frame path** (`methods/frame_path.py`): composition over any method, so every method pays the same copy, preprocessing and synchronisation.
- **Runtime internals** (`runtime/`): `TrtEngine` (γ/β bound to a packed FP32 buffer per execution context, publish is a pointer swap), `Adapter` (thread, slots, spec/deploy queueing, NaN revert, disable, reset to source), `AdapterStep` (forward + backward + SGD as one CUDA graph, micro-batch m).
- **Deploy-mode operations:** NaN/Inf check after every step with revert, runtime disable and reset to source, health metrics logged once per second from their own thread.
- **Reference** (`reference.py`): offline, deterministic FP32, batched by segments that share adaptable parameters, with the lag simulator (`prediction_steps`).
- **Engines** are built once by `scripts/engines.py` with a record (export config, TensorRT version, engine SHA-256, parity with PyTorch). Runs refuse an engine that does not match its record and never build one, so no build touches a measured GPU.
- **Reports** (`report/`) are pure functions over saved runs, separate from the running scripts. Every live run saves its method's config fields (`config` in `metrics.json`), so a report never parses method names.

## Risks on Orin

| Risk | Mitigation |
| --- | --- |
| The 3080 hides decoupled-adapter contention: frame and adapter kernels share 68 SMs, while Orin Nano has only 8 | The decoupling benefit on Orin is stated as not measured. |
| GPU kernels are not preempted: a long backward kernel delays the next frame | Stream priorities. Smaller micro-batches in the adapter. |
| Adapter activation memory does not fit Orin Nano at large B | Micro-batch m bounds it. |
| ViT-B adaptation fps on Orin Nano too low to be useful | Stride S raises fps. ViT-S/16 as a fallback backbone. |
| torchvision build on Orin | NVIDIA / jetson-containers provide matching torchvision. Only the adapter needs it. |

## Future work

- **On-device Orin runs:** measured fps, `tegrastats`, power modes, thermals.
- **C++ frame path:** TensorRT C++ runtime with the GIL released, as a fifth method behind the same protocol.
- **Frame-path pipelining:** several frames in flight under a relaxed latency bound (see ADR 0002).
- **Correlated / label-shift streams** (class-sorted, Dirichlet-ordered) to show where TENT fails in a real camera setting.
- **More TTA methods:** EATA, SAR, DeYO, ROID, through the same adapter interface.
- **Automatic collapse recovery** (as in SAR's model reset).
- **Other tasks:** semantic segmentation with per-pixel entropy (e.g. ACDC, SHIFT).
