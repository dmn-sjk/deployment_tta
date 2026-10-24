# Real-Time Test-Time Adaptation on TensorRT

A concept for real-time edge deployment of a test-time adaptation (TTA) method. TENT ([Wang et al., 2021](https://arxiv.org/abs/2006.10726)) adapts a ViT-B/16 classifier on the ImageNet-C benchmark in a continual setting, simulating a real-time camera stream. Inference runs in TensorRT FP16: the model is exported to ONNX, with the LayerNorm parameters exposed as graph inputs and softmax cast to FP32, and then built into a strongly typed TensorRT engine. Adaptation runs asynchronously in a separate thread, in PyTorch with a captured CUDA graph, and updates the engine's LayerNorm inputs with no engine refit. The system sustains up to 271 FPS on an RTX 3080, compared to 47 FPS for eager PyTorch, without losing accuracy. The repository also includes a naive eager PyTorch implementation for performance comparison and correctness checks.

The dependency versions were chosen to match JetPack 6.2, the software stack on Jetson Orin.

## Purpose of the project
This project is a proof of concept for deploying backpropagation-based test-time adaptation (TTA). It examines how such a method can be deployed and optimized with a modern software stack. It does not propose a new TTA method or aim to improve accuracy, which is why it uses one of the simplest TTA methods (TENT) on one of the simplest tasks (image classification). The design keeps the model always available for prediction and minimizes the computational impact of adaptation. 

The accuracy results compare the optimized pipeline against the naive eager PyTorch implementation used in almost all TTA papers. Their purpose is to verify that the optimized pipeline is correct. Keep in mind that TTA updates are unsupervised and therefore noisy. A faster pipeline can potentially perform more adaptation steps, but more steps do not necessarily lead to higher accuracy (see, e.g., [Niu et al., 2022](https://arxiv.org/abs/2204.02610), [Sreeram et al., 2026](https://arxiv.org/pdf/2602.06136)).

![Diagram of the optimized TTA system](assets/diagram.png)

## Design decisions and tackled issues
- **Single frame at a time, simulating camera stream.** Every frame is predicted on its own (batch 1) as it arrives. Adaptation still uses batches of B adaptation frames (see below).
- **ViT architecture choice:** It avoids the issues related to BatchNorm layer's statistics update in ResNets. Additionally, [Niu et al., 2022](https://arxiv.org/abs/2204.02610) show that ViTs are more robust to distribution shifts than ResNets. 
- **LayerNorm (LN) params as engine inputs, zero-copy, no refit.** TENT adapts only the LayerNorm affine parameters. They enter the TensorRT engine as graph inputs, which costs no measurable time (GPU median about 1.1 - 1.3 ms either way). The engine keeps two packed LN params buffers, each bound once to its own execution context. The adapter writes new LN params into the back buffer in place and publishes it: the next frame reads it, and the old front buffer becomes the new back buffer. A publish copies nothing and leaves the engine unchanged. A refit instead costs 0.56 ms host + 0.45 ms GPU per update and cannot run while a frame is in flight, so a stall-free refit design needs two engines, doubling engine memory (2 × 176 MB, versus one shared engine plus 2 × 3.3 MB for buffers and contexts). See [ADR 0005](docs/adr/0005-layernorm-parameters-as-engine-inputs.md).
- **LayerNorm in FP32.** Each LayerNorm is computed in FP32 from FP32 LN params (`MixedPrecisionLayerNorm`, a cast to FP32 before and back to FP16 after). In FP16, small TENT updates (considering low LR) to LN would round away.
- **Softmax in FP32.** Attention softmax is wrapped in a cast pair after ONNX export. With FP16 softmax, prediction agreement fell below that of bf16 autocast on one of the domains. FP32 LayerNorm, FP32 softmax and LN params inputs together cost within 2% of inference GPU time.
- **Inference FP16, Adaptation bf16.** The backward pass needs range, so the adapter runs in bf16 autocast with FP32 weights. The LN params stay FP32 master copies, for the same rounding reason as above. Forward-only inference needs precision to match the FP32 reference, so the inference model is an FP16 TensorRT engine with LayerNorm and softmax in FP32.
- **Eager PyTorch TENT with gradient accumulation.** A single stream delivers one frame at a time, but TENT adapts on batches of B frames. Each adaptation frame runs one forward with grad, which gives its prediction, then one backward on loss / B into the accumulated gradients. Every B-th adaptation frame runs the SGD step and clears the gradients. With stride S, only every S-th frame is an adaptation frame, and the others run a no-grad forward (at S = 1, the published TENT, every frame adapts). Each adaptation frame pays 1 forward + 1 backward, memory does not depend on B, and frame latency stays even. The alternative, buffering B frames and re-running forward + backward at batch B, costs an extra forward per adaptation frame, memory ∝ B and a latency spike on every step frame. The decoupled optimized approach (`trt_tent`) uses that re-forward form, because there the extra forward runs off the inference thread and batch-B kernels use the GPU better.
- **Adapter step as one CUDA graph.** Forward, backward and the SGD update are captured as a single CUDA graph, so a step is one launch.

## Setup

### Requirements

- NVIDIA GPU (Compute capability 8.0 or newer (Ampere+))
- NVIDIA driver 560 or newer.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker Compose v2 or newer.
- About 25 GB free under Docker's data root for the image, and about 60 GB in the data directory for ImageNet-C (tars plus subset)

### Installation

1. Create `.env` for your machine. Set `HOST_DATA_DIR` which will hold everything that is not source code: datasets, checkpoints, TensorRT engines, run outputs, logs and caches. It will be mounted at `/data` in the container.

   ```bash
   cp .env.example .env
   sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env
   ```

2. Create the data directory.

   ```bash
   mkdir -p "$(grep '^HOST_DATA_DIR=' .env | cut -d= -f2-)"
   ```

3. Build the image (NGC PyTorch 24.08: PyTorch 2.5, CUDA 12.6, cuDNN 9.3, TensorRT 10.3, matching JetPack 6.2).

   ```bash
   docker compose build
   ```

### Data

ImageNet-C ([Hendrycks and Dietterich, 2019](https://zenodo.org/records/2235448)), severity 5, all 15 corruptions, restricted to the RobustBench ([Croce et al., 2021](https://github.com/RobustBench/robustbench)) 5,000-image subset.

1. Download the four corruption families (about 50 GB, resumable) and verify their checksums. The `extra` family is not used.

   ```bash
   docker compose run --rm tta bash -c '
     mkdir -p /data/downloads/imagenet-c && cd /data/downloads/imagenet-c
     for f in blur digital noise weather; do
       wget -c --tries=20 --retry-connrefused -O $f.tar \
         https://zenodo.org/api/records/2235448/files/$f.tar/content
     done
     md5sum -c <<EOF
   2d8e81fdd8e07fef67b9334fa635e45c  blur.tar
   89157860d7b10d5797849337ca2e5c03  digital.tar
   e80562d7f6c3f8834afb1ecf27252745  noise.tar
   33ffea4db4d93fe4a428c40a6ce0c25d  weather.tar
   EOF'
   ```

   Rerun the same command to resume an interrupted download.

2. Extract the subset and build the manifests. Only `<corruption>/5/<class>/<file>` entries on the RobustBench subset of ImageNet-C are written to `/data/datasets/imagenet-c`. Once all 15 corruptions are present, the script verifies every SHA-256 and writes `manifests/imagenet_c_s5_full.csv` (75,000 images) and `manifests/imagenet_c_s5_dev.csv` (7,500 images).

   ```bash
   docker compose run --rm tta python scripts/extract_imagenet_c.py \
     /data/downloads/imagenet-c/{blur,digital,noise,weather}.tar
   ```

3. Optionally delete the tars once extraction is done: `rm -r /data/downloads/imagenet-c` (inside the container).

## Running

All code runs inside the container.

```bash
docker compose run --rm tta                     # interactive shell
docker compose run --rm tta python -m pytest    # one-off command
```

Inside the container:

| Path | Contents |
| --- | --- |
| `/workspace/deployment_tta` | This repository, mounted read-write. `src/` is on `PYTHONPATH`. |
| `/data` | The data directory (`HOST_DATA_DIR`), also in `$DATA_DIR`. |

### Format, lint, type check

```bash
docker compose run --rm tta ruff format .              # format
docker compose run --rm tta ruff check --fix .         # lint, apply safe fixes
docker compose run --rm tta docformatter -i src scripts tests  # wrap docstrings
docker compose run --rm tta mypy src scripts tests     # type check
```

### Experiments

```bash
python scripts/engines.py                    # build every TensorRT engine + parity record (run first)
python scripts/orchestrate.py experiments/reference.yaml --subset full  # reference: source + TENT grid, FP32 deterministic, the reported accuracy
python scripts/orchestrate.py experiments/realtime.yaml   # whole dev stream per method at 30 fps, deploy mode (16, 1) at 30 to 720 fps
python scripts/orchestrate.py experiments/max_fps.yaml    # max sustainable fps per method, whole-dev-stream live run that held, trt_tent 3 repeats
python scripts/run.py --method trt_tent --mode deploy --B 16 --S 1 --fps 240   # one live run of one method, to /data/runs/dev/realtime/<method>/fps240_seed1/
python scripts/orchestrate.py experiments/trtexec.yaml    # GPU time (trtexec) and sizes of every engine
python scripts/orchestrate.py experiments/adapter.yaml    # standalone adapter step per B (m = B, graph)
python scripts/refit_bench.py                # refit cost of a constant-γ/β engine (ADR 0005)
python scripts/report.py --subset full       # main table: dev timing + full reference, to /data/runs/full/report/
```

| Output | Path |
| --- | --- |
| Engines + records | `/data/engines/` |
| Real-time runs | `/data/runs/<subset>/realtime/<method>/fps<f>_seed<n>/` |
| Max fps | `/data/runs/<subset>/max_fps/<method>/seed<n>/` |
| Reference, lag sim | `/data/runs/<subset>/offline/seed<n>/` |
| Benchmarks | `/data/runs/bench/` |
| Report | `/data/runs/<subset>/report/` |

## License

MIT. See [LICENSE](LICENSE).