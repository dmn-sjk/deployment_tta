# LayerNorm γ/β are engine inputs, not refit weights

The frame-path engine takes the 50 LayerNorm γ/β tensors (38,400 FP32 values) as graph inputs next to the image (`export/onnx_export.py`, `ExportConfig.layernorm_inputs`). `TrtEngine` keeps two packed γ/β buffers, each bound once with `set_tensor_address` to its own execution context. The adapter writes the back buffer and publishes it; the next frame runs on the context bound to it. A publish moves no weights and does not touch the engine.

γ/β as inputs cost no measurable frame-path time. Measured with trtexec on the RTX 3080 (GPU median, FP32 LayerNorm, FP32 softmax):

| run | constant γ/β | γ/β inputs | same inputs engine, built by trtexec |
|---|---|---|---|
| Oct 21, 2026 | 1.092 ms | 1.084 ms | 1.088 ms |
| Oct 23, 2026 | 1.271 ms | 1.312 ms | 1.264 ms |
| Oct 23, 2026, rerun | 1.167 ms | 1.100 ms | 1.202 ms |

The difference between constant and input γ/β is within the spread of one engine across builds, and changes sign between runs, so no fusion is lost that matters. The absolute shift between the two runs affects every variant alike and is not explained by the γ/β choice.

## Considered Options

- **Refittable engine with constant γ/β:** measured by `refit_bench.py` (Oct 23, 2026, `/data/runs/bench/refit.json`): 0.56 ms host + 0.45 ms GPU per update of all 50 tensors, and a 176 MB refittable engine. A refit must not run while a frame is in flight on that engine, so a stall-free frame path needs two engines and a swap, which doubles engine memory on an 8 GB Orin. The GPU part alone is about a third of a frame per adaptation step.
- **Decompose LayerNorm into Mul/Add on γ/β inputs:** the fallback if inputs had lost a fusion. Not needed, see the measurement above.

## Consequences

- The adapter and the frame path share only the two γ/β buffers and their events (`publish`, `release_event`), which is what keeps the adapter decoupled.
- The constant-γ/β export (`ExportConfig(layernorm_inputs=False)`) stays only as a comparison variant and as the source of the refit benchmark. It cannot run adapting methods (`methods/configs.py` asserts this).
- Revisit if the adapted parameters change to weights that TensorRT fuses into matmuls (e.g. attention or MLP weights), where inputs would restrict tactics and refit may win.
