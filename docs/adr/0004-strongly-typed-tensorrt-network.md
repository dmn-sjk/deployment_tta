# TensorRT precision comes from a strongly typed network, not layer precision constraints

The frame-path engine is built as a strongly typed TensorRT network from an ONNX graph whose tensor types fix every precision: FP16 weights and activations, an FP32 image input and FP32 logits, each LayerNorm computed in FP32 from FP32 γ/β inputs (`Cast` to FP32 before, back to FP16 after), and each attention softmax in FP32 (same `Cast` pair, added to the ONNX graph after export). The builder gets no FP16 flag and no per-layer precision.

The original route, a weakly typed network with the FP16 flag, `layer.precision = FP32` on the 25 normalization layers and their γ/β broadcast shuffles, and `OBEY_PRECISION_CONSTRAINTS`, does not work on TensorRT 10.3 for this model. Myelin compiles the whole encoder into one foreign node, ignores the layer constraints, and inserts a reformat that casts every γ/β input from FP32 to FP16. Measured: a γ/β change that rounds to the same FP16 values left the logits bit-identical (max |Δ| = 0), exactly like an FP16-LayerNorm engine. In the strongly typed engine the same change moves the logits (max |Δ| = 7.3e-3). `tests/test_trt.py` keeps both sides of this check.

## Considered Options

- **Weakly typed + layer precision constraints:** the original design. Silently ignored by Myelin, see above.
- **Decompose LayerNorm into Mul/Add on γ/β inputs:** not needed once types are explicit, and more graph surgery.
- **Refit with constant γ/β:** two engines and a swap, doubles engine memory on an 8 GB device.

## Consequences

- Precision changes are made in the ONNX export (`export/onnx_export.py`), not in the engine builder. The FP16-LayerNorm comparison variant casts γ/β to FP16 inside the graph.
- Strongly typed networks are supported by TensorRT 10.3 on JetPack 6.2, so the route is portable. `trtexec --stronglyTyped` builds the same engine.
- Measured on the RTX 3080, γ/β as inputs, FP32 LayerNorm and FP32 softmax cost at most about 1% (all five variants at 1.08 to 1.10 ms GPU median).
- FP32 softmax follows the drift rule: with FP16 softmax, fog agreement fell below bf16 autocast's on the dev subset.
