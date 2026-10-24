# TF32 is off everywhere

The NGC PyTorch image sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which forces TF32 in every cuBLAS matmul. The Dockerfile sets it to 0, so precision is chosen in code. The reference and all equivalence tests run strict FP32 (`enable_determinism()`), and fps runs turn both TF32 flags off explicitly (`set_fps_precision()`). TF32 has about 1e-3 relative error per op, which would break the 1e-5 short-horizon tolerance. Off also matches JetPack's PyTorch wheels, which do not force TF32.

## Considered Options

- **TF32 on for fps runs:** a faster FP32 baseline, but it no longer shares numerics with the reference and the equivalence tests.
- **Rely on upstream defaults:** they differ (matmul off, cuDNN on), so they mix precisions.

## Consequences

- Every mode sets both TF32 flags explicitly.
- On the RTX 3080 the gain from TF32 is small (4096² matmul: 6.3 ms → 4.4 ms, max relative error 2.7e-6 → 3.0e-4), because consumer Ampere TF32 tensor throughput is close to its FP32 rate. Orin supports TF32 too.
- bf16 autocast in the adapter and the TensorRT engine (own builder flag) are unaffected. TensorRT parity is checked against strict FP32 PyTorch.
