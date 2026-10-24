# Edge TTA Deployment

Deploying a ViT classifier with test-time adaptation (TENT) on a single frame stream, measured on an RTX 3080 and designed for the Jetson Orin family.

## Adaptation

**Method**:
A deployable way to predict the frame stream: PyTorch source, PyTorch TENT (sync), TensorRT source, TensorRT TENT (decoupled adapter, spec or deploy mode). Every method implements the same `Method` protocol and runs through the same frame path, so methods compare directly.

**Adaptable parameters**:
The LayerNorm γ/β of the model (38,400 values). The only values TENT changes.

**Adaptation frame**:
A frame selected for adaptation by the stride S. Every frame is predicted, only adaptation frames feed the loss.

**Adaptation batch (B)**:
The number of adaptation frames that make up one step.

**Stride (S)**:
One of every S frames is an adaptation frame.

**Effective stride**:
The stride actually achieved after deploy mode drops adaptation frames.

**Micro-batch (m)**:
The number of adaptation frames the adapter runs forward and backward at once inside one step. B is a multiple of m. Bounds activation memory, never changes the step.

**Step**:
One optimizer update of the adaptable parameters.

**Lag (L)**:
The number of steps between the adaptable parameters a prediction uses and the newest ones.

**Lag trace**:
Per frame, the step whose adaptable parameters predicted it, as logged by the live system. The lag simulator replays it.

## System

**Frame path**:
The part of the system that turns each incoming frame into a prediction.

**Adapter**:
The part of the system that runs steps on adaptation batches and publishes the adaptable parameters to the frame path.

**Spec mode**:
Adapter mode that uses every adaptation frame exactly once, in arrival order, to match the reference.

**Deploy mode**:
Adapter mode with a bounded adaptation queue that drops the oldest adaptation batches when full.

**Source**:
The unadapted model.

## Evaluation

**Reference**:
The offline PyTorch FP32 deterministic implementation (`reference.py`) that every method computes at the same (B, S). Its accuracy is the reported accuracy of every method.

**Dev subset**:
A fixed 500-images-per-corruption sample of the evaluation subset, used during development for correctness checks only, never for accuracy claims.

**Final campaign**:
The runs that produce every reported number: timing on the dev subset with seed 1, the reference on the full subset with all seeds.

**Continual protocol**:
All 15 corruptions concatenated in the standard order, no reset. The only evaluation protocol.

**Event span**:
Time between two CUDA events on the frame stream, from the host-to-device copy of a frame to the device-to-host copy of its logits. It includes gaps where the GPU waits for the host to launch work, so it is not GPU busy time. GPU busy time comes from trtexec or Nsight.

**Keeps up**:
In spec mode, the adapter keeps up over a replay or live run when the steps finished by the last answered frame are at least 98% of the adaptation batches that arrived, minus the one in flight, and the frame path never blocked on full adaptation slots. A condition of max sustainable fps.

**Same-rate comparison**:
Every method replayed over the whole dev stream at one fixed input rate (30 fps by default), so latency and resources compare at equal load, and whether each method sustains that rate. Its accuracy is only the deployed-vs-reference check. The reported accuracy is the reference's.

**Collapsed config**:
A (B, S) whose reference average accuracy falls below source. Marked in the report, never dropped.
