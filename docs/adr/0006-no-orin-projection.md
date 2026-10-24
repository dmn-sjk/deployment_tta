# No Orin projection: every reported number is measured on the 3080

An earlier version projected Orin max fps as 3080 GPU time × 14 (Orin Nano) or × 6 (AGX Orin), with host time at its 3080 cost. The report and code no longer project (Oct 24, 2026). Orin stays the design target: the 8 GB memory floor, JetPack 6.2 portability and the Orin Nano risks still shape the design, but no Orin number is reported until Orin runs exist.

Reason: the final campaign showed that TensorRT TENT's max fps for B ≥ 16 is capped by the p99 of frames that overlap a step, not by GPU work per frame. That tail depends on how frame kernels interleave with step kernels on 68 SMs and varies by about 20% between processes. A GPU-time scale factor models neither, and Orin Nano's 8 SMs would change the interleaving itself. A projection would look precise and mean little.

## Considered Options

- **Keep the GPU-time projection, labelled as an upper bound:** cheap, but for the contention-bound configs the bound is loose by an unknown factor, and readers take projected numbers at face value.
- **Project from the step-in-flight split:** would need a model of contention on 8 SMs, which only Orin runs can calibrate.

## Consequences

- `ORIN_GPU_FACTORS` and the projected columns are gone from the code and the report.
- On-device Orin runs ([design.md](../design.md#future-work)) are the only route to Orin numbers.
