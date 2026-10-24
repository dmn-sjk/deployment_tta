import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def device_memory_gb() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "device_memory_used_gb": (total - free) / 1e9,
        "pytorch_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


@contextmanager
def gpu_sampler() -> Iterator[dict[str, float]]:
    """GPU utilisation, power, SM clock and temperature from nvidia-smi
    every 100 ms while the block runs."""
    stats: dict[str, float] = {}
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        yield stats
    finally:
        proc.terminate()
        samples = np.array(
            [
                [float(v) for v in line.split(",")]
                for line in proc.communicate()[0].splitlines()
                if line.strip()
            ]
        )
        stats |= {
            "gpu_utilisation_mean_pct": float(samples[:, 0].mean()),
            "power_mean_w": float(samples[:, 1].mean()),
            "power_max_w": float(samples[:, 1].max()),
            "sm_clock_mean_mhz": float(samples[:, 2].mean()),
            "sm_clock_min_mhz": float(samples[:, 2].min()),
            "temperature_max_c": float(samples[:, 3].max()),
        }
