import subprocess
import sys
import time
from pathlib import Path

# Spacing between measurements, so the previous run's GPU work has
# drained before the next starts.
COOLDOWN_SECONDS = 5.0


def run_script(script: str | Path, *argv: str) -> None:
    """One measurement in its own process, so device memory is never
    mixed between runs."""
    time.sleep(COOLDOWN_SECONDS)
    command = [sys.executable, str(script), *argv]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
