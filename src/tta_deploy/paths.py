from pathlib import Path

from tta_deploy.reference import Precision
from tta_deploy.tent import TentConfig

RUNS_DIR = Path("/data/runs")


def realtime_dir(subset: str, name: str, fps: float, seed: int) -> Path:
    return RUNS_DIR / subset / "realtime" / name / f"fps{fps:g}_seed{seed}"


def max_fps_dir(subset: str, name: str, seed: int, repeat: int) -> Path:
    return RUNS_DIR / subset / "max_fps" / name / f"seed{seed}_repeat{repeat}"


def max_fps_repeat_dirs(subset: str, name: str, seed: int) -> list[Path]:
    return sorted((RUNS_DIR / subset / "max_fps" / name).glob(f"seed{seed}_repeat*"))


def offline_dir(subset: str, seed: int) -> Path:
    return RUNS_DIR / subset / "offline" / f"seed{seed}"


def reference_set(precision: Precision, rerun: bool) -> str:
    return f"reference_{precision}{'_rerun' if rerun else ''}"


def reference_dir(subset: str, seed: int, precision: Precision, rerun: bool) -> Path:
    return offline_dir(subset, seed) / reference_set(precision, rerun)


def lag_sim_dir(subset: str, seed: int, tent: TentConfig) -> Path:
    return offline_dir(subset, seed) / "lag_sim" / tent.name


def bench_dir() -> Path:
    return RUNS_DIR / "bench"


def trtexec_bench_path(engine: str) -> Path:
    return bench_dir() / "trtexec" / f"{engine}.json"


def adapter_bench_path(name: str) -> Path:
    return bench_dir() / "adapter" / f"{name}.json"


def refit_bench_path() -> Path:
    return bench_dir() / "refit.json"


def fault_dir(subset: str) -> Path:
    return bench_dir() / "fault" / subset


def report_dir(subset: str) -> Path:
    return RUNS_DIR / subset / "report"
