import csv
import hashlib
import json
import random
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from tta_deploy.model import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
SEVERITY = 5
IMAGENET_C_DIR = Path("/data/datasets/imagenet-c")
MANIFEST_DIR = Path(__file__).resolve().parents[3] / "manifests"
FULL_MANIFEST = MANIFEST_DIR / "imagenet_c_s5_full.csv"
DEV_MANIFEST = MANIFEST_DIR / "imagenet_c_s5_dev.csv"
DEV_IMAGES_PER_CORRUPTION = 500
DEV_SELECTION_SEED = 0

_LISTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Frame:
    path: str
    label: int
    sha256: str

    @property
    def corruption(self) -> str:
        return self.path.split("/")[0]

    @property
    def image_id(self) -> str:
        return self.path.split("/", 2)[2]


def frame_path(corruption: str, image_id: str) -> str:
    return f"{corruption}/{SEVERITY}/{image_id}"


def robustbench_image_ids() -> list[str]:
    ids = (_LISTS_DIR / "robustbench_imagenet_test_ids.txt").read_text().split()
    assert len(ids) == 5000 and len(set(ids)) == 5000
    return ids


def class_to_label() -> dict[str, int]:
    mapping = json.loads(
        (_LISTS_DIR / "robustbench_imagenet_class_to_id.json").read_text()
    )
    assert list(mapping.values()) == list(range(NUM_CLASSES))
    return mapping


def extract_subset(
    tar_path: Path, out_dir: Path, image_ids: list[str]
) -> dict[str, str]:
    """Stream one ImageNet-C family tar once and write only the
    severity-5 subset images.

    Returns path -> sha256.
    """
    wanted = {frame_path(c, i) for c in CORRUPTIONS for i in image_ids}
    hashes = {}
    # The Zenodo .tar files are gzip-compressed.
    with tarfile.open(tar_path, "r|*") as tar:
        for member in tar:
            if not member.isfile() or member.name not in wanted:
                continue
            file = tar.extractfile(member)
            assert file is not None
            data = file.read()
            target = out_dir / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            hashes[member.name] = hashlib.sha256(data).hexdigest()
    return hashes


def full_manifest_frames(hashes: dict[str, str]) -> list[Frame]:
    labels = class_to_label()
    image_ids = robustbench_image_ids()
    frames = []
    for corruption in CORRUPTIONS:
        for image_id in image_ids:
            path = frame_path(corruption, image_id)
            frames.append(Frame(path, labels[image_id.split("/")[0]], hashes[path]))
    return frames


def dev_manifest_frames(full: list[Frame]) -> list[Frame]:
    dev_ids = set(
        random.Random(DEV_SELECTION_SEED).sample(
            robustbench_image_ids(), DEV_IMAGES_PER_CORRUPTION
        )
    )
    return [f for f in full if f.image_id in dev_ids]


def write_manifest(path: Path, frames: list[Frame]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "sha256"])
        writer.writerows((fr.path, fr.label, fr.sha256) for fr in frames)


def read_manifest(path: Path) -> list[Frame]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    frames = [Frame(r["path"], int(r["label"]), r["sha256"]) for r in rows]
    assert [f.corruption for f in frames[:: len(frames) // len(CORRUPTIONS)]] == list(
        CORRUPTIONS
    )
    return frames


def verify_frames(frames: list[Frame], root: Path = IMAGENET_C_DIR) -> None:
    for frame in frames:
        digest = hashlib.sha256((root / frame.path).read_bytes()).hexdigest()
        assert digest == frame.sha256, f"sha256 mismatch: {frame.path}"


def continual_stream(frames: list[Frame], seed: int) -> list[Frame]:
    """Corruptions in the standard order, no reset, shuffled within each
    corruption as ROID does (random.seed + random.shuffle per
    corruption)."""
    stream = []
    for corruption in CORRUPTIONS:
        block = [f for f in frames if f.corruption == corruption]
        assert block, f"no frames for {corruption}"
        random.Random(seed).shuffle(block)
        stream += block
    return stream


def verified_stream(subset: str, seed: int) -> list[Frame]:
    assert subset in ("dev", "full")
    frames = read_manifest(DEV_MANIFEST if subset == "dev" else FULL_MANIFEST)
    verify_frames(frames)
    return continual_stream(frames, seed)


def stream_labels(stream: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([f.label for f in stream]), np.array([f.corruption for f in stream])


def preprocess(image: Image.Image) -> torch.Tensor:
    # ImageNet-C images are already resized and center-cropped to
    # 224x224 before corruption.
    assert image.size == (224, 224), image.size
    return F.normalize(F.to_tensor(image.convert("RGB")), IMAGENET_MEAN, IMAGENET_STD)


def decode(image: Image.Image) -> torch.Tensor:
    """Uint8 HWC RGB, the decoded frame the frame path receives in host
    memory."""
    assert image.size == (224, 224), image.size
    return torch.from_numpy(np.array(image.convert("RGB")))


def per_corruption_sample(
    stream: list[Frame], frames_per_corruption: int
) -> list[Frame]:
    """The first frames of each corruption block of a continual
    stream."""
    block = len(stream) // len(CORRUPTIONS)
    return [
        stream[c * block + i]
        for c in range(len(CORRUPTIONS))
        for i in range(frames_per_corruption)
    ]


class FrameDataset(Dataset):
    def __init__(
        self,
        stream: list[Frame],
        root: Path = IMAGENET_C_DIR,
        transform: Callable[[Image.Image], torch.Tensor] = preprocess,
    ):
        self.stream = stream
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.stream)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frame = self.stream[index]
        with Image.open(self.root / frame.path) as image:
            return self.transform(image), frame.label


def decoded_frames(stream: list[Frame]) -> list[torch.Tensor]:
    # Decoded up front, so a replay never waits on decoding.
    return [
        frame
        for frame, _ in DataLoader(
            FrameDataset(stream, transform=decode),
            batch_size=None,
            num_workers=8,
        )
    ]
