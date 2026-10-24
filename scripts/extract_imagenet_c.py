"""Extract the RobustBench severity-5 subset from ImageNet-C family
tars, then write the full and dev manifests once all 15 corruptions are
present."""

import argparse
import json
from pathlib import Path

from tta_deploy.data.imagenet_c import (
    CORRUPTIONS,
    DEV_MANIFEST,
    FULL_MANIFEST,
    IMAGENET_C_DIR,
    dev_manifest_frames,
    extract_subset,
    frame_path,
    full_manifest_frames,
    robustbench_image_ids,
    verify_frames,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tars",
        nargs="*",
        type=Path,
        help="family tars, e.g. /data/downloads/imagenet-c/blur.tar",
    )
    args = parser.parse_args()

    image_ids = robustbench_image_ids()
    for tar_path in args.tars:
        hashes = extract_subset(tar_path, IMAGENET_C_DIR, image_ids)
        corruptions = [c for c in CORRUPTIONS if frame_path(c, image_ids[0]) in hashes]
        assert len(hashes) == len(image_ids) * len(corruptions), (
            f"{tar_path}: {len(hashes)} images for {corruptions}"
        )
        (IMAGENET_C_DIR / f"sha256_{tar_path.stem}.json").write_text(
            json.dumps(hashes, sort_keys=True)
        )
        print(f"{tar_path.name}: {len(hashes)} images, {corruptions}")

    hashes = {}
    for path in IMAGENET_C_DIR.glob("sha256_*.json"):
        hashes |= json.loads(path.read_text())
    missing = [c for c in CORRUPTIONS if frame_path(c, image_ids[0]) not in hashes]
    if missing:
        print(f"manifests not written, corruptions missing: {missing}")
        return
    full = full_manifest_frames(hashes)
    verify_frames(full)
    dev = dev_manifest_frames(full)
    assert len(full) == 75_000 and len(dev) == 7_500
    write_manifest(FULL_MANIFEST, full)
    write_manifest(DEV_MANIFEST, dev)
    print(f"wrote {FULL_MANIFEST} ({len(full)}) and {DEV_MANIFEST} ({len(dev)})")


if __name__ == "__main__":
    main()
