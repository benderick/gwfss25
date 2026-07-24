#!/usr/bin/env python3
"""Validate GWFSS files and convert full-dataset RGB masks to class IDs."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (50, 255, 132): 1,
    (50, 132, 255): 2,
    (214, 255, 50): 3,
}

REGION_SPLITS = {
    "train": ("Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege_CRA-W"),
    "val": ("UTokyo",),
    "test": ("UQ_new",),
}

COMPETITION_COUNTS = {
    "gwfss_competition_train": 99,
    "gwfss_competition_val": 99,
    "gwfss_competition_test": 110,
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("DETECTRON2_DATASETS", repo_root)),
        help="Dataset root containing the GWFSS directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate existing files without writing converted masks.",
    )
    return parser.parse_args()


def png_names(directory):
    return {path.name for path in directory.glob("*.png") if path.is_file()}


def validate_competition_data(dataset_root):
    for split, expected in COMPETITION_COUNTS.items():
        split_root = dataset_root / split
        image_names = png_names(split_root / "images")
        label_names = png_names(split_root / "class_id")
        if image_names != label_names:
            raise RuntimeError(
                "{} has {} images but {} matching class-ID masks".format(
                    split, len(image_names), len(image_names & label_names)
                )
            )
        if len(image_names) != expected:
            raise RuntimeError(
                "{}: expected {} samples, found {}".format(
                    split, expected, len(image_names)
                )
            )


def validate_unlabeled_manifest(root, dataset_root):
    manifest = root / "unlabeled_4500.txt"
    relative_paths = [
        line.strip().replace("_-_", " - ")
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
    missing = [path for path in relative_paths if not (dataset_root / path).is_file()]
    if len(relative_paths) != 4500 or len(set(relative_paths)) != 4500:
        raise RuntimeError("unlabeled_4500.txt must contain 4,500 unique paths")
    if missing:
        raise RuntimeError("Missing unlabeled image: {}".format(missing[0]))


def convert_mask(source, destination):
    rgb = np.asarray(Image.open(source).convert("RGB"))
    class_ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in COLOR_TO_CLASS.items():
        class_ids[np.all(rgb == color, axis=-1)] = class_id

    if np.any(class_ids == 255):
        unknown = np.unique(rgb[class_ids == 255].reshape(-1, 3), axis=0)
        raise RuntimeError(
            "{} contains unknown mask colors: {}".format(source, unknown.tolist())
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    Image.fromarray(class_ids, mode="L").save(temporary, format="PNG")
    temporary.replace(destination)


def prepare_full_data(dataset_root, overwrite, check_only):
    full_root = dataset_root / "GWFSS_v1.0_labelled"
    image_root = full_root / "images"
    mask_root = full_root / "masks"
    class_id_root = full_root / "class_id"

    total = 0
    for domain_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        domain = domain_dir.name
        image_names = png_names(domain_dir)
        mask_names = png_names(mask_root / domain)
        if image_names != mask_names:
            raise RuntimeError(
                "{} has {} images and {} masks".format(
                    domain, len(image_names), len(mask_names)
                )
            )
        total += len(image_names)

        if check_only:
            continue
        for name in sorted(mask_names):
            destination = class_id_root / domain / name
            if overwrite or not destination.is_file():
                convert_mask(mask_root / domain / name, destination)

    if total != 1096:
        raise RuntimeError("Expected 1,096 full-dataset samples, found {}".format(total))

    if not check_only:
        converted = sum(1 for _ in class_id_root.glob("*/*.png"))
        if converted != total:
            raise RuntimeError(
                "Expected {} converted masks, found {}".format(total, converted)
            )

    for split, domains in REGION_SPLITS.items():
        count = sum(len(png_names(image_root / domain)) for domain in domains)
        print("{}: {} images ({})".format(split, count, ", ".join(domains)))
    print("USASK: 110 images (unused by the dataset paper's region split)")


def main():
    args = parse_args()
    root = args.root.resolve()
    dataset_root = root / "GWFSS"
    validate_competition_data(dataset_root)
    validate_unlabeled_manifest(root, dataset_root)
    prepare_full_data(dataset_root, args.overwrite, args.check_only)
    print("GWFSS data validation complete.")


if __name__ == "__main__":
    main()
