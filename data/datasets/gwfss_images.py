# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import os

from .. import DatasetCatalog, MetadataCatalog
from .gwfss_semantic import GWFSS_CATEGORIES
from .gwfss_domains import infer_gwfss_domain
from detectron2.utils.file_io import PathManager
# from projects.GeneSSIS.data_perc import _RAW_CITYSCAPES_PANOPTIC_SPLITS
"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _normalise_manifest_path(path):
    # The released list replaces spaces around the dash in "Uliege - CRA-W".
    return path.replace("_-_", " - ")


def _select_domain_balanced(
    image_files,
    image_dir,
    samples_per_domain,
    seed,
):
    files_by_domain = {}
    for image_file in image_files:
        domain = os.path.basename(os.path.dirname(image_file))
        files_by_domain.setdefault(domain, []).append(image_file)

    selected = []
    for domain in sorted(files_by_domain):
        domain_files = files_by_domain[domain]
        if len(domain_files) < samples_per_domain:
            raise ValueError(
                "{} contains only {} images; {} requested".format(
                    domain, len(domain_files), samples_per_domain
                )
            )
        ranked = sorted(
            domain_files,
            key=lambda path: hashlib.sha256(
                "{}:{}".format(seed, os.path.relpath(path, image_dir)).encode("utf-8")
            ).digest(),
        )
        selected.extend(ranked[:samples_per_domain])
    return selected


def get_gwfss_unlabel_files(
    image_dir,
    manifest_file=None,
    samples_per_domain=None,
    seed=0,
):
    if manifest_file:
        with PathManager.open(manifest_file, "r") as handle:
            relative_paths = [
                _normalise_manifest_path(line.strip())
                for line in handle
                if line.strip()
            ]
        image_files = [os.path.join(image_dir, path) for path in relative_paths]
    else:
        image_files = []
        for current_dir, _, basenames in os.walk(image_dir):
            for basename in basenames:
                if os.path.splitext(basename)[1].lower() in _IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(current_dir, basename))
        image_files.sort()
        if samples_per_domain is not None:
            image_files = _select_domain_balanced(
                image_files,
                image_dir,
                samples_per_domain,
                seed,
            )

    assert image_files, "No images found in {}".format(image_dir)
    missing = [path for path in image_files if not PathManager.isfile(path)]
    assert not missing, "Missing unlabeled image: {}".format(missing[0])
    return [(path,) for path in image_files]


def load_gwfss_unlabel(
    image_dir,
    manifest_file=None,
    samples_per_domain=None,
    seed=0,
):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        manifest_file (str, optional): file containing paths relative to
            ``image_dir``. If omitted, all images below ``image_dir`` are used.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    files = get_gwfss_unlabel_files(
        image_dir,
        manifest_file,
        samples_per_domain,
        seed,
    )
    ret = []
    for image_file, in files:
        relative_stem = os.path.splitext(os.path.relpath(image_file, image_dir))[0]
        domain_id, domain_name = infer_gwfss_domain(image_file)
        ret.append(
            {
                "file_name": image_file,
                "image_id": relative_stem.replace(os.sep, "__"),
                "domain_id": domain_id,
                "domain_name": domain_name,
                # "sem_seg_file_name": '',
                # "pan_seg_file_name": '',
                # "segments_info": {'s':[]},
            }
        )
    assert len(ret), f"No images found in {image_dir} (unlabel)!"
    return ret


_RAW_GWFSS_UNLABELED_SPLITS = {
    "gwfss_unlabel_stem4500": ("GWFSS", "unlabeled_4500.txt", None, 0),
    "gwfss_unlabel_random4500_seed2025": (
        "GWFSS/gwfss_competition_pretrain", None, 500, 2025,
    ),
    "gwfss_unlabel_all": (
        "GWFSS/gwfss_competition_pretrain", None, None, 0,
    ),
    # Compatibility alias. This is the prior method's stem-aware selection,
    # not a neutral 4,500-image subset.
    "gwfss_unlabel_4500": ("GWFSS", "unlabeled_4500.txt", None, 0),
    # Compatibility alias used by the released training code.
    "gwfss_unlabel_train": ("GWFSS", "unlabeled_4500.txt", None, 0),
}


def register_all_gwfss_unlabel(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in GWFSS_CATEGORIES]
    thing_colors = [k["color"] for k in GWFSS_CATEGORIES]
    stuff_classes = [k["name"] for k in GWFSS_CATEGORIES]
    stuff_colors = [k["color"] for k in GWFSS_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in GWFSS_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (
        image_dir,
        manifest_file,
        samples_per_domain,
        seed,
    ) in _RAW_GWFSS_UNLABELED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        if manifest_file:
            manifest_file = os.path.join(root, manifest_file)

        DatasetCatalog.register(
            key,
            lambda x=image_dir, y=manifest_file, n=samples_per_domain, s=seed:
                load_gwfss_unlabel(x, y, n, s),
        )
        MetadataCatalog.get(key).set(
            image_root=image_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )
