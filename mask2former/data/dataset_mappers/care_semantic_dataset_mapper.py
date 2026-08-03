"""Semantic mapper that attaches precomputed CARE donor statistics."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import get_worker_info

from detectron2.config import configurable

from data import DatasetCatalog
from ...care_protocol import validate_bank_protocol

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

__all__ = ["CARESemanticDatasetMapper"]


class CARESemanticDatasetMapper:
    """Run the baseline mapper unchanged, then attach one compatible donor."""

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        base_mapper,
        bank_dir,
        feature_name,
        seed,
        training_anchor_keys,
    ):
        if not is_train:
            raise ValueError("CARE mapper is only valid for training")
        self.base_mapper = base_mapper
        self.bank_dir = Path(bank_dir)
        self.feature_name = str(feature_name)
        self.seed = int(seed)
        self._worker_seed = None
        self._rng = None

        manifest_path = self.bank_dir / "manifest.json"
        bank_path = self.bank_dir / "feature_bank.npz"
        if not manifest_path.is_file() or not bank_path.is_file():
            raise FileNotFoundError(
                "CARE bank requires manifest.json and feature_bank.npz in {}".format(
                    self.bank_dir
                )
            )

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_bank_protocol(manifest)
        if manifest.get("phase0_verdict") != "care_phase0_supported":
            raise ValueError("CARE bank was not produced by a supported Phase-0 audit")
        if manifest.get("feature_name") != self.feature_name:
            raise ValueError(
                "CARE bank feature {} does not match configured feature {}".format(
                    manifest.get("feature_name"),
                    self.feature_name,
                )
            )

        with np.load(str(bank_path), allow_pickle=False) as payload:
            self.donor_ids = [str(value) for value in payload["donor_ids"]]
            self.donor_mean = np.asarray(payload["mean"], dtype=np.float32)
            self.donor_std = np.asarray(payload["std"], dtype=np.float32)

        expected_shape = (len(self.donor_ids), int(manifest["channels"]))
        if self.donor_mean.shape != expected_shape:
            raise ValueError("CARE donor mean bank has an unexpected shape")
        if self.donor_std.shape != expected_shape:
            raise ValueError("CARE donor std bank has an unexpected shape")
        if not np.isfinite(self.donor_mean).all():
            raise ValueError("CARE donor mean bank contains non-finite values")
        if not np.isfinite(self.donor_std).all() or np.any(self.donor_std < 0):
            raise ValueError("CARE donor std bank is invalid")

        manifest_donor_ids = [row["donor_id"] for row in manifest["donors"]]
        if manifest_donor_ids != self.donor_ids:
            raise ValueError("CARE donor ordering differs between manifest and bank")

        self.anchor_choices = manifest["anchors"]
        known_anchors = set(training_anchor_keys)
        unknown_anchors = sorted(set(self.anchor_choices) - known_anchors)
        if unknown_anchors:
            raise ValueError(
                "CARE manifest contains an unknown training anchor: {}".format(
                    unknown_anchors[0]
                )
            )
        if not self.anchor_choices:
            raise ValueError("CARE manifest contains no supported anchors")

        for anchor_key, choices in self.anchor_choices.items():
            if not choices:
                raise ValueError("CARE anchor {} has no donor choices".format(anchor_key))
            for choice in choices:
                donor_index = int(choice["donor_index"])
                weight = float(choice["compatibility_weight"])
                if not 0 <= donor_index < len(self.donor_ids):
                    raise ValueError("CARE donor index is out of range")
                if not 0.0 < weight <= 1.0:
                    raise ValueError("CARE compatibility weight must be in (0, 1]")

        logging.getLogger(__name__).info(
            "CARE bank loaded: %d/%d supported anchors, %d donors, feature %s",
            len(self.anchor_choices),
            len(known_anchors),
            len(self.donor_ids),
            self.feature_name,
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        training_anchor_keys = sorted(
            {Path(record["file_name"]).name for record in records}
        )
        return {
            "is_train": is_train,
            "base_mapper": MaskFormerSemanticDatasetMapper(cfg, is_train),
            "bank_dir": cfg.MODEL.CARE.BANK_DIR,
            "feature_name": cfg.MODEL.CARE.FEATURE_NAME,
            "seed": cfg.SEED,
            "training_anchor_keys": training_anchor_keys,
        }

    def _local_rng(self):
        worker = get_worker_info()
        worker_seed = self.seed if worker is None else int(worker.seed)
        worker_seed ^= 0x43415245
        if self._rng is None or self._worker_seed != worker_seed:
            self._worker_seed = worker_seed
            self._rng = random.Random(worker_seed)
        return self._rng

    def __call__(self, dataset_dict):
        anchor_key = Path(dataset_dict["file_name"]).name
        output = self.base_mapper(dataset_dict)
        choices = self.anchor_choices.get(anchor_key)
        if not choices:
            return output

        choice = choices[self._local_rng().randrange(len(choices))]
        donor_index = int(choice["donor_index"])
        output["care_donor_mean"] = torch.tensor(
            self.donor_mean[donor_index],
            dtype=torch.float32,
        )
        output["care_donor_std"] = torch.tensor(
            self.donor_std[donor_index],
            dtype=torch.float32,
        )
        output["care_weight"] = torch.tensor(
            float(choice["compatibility_weight"]),
            dtype=torch.float32,
        )
        output["care_donor_id"] = self.donor_ids[donor_index]
        return output
