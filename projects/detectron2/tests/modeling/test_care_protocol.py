import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = REPO_ROOT / "mask2former/care_protocol.py"


def load_protocol_module():
    spec = importlib.util.spec_from_file_location("care_protocol", PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCAREProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol_module()

    def _calibration(self):
        return {
            "source": self.protocol.RUNTIME_CALIBRATION_SOURCE,
            "dataset_role": "training_anchors",
            "dataset": "train",
            "descriptor_inputs": "frozen_teacher_predictions",
            "standardization_source": (
                self.protocol.RUNTIME_STANDARDIZATION_SOURCE
            ),
            "sample_count": 4,
            "quantile": self.protocol.RUNTIME_CALIBRATION_QUANTILE,
            "teacher_nn_distance_median": 0.5,
            "teacher_nn_distance_q75": 0.75,
            "compatibility_threshold": 0.75,
            "uses_validation_images": False,
            "uses_ground_truth": False,
        }

    def _phase0_summary(self):
        calibration = self._calibration()
        return {
            "audit_version": self.protocol.PHASE0_AUDIT_VERSION,
            "datasets": {"anchors": {"name": "train", "selected": 4}},
            "runtime_calibration": calibration,
            "donor_support": {
                "compatibility_threshold": calibration[
                    "compatibility_threshold"
                ],
                "compatibility_calibration_source": calibration["source"],
            },
        }

    def test_calibration_uses_only_cross_domain_training_anchors(self):
        rows = [
            {"domain_id": 0},
            {"domain_id": 0},
            {"domain_id": 1},
            {"domain_id": 1},
        ]
        signatures = np.asarray(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        calibration = self.protocol.calibrate_training_anchor_compatibility(
            rows,
            signatures,
            "train",
        )

        nearest = np.asarray([1.0, 0.5, 0.5, 2.5]) / np.sqrt(2.0)
        expected = np.quantile(
            nearest,
            self.protocol.RUNTIME_CALIBRATION_QUANTILE,
        )
        self.assertAlmostEqual(
            calibration["compatibility_threshold"],
            expected,
        )
        self.assertFalse(calibration["uses_validation_images"])
        self.assertFalse(calibration["uses_ground_truth"])

    def test_phase0_v1_is_rejected(self):
        summary = self._phase0_summary()
        summary["audit_version"] = 1
        with self.assertRaisesRegex(ValueError, "regenerate Phase 0"):
            self.protocol.validate_phase0_protocol(summary)

    def test_validation_transductive_calibration_is_rejected(self):
        summary = self._phase0_summary()
        summary["runtime_calibration"]["uses_validation_images"] = True
        with self.assertRaisesRegex(ValueError, "validation images"):
            self.protocol.validate_phase0_protocol(summary)

    def test_bank_must_carry_the_v2_phase0_protocol(self):
        manifest = {
            "bank_version": self.protocol.CARE_BANK_VERSION,
            "phase0_audit_version": self.protocol.PHASE0_AUDIT_VERSION,
            "runtime_calibration": copy.deepcopy(self._calibration()),
            "total_anchor_count": 4,
        }
        self.protocol.validate_bank_protocol(manifest)

        manifest["bank_version"] = 1
        with self.assertRaisesRegex(ValueError, "regenerate the bank"):
            self.protocol.validate_bank_protocol(manifest)


if __name__ == "__main__":
    unittest.main()
