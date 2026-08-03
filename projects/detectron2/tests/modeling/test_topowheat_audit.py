import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_PATH = REPO_ROOT / "mask2former/topowheat/audit.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "topowheat_audit",
        AUDIT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTRPLAuditMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_audit_module()

    def test_matched_topk_mask_has_exact_reference_coverage(self):
        scores = torch.tensor([[0.1, 0.8], [0.7, 0.4]])
        reference = torch.tensor([[True, False], [False, True]])
        valid = torch.tensor([[True, True], [True, False]])
        selected = self.audit.matched_topk_mask(scores, reference, valid)

        self.assertEqual(int(selected.sum()), 1)
        self.assertTrue(bool(selected[0, 1]))
        self.assertFalse(bool(selected[1, 1]))

    def test_class_matched_topk_preserves_each_predicted_class_count(self):
        scores = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
        labels = torch.tensor([[0, 0], [1, 1]])
        reference = torch.tensor([[False, True], [True, False]])
        selected = self.audit.matched_topk_mask_by_class(
            scores,
            labels,
            reference,
            num_classes=2,
        )

        self.assertEqual(int((selected & labels.eq(0)).sum()), 1)
        self.assertEqual(int((selected & labels.eq(1)).sum()), 1)
        self.assertTrue(bool(selected[0, 0]))
        self.assertTrue(bool(selected[1, 0]))

    def test_selective_metrics_count_unselected_targets_as_false_negatives(self):
        accumulator = self.audit.SelectiveSegmentationAccumulator(
            2,
            class_names=("background", "stem"),
        )
        prediction = torch.tensor([[0, 1], [1, 0]])
        target = torch.tensor([[0, 0], [1, 1]])
        selected = torch.tensor([[True, False], [True, False]])
        accumulator.update(prediction, target, selected)
        summary = accumulator.summary()

        self.assertEqual(summary["valid_pixels"], 4)
        self.assertEqual(summary["accepted_pixels"], 2)
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["accepted_accuracy"], 1.0)
        self.assertAlmostEqual(
            summary["per_class"]["stem"]["precision"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["per_class"]["stem"]["recall"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["per_class"]["stem"]["iou"],
            0.5,
        )

    def test_calibration_metrics(self):
        accumulator = self.audit.CalibrationAccumulator(num_bins=10)
        scores = torch.tensor([0.9, 0.1])
        correct = torch.tensor([True, False])
        accumulator.update(scores, correct)
        summary = accumulator.summary()

        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["mean_score"], 0.5)
        self.assertAlmostEqual(summary["ece"], 0.1, places=6)
        self.assertAlmostEqual(summary["brier"], 0.01, places=6)

    def test_topology_alignment_rewards_supported_centerline(self):
        target_stem = torch.zeros((9, 9), dtype=torch.bool)
        target_stem[:, 3:6] = True
        target_skeleton = torch.zeros_like(target_stem)
        target_skeleton[:, 4] = True
        candidate = target_skeleton.clone()

        accumulator = self.audit.TopologyAlignmentAccumulator(tolerance=1)
        accumulator.update(candidate, target_stem, target_skeleton)
        summary = accumulator.summary()

        self.assertAlmostEqual(summary["precision"], 1.0)
        self.assertAlmostEqual(summary["sensitivity"], 1.0)
        self.assertAlmostEqual(summary["cldice"], 1.0)


if __name__ == "__main__":
    unittest.main()
