import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
TOPOLOGY_PATH = REPO_ROOT / "mask2former/topowheat/topology.py"


def load_topology_module():
    spec = importlib.util.spec_from_file_location(
        "topowheat_topology",
        TOPOLOGY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTRPL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = load_topology_module()

    @staticmethod
    def teacher_view():
        probability = torch.full((1, 4, 9, 9), 0.02)
        probability[:, 0] = 0.94
        probability[:, 0, :, 4] = 0.04
        probability[:, 2, :, 4] = 0.92
        return probability

    def test_agreed_views_produce_reliable_region_and_centerline(self):
        view = self.teacher_view()
        target = self.topology.build_trpl_targets(
            [view, view.clone()],
            reliability_threshold=0.75,
            stem_class=2,
        )

        self.assertTrue(target["reliable"].all())
        self.assertGreater(int(target["stable_skeleton"].sum()), 0)
        self.assertTrue(
            target["labels"][target["stable_skeleton"]].eq(2).all()
        )

    def test_view_disagreement_rejects_pixel(self):
        first = self.teacher_view()
        second = first.clone()
        second[:, :, 4, 4] = torch.tensor([0.02, 0.02, 0.02, 0.94])
        target = self.topology.build_trpl_targets(
            [first, second],
            reliability_threshold=0.75,
            stem_class=2,
        )

        self.assertFalse(bool(target["reliable"][0, 4, 4]))
        self.assertGreater(float(target["view_disagreement"][0, 4, 4]), 0.0)

    def test_consistency_and_centerline_losses_have_expected_direction(self):
        teacher = self.teacher_view()
        labels = teacher.argmax(dim=1)
        valid = torch.ones_like(labels, dtype=torch.bool)
        weights = torch.ones_like(labels, dtype=torch.float32)
        matched = self.topology.class_balanced_consistency_loss(
            teacher,
            teacher,
            labels,
            weights,
            valid,
        )
        mismatched = self.topology.class_balanced_consistency_loss(
            teacher.roll(shifts=1, dims=1),
            teacher,
            labels,
            weights,
            valid,
        )
        self.assertAlmostEqual(float(matched), 0.0, places=6)
        self.assertGreater(float(mismatched), float(matched))

        centerline = labels.eq(2).unsqueeze(1).float()
        aligned = 0.05 + 0.90 * centerline
        missing = torch.full_like(aligned, 0.05)
        aligned_loss = self.topology.stable_centerline_loss(
            aligned,
            centerline,
        )
        missing_loss = self.topology.stable_centerline_loss(
            missing,
            centerline,
        )
        self.assertLess(float(aligned_loss), float(missing_loss))

    def test_stable_centerline_bypasses_region_gate_for_tcpm_core(self):
        labels = torch.zeros((1, 9, 9), dtype=torch.long)
        labels[:, :, 4] = 2
        reliable = torch.zeros_like(labels, dtype=torch.bool)
        stable = labels.eq(2)
        core = self.topology.build_core_mask(
            labels,
            reliable,
            num_classes=4,
            stem_class=2,
            stable_skeleton=stable,
            strategy="topology",
            stem_radius=1,
        )

        self.assertTrue(core[stable].all())
        self.assertFalse(core[labels.ne(2)].any())


if __name__ == "__main__":
    unittest.main()
