import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
CARE_PATH = REPO_ROOT / "mask2former/care.py"


def load_care_module():
    spec = importlib.util.spec_from_file_location("care_statistics", CARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCARE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.care = load_care_module()

    def test_zero_weight_keeps_unsupported_anchor_exactly(self):
        feature = torch.randn(2, 3, 7, 5)
        donor_mean = torch.randn(2, 3)
        donor_std = torch.rand(2, 3) + 0.2
        output = self.care.interpolate_feature_statistics(
            feature,
            donor_mean,
            donor_std,
            torch.zeros(2),
        )
        self.assertTrue(torch.equal(feature, output))

    def test_unit_weight_matches_donor_channel_statistics(self):
        feature = torch.randn(2, 3, 11, 9)
        donor_mean = torch.tensor([[0.2, -0.5, 1.1], [-0.3, 0.7, 0.4]])
        donor_std = torch.tensor([[0.4, 1.3, 0.8], [1.1, 0.6, 1.7]])
        output = self.care.interpolate_feature_statistics(
            feature,
            donor_mean,
            donor_std,
            torch.ones(2),
        )
        output_mean = output.mean(dim=(2, 3))
        output_std = (
            (output - output_mean[:, :, None, None])
            .square()
            .mean(dim=(2, 3))
            .sqrt()
        )
        self.assertTrue(torch.allclose(output_mean, donor_mean, atol=1.0e-5))
        self.assertTrue(torch.allclose(output_std, donor_std, atol=1.0e-5))

    def test_partial_weight_interpolates_statistics(self):
        feature = torch.randn(1, 2, 8, 8)
        source_mean = feature.mean(dim=(2, 3))
        source_std = (
            (feature - source_mean[:, :, None, None])
            .square()
            .mean(dim=(2, 3))
            .add(1.0e-6)
            .sqrt()
        )
        donor_mean = torch.tensor([[2.0, -1.0]])
        donor_std = torch.tensor([[0.5, 1.5]])
        weight = torch.tensor([0.25])
        output = self.care.interpolate_feature_statistics(
            feature,
            donor_mean,
            donor_std,
            weight,
        )
        expected_mean = source_mean.lerp(donor_mean, weight[:, None])
        expected_std = source_std.lerp(donor_std, weight[:, None])
        output_mean = output.mean(dim=(2, 3))
        output_std = (
            (output - output_mean[:, :, None, None])
            .square()
            .mean(dim=(2, 3))
            .sqrt()
        )
        self.assertTrue(torch.allclose(output_mean, expected_mean, atol=1.0e-5))
        self.assertTrue(torch.allclose(output_std, expected_std, atol=1.0e-5))

    def test_invalid_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            self.care.interpolate_feature_statistics(
                torch.randn(1, 2, 3, 3),
                torch.zeros(1, 2),
                torch.ones(1, 2),
                torch.tensor([1.1]),
            )

    def test_optional_statistics_describe_the_applied_intervention(self):
        feature = torch.randn(2, 3, 9, 7)
        source_mean = feature.mean(dim=(2, 3))
        source_std = (
            (feature - source_mean[:, :, None, None])
            .square()
            .mean(dim=(2, 3))
            .add(1.0e-6)
            .sqrt()
        )
        donor_mean = source_mean + source_std
        donor_std = source_std * 2.0
        output, statistics = self.care.interpolate_feature_statistics(
            feature,
            donor_mean,
            donor_std,
            torch.tensor([0.0, 0.5]),
            return_statistics=True,
        )

        self.assertTrue(torch.equal(output[0], feature[0]))
        self.assertTrue(
            torch.allclose(
                statistics["normalized_mean_shift"][0],
                torch.zeros(3),
            )
        )
        self.assertTrue(
            torch.allclose(
                statistics["normalized_mean_shift"][1],
                torch.full((3,), 0.5),
                atol=1.0e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                statistics["target_to_source_std_ratio"][1],
                torch.full((3,), 1.5),
                atol=1.0e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
