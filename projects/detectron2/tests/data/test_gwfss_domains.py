import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DOMAIN_MODULES = (
    REPO_ROOT / "data/datasets/gwfss_domains.py",
    REPO_ROOT / "projects/detectron2/detectron2/data/datasets/gwfss_domains.py",
)


def load_infer_gwfss_domain(path, index):
    spec = importlib.util.spec_from_file_location(
        "gwfss_domains_{}".format(index), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.infer_gwfss_domain


class TestGWFSSDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.infer_functions = [
            load_infer_gwfss_domain(path, index)
            for index, path in enumerate(DOMAIN_MODULES)
        ]

    def assert_inferred_domain(self, path, expected):
        for infer_gwfss_domain in self.infer_functions:
            self.assertEqual(infer_gwfss_domain(path), expected)

    def test_competition_source_domains(self):
        self.assert_inferred_domain(
            "/dataset/images/domain1_00000.png",
            (0, "domain1"),
        )
        self.assert_inferred_domain(
            "/dataset/images/domain9_00000.png",
            (8, "domain9"),
        )

    def test_competition_test_domain(self):
        self.assert_inferred_domain(
            "/dataset/images/domain0_00000.png",
            (9, "domain0"),
        )

    def test_named_domain(self):
        self.assert_inferred_domain(
            "/dataset/Arvalis/image.png",
            (9, "Arvalis"),
        )


if __name__ == "__main__":
    unittest.main()
