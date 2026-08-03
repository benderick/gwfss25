"""Protocol constants and validation for CARE artifacts."""

import math

import numpy as np


PHASE0_AUDIT_VERSION = 2
CARE_BANK_VERSION = 2
RUNTIME_CALIBRATION_SOURCE = "training_anchor_teacher_cross_domain_nn"
RUNTIME_CALIBRATION_QUANTILE = 0.75
RUNTIME_STANDARDIZATION_SOURCE = "unlabeled_donor_pool_robust_iqr"


def calibrate_training_anchor_compatibility(rows, signatures, dataset_name):
    """Calibrate the donor cutoff without validation images or ground truth."""
    signatures = np.asarray(signatures, dtype=np.float64)
    if signatures.ndim != 2 or len(signatures) != len(rows) or not len(rows):
        raise ValueError("CARE calibration requires one signature per anchor")
    if not np.isfinite(signatures).all():
        raise ValueError("CARE calibration signatures must be finite")

    domains = np.asarray([row["domain_id"] for row in rows], dtype=np.int64)
    nearest_distances = []
    for index in range(len(rows)):
        candidates = np.flatnonzero(domains != domains[index])
        if not len(candidates):
            raise ValueError(
                "CARE calibration requires a cross-domain anchor for every image"
            )
        differences = signatures[candidates] - signatures[index]
        distances = np.sqrt(np.mean(np.square(differences), axis=1))
        nearest_distances.append(float(distances.min()))

    threshold = float(
        np.quantile(nearest_distances, RUNTIME_CALIBRATION_QUANTILE)
    )
    return {
        "source": RUNTIME_CALIBRATION_SOURCE,
        "dataset_role": "training_anchors",
        "dataset": str(dataset_name),
        "descriptor_inputs": "frozen_teacher_predictions",
        "standardization_source": RUNTIME_STANDARDIZATION_SOURCE,
        "sample_count": len(rows),
        "quantile": RUNTIME_CALIBRATION_QUANTILE,
        "teacher_nn_distance_median": float(np.median(nearest_distances)),
        "teacher_nn_distance_q75": threshold,
        "compatibility_threshold": threshold,
        "uses_validation_images": False,
        "uses_ground_truth": False,
    }


def validate_runtime_calibration(
    calibration,
    *,
    expected_dataset=None,
    expected_sample_count=None,
):
    if not isinstance(calibration, dict):
        raise ValueError("CARE runtime calibration metadata is missing")
    if calibration.get("source") != RUNTIME_CALIBRATION_SOURCE:
        raise ValueError("CARE runtime cutoff was not calibrated on training anchors")
    if calibration.get("dataset_role") != "training_anchors":
        raise ValueError("CARE runtime calibration has an invalid dataset role")
    if not str(calibration.get("dataset", "")):
        raise ValueError("CARE runtime calibration dataset is missing")
    if calibration.get("descriptor_inputs") != "frozen_teacher_predictions":
        raise ValueError("CARE runtime calibration descriptor source is unsupported")
    if calibration.get("standardization_source") != (
        RUNTIME_STANDARDIZATION_SOURCE
    ):
        raise ValueError("CARE runtime calibration standardization is unsupported")
    if calibration.get("uses_validation_images") is not False:
        raise ValueError("CARE runtime calibration must not use validation images")
    if calibration.get("uses_ground_truth") is not False:
        raise ValueError("CARE runtime calibration must not use ground truth")

    quantile = float(calibration.get("quantile", float("nan")))
    if not math.isclose(
        quantile,
        RUNTIME_CALIBRATION_QUANTILE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("CARE runtime calibration quantile is unsupported")
    threshold = float(calibration.get("compatibility_threshold", float("nan")))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("CARE runtime compatibility threshold is invalid")

    sample_count = int(calibration.get("sample_count", -1))
    if sample_count <= 0:
        raise ValueError("CARE runtime calibration sample count is invalid")
    if expected_sample_count is not None and sample_count != int(
        expected_sample_count
    ):
        raise ValueError("CARE runtime calibration anchor count is inconsistent")
    if expected_dataset is not None and calibration.get("dataset") != str(
        expected_dataset
    ):
        raise ValueError("CARE runtime calibration dataset is inconsistent")
    return calibration


def validate_phase0_protocol(summary):
    if int(summary.get("audit_version", -1)) != PHASE0_AUDIT_VERSION:
        raise ValueError(
            "unsupported CARE Phase-0 audit version; regenerate Phase 0"
        )
    anchors = summary.get("datasets", {}).get("anchors", {})
    if not anchors.get("name") or int(anchors.get("selected", -1)) <= 0:
        raise ValueError("CARE Phase-0 training-anchor metadata is invalid")
    calibration = validate_runtime_calibration(
        summary.get("runtime_calibration"),
        expected_dataset=anchors.get("name"),
        expected_sample_count=anchors.get("selected"),
    )
    donor_support = summary.get("donor_support", {})
    donor_threshold = float(
        donor_support.get("compatibility_threshold", float("nan"))
    )
    if not math.isclose(
        donor_threshold,
        float(calibration["compatibility_threshold"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("CARE donor selection used a different runtime cutoff")
    if donor_support.get("compatibility_calibration_source") != (
        RUNTIME_CALIBRATION_SOURCE
    ):
        raise ValueError("CARE donor selection calibration source is inconsistent")
    return calibration


def validate_bank_protocol(manifest):
    if int(manifest.get("bank_version", -1)) != CARE_BANK_VERSION:
        raise ValueError("unsupported CARE bank version; regenerate the bank")
    if int(manifest.get("phase0_audit_version", -1)) != PHASE0_AUDIT_VERSION:
        raise ValueError("CARE bank was built from an unsupported Phase-0 audit")
    total_anchor_count = int(manifest.get("total_anchor_count", -1))
    if total_anchor_count <= 0:
        raise ValueError("CARE bank training-anchor count is invalid")
    return validate_runtime_calibration(
        manifest.get("runtime_calibration"),
        expected_sample_count=total_anchor_count,
    )
