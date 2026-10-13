from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/embedding"))

from confidence_features import (  # noqa: E402
    build_confidence_features,
    feature_names,
    multimodal_consistency,
    normalize_rows,
    robust_scale_support,
)


class ConfidenceFeatureTests(unittest.TestCase):
    def test_identical_sources_have_maximum_agreement(self):
        base = normalize_rows(np.asarray([[1.0, 2.0, 0.5], [0.2, -1.0, 3.0]]))
        sources = np.stack([base, base, base, base], axis=1)
        available = np.ones((2, 4), dtype=bool)
        confidence = np.ones((2, 4), dtype=np.float32)
        features, agreement = build_confidence_features(sources, available, confidence)
        self.assertEqual(features.shape, (2, len(feature_names(["a", "b", "c", "d"]))))
        np.testing.assert_allclose(agreement, 1.0, atol=1e-6)

    def test_outlier_source_receives_lower_geometric_agreement(self):
        sources = np.asarray(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]],
            dtype=np.float32,
        )
        _, agreement = build_confidence_features(
            sources,
            np.ones((1, 4), dtype=bool),
            np.ones((1, 4), dtype=np.float32),
        )
        self.assertLess(float(agreement[0, 3]), float(agreement[0, 0]))

    def test_multimodal_consistency_tracks_variant_disagreement(self):
        variants = np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [-1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        consistency = multimodal_consistency(variants)
        np.testing.assert_allclose(consistency, [1.0, 0.0], atol=1e-6)

    def test_support_scaling_is_bounded_and_monotonic(self):
        scaled = robust_scale_support(np.asarray([0.0, 1.0, 10.0, 1000.0]))
        self.assertTrue(np.all((scaled >= 0.0) & (scaled <= 1.0)))
        self.assertTrue(np.all(np.diff(scaled) >= 0.0))


if __name__ == "__main__":
    unittest.main()
