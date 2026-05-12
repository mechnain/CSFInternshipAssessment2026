from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate  # noqa: E402
from src.utils import OpenSetThresholds  # noqa: E402


class EvaluateMetricsTest(unittest.TestCase):
    def test_zero_precision_recall_gives_zero_f1_not_none(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "query": "q1.jpg",
                    "rank": 1,
                    "gallery_id": "wrong",
                    "similarity": 0.9,
                    "decision": "match",
                }
            ]
        )
        labels = pd.DataFrame([{"query_image": "q1.jpg", "true_id": "right"}])

        metrics = evaluate(results, labels, OpenSetThresholds(match=0.7, possible=0.55))

        self.assertEqual(metrics["thresholded"]["TP"], 0)
        self.assertEqual(metrics["thresholded"]["FP"], 1)
        self.assertEqual(metrics["thresholded"]["FN"], 1)
        self.assertEqual(metrics["thresholded"]["precision"], 0.0)
        self.assertEqual(metrics["thresholded"]["recall"], 0.0)
        self.assertEqual(metrics["thresholded"]["f1"], 0.0)

    def test_possible_unknown_is_abstention_not_true_negative(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "query": "u.jpg",
                    "rank": 1,
                    "gallery_id": "dog_001",
                    "similarity": 0.6,
                    "decision": "possible_match",
                }
            ]
        )
        labels = pd.DataFrame([{"query_image": "u.jpg", "true_id": "unknown"}])

        metrics = evaluate(results, labels, OpenSetThresholds(match=0.7, possible=0.55))

        self.assertEqual(metrics["thresholded"]["TN"], 0)
        self.assertEqual(metrics["thresholded"]["abstain_unknown"], 1)
        self.assertEqual(metrics["open_set"]["unknown_accuracy"], 0.0)
        self.assertEqual(metrics["open_set"]["non_match_rejection_rate"], 1.0)

    def test_duplicate_labels_are_rejected(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "query": "q.jpg",
                    "rank": 1,
                    "gallery_id": "dog_001",
                    "similarity": 0.9,
                    "decision": "match",
                }
            ]
        )
        labels = pd.DataFrame(
            [
                {"query_image": "q.jpg", "true_id": "dog_001"},
                {"query_image": "q.jpg", "true_id": "dog_002"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate query_image"):
            evaluate(results, labels, OpenSetThresholds(match=0.7, possible=0.55))

    def test_bad_threshold_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "possible <= match"):
            OpenSetThresholds(match=0.5, possible=0.6)


if __name__ == "__main__":
    unittest.main()
