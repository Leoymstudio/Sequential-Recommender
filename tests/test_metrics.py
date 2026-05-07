import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.metrics import ndcg_at_k


class NdcgMetricTest(unittest.TestCase):
    def test_ndcg_first_rank_is_one(self):
        self.assertEqual(ndcg_at_k(["A", "B"], "A", k=10), 1.0)

    def test_ndcg_uses_one_based_rank_discount(self):
        self.assertEqual(
            round(ndcg_at_k(["A", "B"], "B", k=10), 6),
            round(1.0 / 1.584962500721156, 6),
        )

    def test_ndcg_miss_is_zero(self):
        self.assertEqual(ndcg_at_k(["A", "B"], "C", k=10), 0.0)


if __name__ == "__main__":
    unittest.main()
