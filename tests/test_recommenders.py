import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import InteractionRow
from recsys.recommenders import HybridParams, HybridSequentialRecommender


def row(user, target, history):
    return InteractionRow(user_id=user, parent_asin=target, rating=5.0, timestamp=1, history=history)


class HybridRecommenderTest(unittest.TestCase):
    def test_hybrid_learns_last_item_transition_and_filters_seen_items(self):
        model = HybridSequentialRecommender(HybridParams(top_k=3, popularity_fallback=10))
        model.fit(
            [
                row("u1", "B", "A"),
                row("u2", "B", "A"),
                row("u3", "C", "A B"),
                row("u4", "D", ""),
            ]
        )

        recs = model.recommend("A", k=3)

        self.assertEqual(recs[0], "B")
        self.assertNotIn("A", recs)
        self.assertEqual(len(recs), 3)


if __name__ == "__main__":
    unittest.main()
