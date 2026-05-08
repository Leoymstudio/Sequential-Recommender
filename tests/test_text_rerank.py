import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import InteractionRow
from recsys.text_rerank import TextEmbeddingIndex, TextRerankConfig, encode_hashing, rerank_text_candidates


class TextRerankTest(unittest.TestCase):
    def test_hashing_embeddings_are_normalized(self):
        embeddings = encode_hashing(["red guitar strings", "blue vinyl record"], dim=32)

        self.assertEqual(embeddings.shape, (2, 32))
        self.assertAlmostEqual(float(np.linalg.norm(embeddings[0])), 1.0, places=5)

    def test_rerank_uses_recent_history_text_similarity(self):
        index = TextEmbeddingIndex(
            ["A", "B", "C"],
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        row = InteractionRow(user_id="u", parent_asin="B", rating=5.0, timestamp=1, history="A")
        cfg = TextRerankConfig(base_rank_weight=0.0, text_score_weight=1.0, max_history_items=1)

        recs = rerank_text_candidates(row, ["C", "B"], index, cfg)

        self.assertEqual(recs[0], "B")


if __name__ == "__main__":
    unittest.main()
