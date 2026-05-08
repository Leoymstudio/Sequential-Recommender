import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    from recsys.graph_rerank import GraphData, GraphRerankConfig, build_normalized_adjacency, rerank_graph_candidates
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GraphRerankTest(unittest.TestCase):
    def test_normalized_adjacency_shape(self):
        graph = GraphData()
        user = graph.user_id("u")
        item = graph.item_id("A")
        graph.edge_users.append(user)
        graph.edge_items.append(item)

        adjacency = build_normalized_adjacency(graph, torch.device("cpu"))

        self.assertEqual(tuple(adjacency.shape), (2, 2))

    def test_graph_rerank_can_promote_graph_similar_candidate(self):
        graph = GraphData()
        user = graph.user_id("u")
        graph.item_id("A")
        graph.item_id("B")
        user_emb = torch.tensor([[1.0, 0.0]])
        item_emb = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        cfg = GraphRerankConfig(base_rank_weight=0.0, graph_score_weight=1.0)

        recs = rerank_graph_candidates("u", ["A", "B"], graph, user_emb, item_emb, cfg)

        self.assertEqual(recs[0], "B")


if __name__ == "__main__":
    unittest.main()
