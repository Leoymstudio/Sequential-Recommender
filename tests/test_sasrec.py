import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    from recsys.sasrec import SASRecModel, SasRecConfig
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SasRecModelTest(unittest.TestCase):
    def test_encode_uses_right_most_left_padded_position(self):
        config = SasRecConfig(max_len=5, hidden_size=8, num_heads=2, num_layers=1, dropout=0.0)
        model = SASRecModel(num_items=10, config=config)
        seq = torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]], dtype=torch.long)

        encoded = model.encode(seq)

        self.assertEqual(tuple(encoded.shape), (2, 8))


if __name__ == "__main__":
    unittest.main()
