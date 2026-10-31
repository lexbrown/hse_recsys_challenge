"""Small independent correctness checks; no competition-quality smoke scores."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse, linalg

from src.data import Data, Index, split_tail, recall_at_k, top_k
from src.baselines import fit_ease


class DataTests(unittest.TestCase):
    def test_metric_uses_macro_average_and_counts_missing_predictions(self):
        score, rows = recall_at_k({'a': ['x', 'z']}, {'a': {'x', 'y'}, 'b': {'q'}}, 10)
        self.assertAlmostEqual(score, 0.25)
        self.assertEqual(len(rows), 2)

    def test_per_user_time_split_does_not_compare_users_timestamps(self):
        events = pd.DataFrame({
            'user_id': ['a'] * 5 + ['b'] * 3, 'item_id': list('12345678'),
            'timestamp': [0, 1, 2, 3, 4, 0, 1, 2], 'rating': [5] * 8,
        })
        train, valid = split_tail(events, 0.2)
        self.assertEqual(set(valid.item_id), {'5', '8'})
        self.assertEqual(len(train), 6)

    def test_seen_filter_and_tie_breaking(self):
        rec = top_k(np.array([5., 5., 4., 3.]), np.array([1., 3., 2., 1.]),
                    np.array([1]), np.array(['a', 'b', 'c', 'd']), k=2)
        self.assertEqual(rec, ['a', 'c'])

    def test_ease_matches_independent_constrained_ridge(self):
        x = sparse.csr_matrix([[1., 1., 0.], [1., 0., 1.], [1., 0., 0.], [0., 1., 1.]])
        weights = fit_ease(x, regularization=2)
        gram = (x.T @ x).toarray()
        for column in range(3):
            others = np.arange(3) != column
            expected = linalg.solve(gram[np.ix_(others, others)] + 2 * np.eye(2), gram[others, column])
            np.testing.assert_allclose(weights[others, column], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(weights.diagonal(), np.zeros(3))


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)

    def test_padding_does_not_create_a_training_target(self):
        from src.sasrec import Windows
        sample = Windows({'u': np.array([0, 1, 2])}, 6, 3, 8)[0]
        inputs, labels, mask = sample
        self.assertEqual(mask.sum().item(), 2)
        self.assertEqual(inputs[mask].tolist(), [0, 1])
        self.assertEqual(labels[mask].tolist(), [1, 2])

    def test_checkpoint_roundtrip_and_causal_attention(self):
        from src.sasrec import create_model, save_checkpoint, load_checkpoint, seed_everything
        torch = self.torch
        parameters = {'embedding_dim': 8, 'num_heads': 1, 'num_blocks': 1,
                      'max_sequence_length': 6, 'dropout': 0.2}
        seed_everything(42)
        model = create_model(parameters, 10).eval()
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
        altered = tokens.clone()
        altered[:, -2:] = torch.tensor([[7, 8]])
        with torch.inference_mode():
            before = model.body({'item_id': tokens}, torch.ones_like(tokens, dtype=torch.bool))
            after = model.body({'item_id': altered}, torch.ones_like(tokens, dtype=torch.bool))
        torch.testing.assert_close(before[:, :-2], after[:, :-2], rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(model, parameters, 3, [str(i) for i in range(10)], path,
                            {'training_split': 'full', 'data_hashes': {'events': 'abc'}})
            restored, payload = load_checkpoint(path, expected_hashes={'events': 'abc'}, expected_split='full')
            torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
            self.assertEqual(payload['epoch'], 3)
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_hashes={'events': 'different'})
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_split='train')


if __name__ == '__main__':
    unittest.main()
