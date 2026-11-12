import unittest

import torch

from models.rnn import LSTMForecaster, RNNForecaster


class TestForecasters(unittest.TestCase):
    def test_rnn_returns_one_value_per_sequence(self):
        model = RNNForecaster()
        inputs = torch.zeros(4, 30, 4)

        predictions = model(inputs)

        self.assertEqual(tuple(predictions.shape), (4, 1))

    def test_lstm_returns_one_value_per_sequence(self):
        model = LSTMForecaster()
        inputs = torch.zeros(4, 30, 4)

        predictions = model(inputs)

        self.assertEqual(tuple(predictions.shape), (4, 1))

    def test_both_models_keep_single_item_batch_dimension(self):
        inputs = torch.zeros(1, 30, 4)

        for model in (RNNForecaster(), LSTMForecaster()):
            predictions = model(inputs)

            self.assertEqual(tuple(predictions.shape), (1, 1))
            self.assertTrue(torch.isfinite(predictions).all().item())


if __name__ == "__main__":
    unittest.main()
