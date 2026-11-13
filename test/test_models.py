import torch

from models.rnn import LSTMForecaster, RNNForecaster


def test_rnn_returns_one_value_per_sequence():
    model = RNNForecaster()
    inputs = torch.zeros(4, 30, 4)

    predictions = model(inputs)

    assert tuple(predictions.shape) == (4, 1)


def test_lstm_returns_one_value_per_sequence():
    model = LSTMForecaster()
    inputs = torch.zeros(4, 30, 4)

    predictions = model(inputs)

    assert tuple(predictions.shape) == (4, 1)


def test_rnn_and_lstm_keep_batch_dimension_for_single_sample():
    inputs = torch.zeros(1, 30, 4)

    for model in (RNNForecaster(), LSTMForecaster()):
        predictions = model(inputs)

        assert tuple(predictions.shape) == (1, 1)
        assert torch.isfinite(predictions).all().item()
