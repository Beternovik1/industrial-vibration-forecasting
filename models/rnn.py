import torch.nn as nn


class RNNForecaster(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=32,
        output_size=1,
    ):
        super(RNNForecaster, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

    def forward(self, x):
        output, _ = self.rnn(x)

        # Keep the output from the last minute in the input window.
        last_hidden = output[:, -1, :]
        return self.output(last_hidden)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=32,
        output_size=1,
    ):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        # Keep the output from the last minute in the input window.
        last_hidden = output[:, -1, :]
        return self.output(last_hidden)
