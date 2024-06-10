import torch
from torch import nn

# TODO добавить нормализацию и embedding

class CharModel(nn.Module):
    def __init__(self, n_vocab: int, hidden_size: int, n_layer: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, n_vocab)

    def forward(self, x, hidden_state) -> torch.Tensor:
        # embedding = self.embedding(x)
        output, hidden_state = self.lstm(x, hidden_state)
        output = self.linear(self.dropout(output))
        return output, hidden_state

        # embedding = self.embedding(input_seq)
        # output, hidden_state = self.rnn(embedding, hidden_state)
        # output = self.decoder(output)
        # return output, (hidden_state[0].detach(), hidden_state[1].detach())