import logging
import torch
from torch import nn

 
class CharModel(nn.Module):
    def __init__(self, n_vocab) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
        
    def forward(self, x) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x
