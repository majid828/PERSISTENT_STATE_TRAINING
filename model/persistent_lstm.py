import torch
import torch.nn as nn


class LSTMPersistent(nn.Module):
    """Stateful LSTM: carries hidden state across batches/windows when you pass it in."""

    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout_p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden_states=None):
        """
        If hidden_states is None, starts from zeros.
        Otherwise, continues from provided (h, c).
        """
        if hidden_states is None:
            B = x.size(0)
            h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            hidden_states = (h0, c0)

        out, new_hidden = self.lstm(x, hidden_states)
        out = self.dropout(out)
        pred = self.fc(out)
        return pred, new_hidden

