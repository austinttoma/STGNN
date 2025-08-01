import torch
import torch.nn as nn
from typing import Optional

class RNNPredictor(nn.Module):
    def __init__(self,
                 graph_emb_dim: int = 256,
                 tab_emb_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.3,
                 bidirectional: bool = False,
                 num_classes: int = 2):
        super().__init__()

        self.tab_emb_dim = tab_emb_dim
        self.input_dim = graph_emb_dim + tab_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        self.rnn = nn.RNN(input_size=self.input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional,
                          nonlinearity='tanh')  # or 'relu'

        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, graph_seq: torch.Tensor, tab_seq: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if tab_seq is None or self.tab_emb_dim == 0:
            fused = graph_seq
        else:
            fused = torch.cat([graph_seq, tab_seq], dim=-1)

        if lengths is not None:
            fused_packed = nn.utils.rnn.pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output_packed, h_n = self.rnn(fused_packed)
        else:
            output, h_n = self.rnn(fused)

        if self.bidirectional:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_hidden = h_n[-1]

        logits = self.classifier(final_hidden)
        return logits