import torch
import torch.nn as nn
from typing import List, Optional

class TemporalTabGNNClassifier(nn.Module):
    def __init__(self,
                 graph_emb_dim: int = 256,
                 tab_emb_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.3,
                 bidirectional: bool = False,
                 num_classes: int = 2):
        super().__init__()
        
        # Allow tabular embedding to be optional (set tab_emb_dim=0 for graph-only model)
        self.tab_emb_dim = tab_emb_dim
        self.input_dim = graph_emb_dim + tab_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self,
                graph_seq: torch.Tensor,   # [B, T, graph_emb_dim]
                tab_seq: Optional[torch.Tensor] = None,     # [B, T, tab_emb_dim] or None if no tabular data
                lengths: Optional[torch.Tensor] = None,  # [B] true sequence lengths
                mask: Optional[torch.Tensor] = None      # [B, T] attention mask
                ) -> torch.Tensor:
        """
        Forward pass through ST-GNN.
        
        Args:
            graph_seq: [B, T, 256] encoded graph sequence
            tab_seq: [B, T, 64] encoded tabular sequence
            lengths: [B] true sequence lengths (for packed sequences)
            mask: [B, T] attention mask (True for real data, False for padding)
            
        Returns:
            logits: [B, num_classes] classification logits
        """

        # Step 1: Concatenate graph and tabular embeddings
        if tab_seq is None or self.tab_emb_dim == 0:
            fused = graph_seq  # [B, T, graph_emb_dim]
        else:
            fused = torch.cat([graph_seq, tab_seq], dim=-1)  # [B, T, graph_emb_dim + tab_emb_dim]

        # Step 2: Process through LSTM
        if lengths is not None:
            # Use packed sequences for efficiency
            fused_packed = nn.utils.rnn.pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output_packed, (h_n, c_n) = self.lstm(fused_packed)
            # Unpack if needed (we only use hidden state)
            # output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        else:
            # Standard processing (padding will be ignored in final hidden state)
            output, (h_n, c_n) = self.lstm(fused)

        # Step 3: Use final hidden state from the top layer
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2*hidden_dim]
        else:
            final_hidden = h_n[-1]  # [B, hidden_dim]

        # Step 4: Classification
        logits = self.classifier(final_hidden)  # [B, num_classes]
        return logits