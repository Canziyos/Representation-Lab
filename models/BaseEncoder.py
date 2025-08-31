import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    """
    Generic encoder: Takes in token indices (e.g., words) and returns a sequence
    of hidden representations along with the final hidden state.
    """
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)  # Optional dropout
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)            # (batch_size, seq_len, emb_dim)
        embedded = self.dropout(embedded)         # Apply dropout if dropout > 0
        outputs, hidden = self.rnn(embedded)      # outputs: (batch_size, seq_len, hidden_dim)
        return outputs, hidden                    # hidden: (num_layers, batch_size, hidden_dim)

print("\nDone\n")
