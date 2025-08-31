import torch
import torch.nn as nn

class BaseDecoder(nn.Module):
    """
    Generic decoder: Takes in the previous token and hidden state,
    and returns a probability distribution over the next token and the next hidden state.
    """
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)  # Optional dropout
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # x: (batch_size,) or (batch_size, 1) with token indices
        if x.dim() == 1:
            x = x.unsqueeze(1)              # (batch_size, 1)
        embedded = self.embedding(x)          # (batch_size, 1, emb_dim)
        embedded = self.dropout(embedded)     # Apply dropout if needed
        outputs, hidden = self.rnn(embedded, hidden)  # (batch_size, 1, hidden_dim)
        predictions = self.fc(outputs)        # (batch_size, 1, output_dim)
        return predictions, hidden

print("\nDone\n")
