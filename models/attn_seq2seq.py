# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BahdanauAttention(nn.Module):
#     """
#     Simple Bahdanau (Additive) attention mechanism.
#     """
#     def __init__(self, enc_hidden_dim, dec_hidden_dim):
#         super().__init__()
#         self.W_enc = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
#         self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
#         self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

#     def forward(self, decoder_hidden, encoder_outputs):
#         """
#         decoder_hidden: (batch_size, dec_hidden_dim)
#         encoder_outputs: (batch_size, src_len, enc_hidden_dim)
#         """
#         # Expand decoder_hidden to match source length
#         src_len = encoder_outputs.shape[1]
#         decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
#         # Transform
#         enc_transform = self.W_enc(encoder_outputs)  # (batch_size, src_len, dec_hidden_dim)
#         dec_transform = self.W_dec(decoder_hidden_expanded)  # (batch_size, src_len, dec_hidden_dim)
        
#         # Combine and apply activation
#         combined = torch.tanh(enc_transform + dec_transform)
#         scores = self.v(combined).squeeze(-1)  # (batch_size, src_len)

#         # Normalize to get attention weights
#         attn_weights = F.softmax(scores, dim=1)  # (batch_size, src_len)
#         return attn_weights


# class AttnEncoder(nn.Module):
#     """
#     Simple GRU encoder. Returns outputs (for attention) plus final hidden state.
#     """
#     def __init__(self, input_dim, emb_dim, enc_hidden_dim, num_layers=1):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, emb_dim)
#         self.rnn = nn.GRU(emb_dim, enc_hidden_dim, num_layers, batch_first=True)

#     def forward(self, src):
#         # src: (batch_size, src_len)
#         embedded = self.embedding(src)  # (batch_size, src_len, emb_dim)
#         outputs, hidden = self.rnn(embedded)  
#         # outputs: (batch_size, src_len, enc_hidden_dim)
#         # hidden:  (num_layers, batch_size, enc_hidden_dim)
#         return outputs, hidden


# class AttnDecoder(nn.Module):
#     """
#     GRU decoder with Bahdanau attention.
#     """
#     def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, num_layers=1):
#         super().__init__()
#         self.output_dim = output_dim
#         self.embedding = nn.Embedding(output_dim, emb_dim)

#         # For simplicity, assume enc_hidden_dim == dec_hidden_dim 
#         # (otherwise we'd need a projection)
#         self.rnn = nn.GRU(enc_hidden_dim + emb_dim, dec_hidden_dim, num_layers, batch_first=True)
#         self.fc_out = nn.Linear(dec_hidden_dim, output_dim)

#         self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim)

#     def forward(self, input_token, hidden, encoder_outputs):
#         """
#         input_token: (batch_size,) with next token index
#         hidden: (num_layers, batch_size, dec_hidden_dim)
#         encoder_outputs: (batch_size, src_len, enc_hidden_dim)
#         """
#         # Embedding
#         input_token = input_token.unsqueeze(1)  # (batch_size, 1)
#         embedded = self.embedding(input_token)  # (batch_size, 1, emb_dim)

#         # Compute attention over encoder_outputs
#         # hidden[-1] is (batch_size, dec_hidden_dim) – the top layer hidden state
#         dec_hidden_top = hidden[-1]  
#         attn_weights = self.attention(dec_hidden_top, encoder_outputs)  
#         # attn_weights: (batch_size, src_len)

#         # Weighted sum of encoder outputs
#         attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, src_len)
#         context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, 1, enc_hidden_dim)

#         # Concatenate context + embedded
#         rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, emb_dim+enc_hidden_dim)

#         # Pass through GRU
#         output, hidden = self.rnn(rnn_input, hidden)
#         # output: (batch_size, 1, dec_hidden_dim)

#         # Predict next token
#         prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_dim)
#         return prediction, hidden, attn_weights.squeeze(1)


# class AttnSeq2Seq(nn.Module):
#     """
#     Seq2Seq that uses AttnEncoder and AttnDecoder. 
#     """
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device

#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         """
#         src: (batch_size, src_len)
#         trg: (batch_size, trg_len)
#         """
#         batch_size, trg_len = trg.size()
#         output_dim = self.decoder.output_dim

#         # To store decoder outputs
#         outputs = torch.zeros(batch_size, trg_len, output_dim, device=self.device)

#         encoder_outputs, hidden = self.encoder(src)
#         # hidden: (num_layers, batch_size, enc_hidden_dim)

#         input_token = trg[:, 0]  # <sos> token

#         for t in range(1, trg_len):
#             prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
#             outputs[:, t, :] = prediction

#             teacher_force = torch.rand(1).item() < teacher_forcing_ratio
#             top1 = prediction.argmax(1)

#             input_token = trg[:, t] if teacher_force else top1

#         return outputs
