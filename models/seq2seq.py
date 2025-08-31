import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    """
    A generic sequence-to-sequence model that connects an encoder and decoder.
    Can be easily extended with attention, masks, custom loss functions, etc.
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.9):  # Increase teacher forcing
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        input_token = trg[:, 0]  # Start with <SOS>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = output.squeeze(1)

            # 🔥 Debugging: Check raw outputs before choosing next word
            print(f"Step {t}, Decoder Output (raw logits): {output.squeeze(1)}")

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

    def inference(self, src, max_len=15, start_symbol=1, end_symbol=2, temperature=0.7):
        """
        Inference mode: Given only a source (src), generate a sequence until <EOS> or max_len.
        """
        batch_size = src.size(0)
        outputs = []

        # Get hidden state from encoder
        encoder_outputs, hidden = self.encoder(src)

        # Start with <SOS> token
        input_token = torch.tensor([start_symbol] * batch_size, device=self.device)

        for _ in range(max_len):
            output, hidden = self.decoder(input_token, hidden)
            
            # Apply temperature scaling
            output = output / temperature  # Lower temp = less random
            output = torch.nn.functional.softmax(output, dim=-1)  # Normalize
            
            # Select the most probable word
            pred_token = output.argmax(2).squeeze(1)  # Use argmax instead of multinomial()

            outputs.append(pred_token.unsqueeze(1))
            input_token = pred_token

            # Stop if all sequences predicted <EOS>
            if (pred_token == end_symbol).all():
                break

        # Concatenate tokens to form final sequence: (batch_size, sequence_length)
        outputs = torch.cat(outputs, dim=1)
        return outputs
