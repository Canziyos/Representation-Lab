import torch

def greedy_decode(model, src, max_len=50, start_symbol=1):
    """
    Example of a very basic greedy decode.
    Args:
        model: The trained seq2seq model.
        src: Source sequence (tensor of shape [1, src_len]).
        max_len: Max steps to decode.
        start_symbol: <sos> index
    """
    model.eval()
    with torch.no_grad():
        _, hidden = model.encoder(src)
        input_token = torch.tensor([start_symbol], device=src.device).unsqueeze(0)  # shape: (1,1)

        outputs = []
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            # output shape: (1, 1, vocab_size)
            pred_token = output.argmax(dim=-1)  # (1, 1)
            outputs.append(pred_token.item())
            input_token = pred_token
            # Optionally break on <eos>
            # if pred_token.item() == our_eos_idx:
            #     break

        return outputs

# Example usage:
if __name__ == "__main__":
    # we have a loaded model already
    print("Inference script placeholder")
