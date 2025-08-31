# import torch
# from models.base_encoder import BaseEncoder
# from models.base_decoder import BaseDecoder
# from models.seq2seq import Seq2Seq

# def test_seq2seq_forward():
#     encoder = BaseEncoder(input_dim=100, emb_dim=32, hidden_dim=16, num_layers=1)
#     decoder = BaseDecoder(output_dim=100, emb_dim=32, hidden_dim=16, num_layers=1)

#     model = Seq2Seq(encoder, decoder, device="cpu")
#     src = torch.randint(0, 100, (2, 5))  # (batch_size=2, src_len=5)
#     trg = torch.randint(0, 100, (2, 6))  # (batch_size=2, trg_len=6)

#     outputs = model(src, trg, teacher_forcing_ratio=0.0)
#     assert outputs.shape == (2, 6, 100), "Seq2Seq output shape is incorrect."

# def main():
#     test_seq2seq_forward()
#     print("All tests passed!")

# if __name__ == "__main__":
#     main()
