import torch
from models.BaseEncoder import BaseEncoder
from models.BaseDecoder import BaseDecoder
from models.seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
INPUT_DIM = 100
OUTPUT_DIM = 100
EMB_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
SEQ_LEN = 10
BATCH_SIZE = 4

# Initialize models
encoder = BaseEncoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
decoder = BaseDecoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)

# Generate random inputs
src = torch.randint(0, INPUT_DIM, (BATCH_SIZE, SEQ_LEN)).to(device)
trg = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE, SEQ_LEN)).to(device)

print("\n--- Testing Encoder ---")
encoder_outputs, hidden = encoder(src)
print("Input shape:", src.shape)
print("Encoder output shape:", encoder_outputs.shape)
print("Hidden state shape:", hidden.shape)
print("Hidden state values:", hidden)

print("\n--- Testing Decoder ---")
first_token = trg[:, 0]
hidden_state = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM).to(device)
decoder_output, new_hidden = decoder(first_token, hidden_state)
print("Decoder input shape:", first_token.shape)
print("Decoder output shape:", decoder_output.shape)
print("New hidden state shape:", new_hidden.shape)
print("New hidden state values:", new_hidden)

print("\n--- Testing Full Seq2Seq Model ---")
seq2seq_output = seq2seq_model(src, trg)
print("Seq2Seq output shape:", seq2seq_output.shape)
print("Seq2Seq output values:", seq2seq_output)

print("\n--- Testing Inference ---")
generated_sequence = seq2seq_model.inference(src, max_len=SEQ_LEN)
print("Generated sequence shape:", generated_sequence.shape)
print("Generated sequence:", generated_sequence)
