from dataloader import get_dataloader

# Get the DataLoader
dataloader = get_dataloader()

# Test DataLoader
for image_batch, label_batch in dataloader:
    print(f"Loaded batch with shape: {image_batch.shape}")
    print(f"Pixel range (min: {image_batch.min()}, max: {image_batch.max()})")
    break  # Test one batch and exit
