import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("data/1.jpg")

# Pick scale based on bit depth.
if img.mode in ["L", "RGB", "RGBA"]:
    scale = 255.0
elif img.mode in ["I;16", "I;16B"]:
    scale = 65535.0
else:
    scale = 1.0 

arr = np.array(img) / scale

# Prepare data for PCA.
if img.mode == "L":  # grayscale.
    data = arr                     # shape (H, W).
elif img.mode == "RGB":
    data = arr.reshape(arr.shape[0], -1)  # shape (H, W*3).
elif img.mode == "RGBA":
    data = arr.reshape(arr.shape[0], -1)  # shape (H, W*4).
else:
    raise ValueError(f"Unsupported mode: {img.mode}")

# Center the data.
feature_mean = np.mean(data, axis=0)
centered = data - feature_mean

# Covariance of the samples.
covar = np.cov(centered, rowvar=False)

# Eigen decomposition.
eigenvalues, eigenvectors = np.linalg.eigh(covar)

# Sort eigenpairs.
idx = np.argsort(eigenvalues)[::-1]
ordered_ev = eigenvalues[idx]
ordered_ve = eigenvectors[:, idx]

# Explained variance ratio.
explained_var_ratio = ordered_ev / np.sum(ordered_ev)
cum_explained_var = np.cumsum(explained_var_ratio)

# Choose k components to keep 95% of variance.
target_variance = 0.95
k = np.argmax(cum_explained_var >= target_variance) + 1
print(f"Keeping {k} components to preserve {target_variance*100:.1f}% variance.")

# Projection onto top-k components.
top_k_vecs = ordered_ve[:, :k]            # (n_features, k)
proj_data = centered @ top_k_vecs         # (n_samples, k)

print("Projection shape:", proj_data.shape)
print("First 5 explained variance ratios:", explained_var_ratio[:5])
print("Cumulative variance at k:", cum_explained_var[k-1])

# ==== Reconstruction ====
reconstructed = proj_data @ top_k_vecs.T  # back to original space
reconstructed = reconstructed + feature_mean
reconstructed = np.clip(reconstructed, 0, 1)

# Reshape back to image.
if img.mode == "L":
    recon_img = reconstructed
elif img.mode == "RGB":
    recon_img = reconstructed.reshape(arr.shape[0], arr.shape[1], 3)
elif img.mode == "RGBA":
    recon_img = reconstructed.reshape(arr.shape[0], arr.shape[1], 4)


# ==== Visualization ====
plt.figure(figsize=(10, 4))

# Original
plt.subplot(1, 2, 1)
plt.imshow(arr)
plt.title("Original")
plt.axis("off")

# Reconstructed
plt.subplot(1, 2, 2)
plt.imshow(recon_img)
plt.title(f"Reconstructed ({k} PCs)")
plt.axis("off")

plt.show()
# ---- Scatter plots of multiple PC pairs ----
pairs = [(0, 1), (1, 2), (2, 3)]  # PC1 vs PC2, PC2 vs PC3, PC3 vs PC4
plt.figure(figsize=(15, 4))
for i, (a, b) in enumerate(pairs):
    plt.subplot(1, len(pairs), i+1)
    plt.scatter(centered @ ordered_ve[:, a],
                centered @ ordered_ve[:, b],
                s=5, alpha=0.6)
    plt.xlabel(f"PC{a+1}")
    plt.ylabel(f"PC{b+1}")
    plt.title(f"PC{a+1} vs PC{b+1}")
plt.tight_layout()
plt.show()

# ---- Heatmap of all projections ----
plt.figure(figsize=(8, 6))
plt.imshow(proj_data[:, :50], aspect="auto", cmap="viridis")
plt.colorbar(label="Projection Value")
plt.title("Projections onto First 50 PCs")
plt.xlabel("Principal Component")
plt.ylabel("Sample (row index)")
plt.show()

from utils import pca_compression_ratio

stats = pca_compression_ratio(centered.shape, k)
print(f"Original size: {stats['original_size']}")
print(f"Compressed size: {stats['compressed_size']}")
print(f"Compression ratio: {stats['ratio']:.4f}")
