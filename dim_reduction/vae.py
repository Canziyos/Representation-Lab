import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1) Data -----
img = Image.open("data/1.jpg").convert("L")
X = np.array(img, dtype=np.float32) / 255.0                 # (H, W)
H, W = X.shape

# center per feature (columns), same as PCA.
mu = X.mean(axis=0, keepdims=True)
Xc = X - mu
Xt = torch.from_numpy(Xc)

# 2) Tied, bias-free AE.
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)  # y = X @ W^T

    def forward(self, x):
        z = self.encoder(x)                    # (N, k).
        x_hat = z @ self.encoder.weight        # decoder = W  -> (N, D).
        return z, x_hat

k = 19
# tied version
model = AutoEncoder(input_dim=W, latent_dim=k) 

# 3) Train on centered data.
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for ep in range(120):
    z, Xhat = model(Xt)
    loss = crit(Xhat, Xt)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if (ep+1) % 10 == 0:
        print(f"Epoch {ep+1}/120  MSE(centered): {loss.item():.6f}")

# 4) Reconstruct for viewing (add mean back).
model.eval()
with torch.no_grad():
    z_ae, Xhat = model(Xt)
recon = np.clip(Xhat.numpy() + mu, 0, 1)

# 5) PCA on the same centered data.
cov = (Xc.T @ Xc) / (H - 1)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
V_k = evecs[:, :k]               # (W, k), orthonormal PCs.
Z_pca = Xc @ V_k                 # (H, k)

# 6) Align PCA latents to AE latents (rotation-invariant).
Z_ae = z_ae.numpy()              # (H, k)
M = Z_pca.T @ Z_ae
U, _, Vt = np.linalg.svd(M, full_matrices=False)
R = U @ Vt                        # orthogonal alignment
Z_pca_aligned = Z_pca @ R

# metrics
fro_rel = np.linalg.norm(Z_pca_aligned - Z_ae, 'fro') / np.linalg.norm(Z_ae, 'fro')
corrs = [np.corrcoef(Z_pca_aligned[:, i], Z_ae[:, i])[0, 1] for i in range(k)]
print(f"Relative Frobenius diff (aligned): {fro_rel:.6f}")
print(f"Mean per-dim corr (aligned): {np.mean(corrs):.4f} | "
      f"min: {np.min(corrs):.4f}  max: {np.max(corrs):.4f}")

# principal angles between subspaces
Qw, _ = np.linalg.qr(model.encoder.weight.detach().numpy().T)  # (W, k) AE basis (orthonormalized)
sv = np.linalg.svd(V_k.T @ Qw, compute_uv=False)               # cosines.
angles_deg = np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))
print(f"Principal angles (deg): min={angles_deg.min():.4f}, mean={angles_deg.mean():.4f}, max={angles_deg.max():.4f}")

# 7) Save latents.
os.makedirs("outputs", exist_ok=True)
np.save("outputs/latents_ae.npy", Z_ae)
np.save("outputs/latents_pca.npy", Z_pca)
np.save("outputs/latents_pca_aligned.npy", Z_pca_aligned)

#8) visuals.
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(X, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(recon, cmap="gray"); plt.title(f"Tied AE Recon ({k} dims)"); plt.axis("off")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(Z_ae[:,0], Z_ae[:,1], s=3, label="AE", alpha=0.6)
plt.scatter(Z_pca_aligned[:,0], Z_pca_aligned[:,1], s=3, label="PCA (aligned)", alpha=0.6)
plt.title("Latent space: AE vs PCA (aligned)"); plt.legend(); plt.show()
