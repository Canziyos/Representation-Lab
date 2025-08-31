import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden_dim=512, latent_dim=128):
        super(VAE, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space mean and variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # To ensure output values are between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        epsilon = torch.randn_like(std)  # Sample random noise
        return mu + std * epsilon  # Apply reparameterization trick

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)

        # Encode to latent space
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode back to original dimension
        reconstruction = self.decoder(z)
        reconstruction = reconstruction.view(x.size(0), 3, 32, 32)  # Reshape to image
        return reconstruction, mu, logvar
