"""
DCGAN architecture adapted for small datasets (~100 images).

Architecture decisions:
  - Smaller than standard DCGAN → fewer parameters → less overfitting on small data
  - Spectral normalisation on Discriminator → more stable training
  - Instance noise on Discriminator inputs → prevents mode collapse
  - Grayscale output (1 channel) → matches your sketch images
  - Easily upgradeable to 128×128 by adding one more block
"""

import torch
import torch.nn as nn


# ── Hyper-parameters ─────────────────────────────────────────────────────────

LATENT_DIM  = 128   # Size of the random noise vector (z)
IMAGE_SIZE  = 64    # Must match preprocess.IMAGE_SIZE
N_CHANNELS  = 1     # 1 = grayscale sketches;  3 = colour (later phase)
BASE_FEAT_G = 64    # Base feature count in Generator
BASE_FEAT_D = 64    # Base feature count in Discriminator


# ── Building blocks ───────────────────────────────────────────────────────────

def _conv_block_g(in_ch, out_ch, kernel=4, stride=2, padding=1):
    """Upsampling block for the Generator (ConvTranspose2d + BN + ReLU)."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _conv_block_d(in_ch, out_ch, kernel=4, stride=2, padding=1):
    """Downsampling block for the Discriminator (Conv2d + SpecNorm + LeakyReLU)."""
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Takes a random vector z ∈ R^{LATENT_DIM} and outputs a sketch image.

    Architecture (64×64 output):
      z (128)  →  4×4  →  8×8  →  16×16  →  32×32  →  64×64

    To get 128×128 output, add one more _conv_block_g at the end and
    change the first projection to 2×2.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        base_feat:  int = BASE_FEAT_G,
        n_channels: int = N_CHANNELS,
    ):
        super().__init__()
        bf = base_feat

        self.net = nn.Sequential(
            # Project noise → 4×4 feature map
            nn.ConvTranspose2d(latent_dim, bf * 8, 4, 1, 0, bias=False),  # → (bf*8) × 4 × 4
            nn.BatchNorm2d(bf * 8),
            nn.ReLU(inplace=True),

            _conv_block_g(bf * 8, bf * 4),  # → (bf*4) × 8 × 8
            _conv_block_g(bf * 4, bf * 2),  # → (bf*2) × 16 × 16
            _conv_block_g(bf * 2, bf),      # → (bf)   × 32 × 32

            # Final layer: no BN, Tanh squishes output to [-1, 1]
            nn.ConvTranspose2d(bf, n_channels, 4, 2, 1, bias=False),  # → n_ch × 64 × 64
            nn.Tanh(),
        )

        self._init_weights()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Takes a 64×64 image (real or fake) and outputs a probability score.
    Spectral normalisation is used instead of BatchNorm for stability.

    Instance noise (added during training, not here) prevents the discriminator
    from becoming too confident and crashing the generator.
    """

    def __init__(
        self,
        base_feat:  int = BASE_FEAT_D,
        n_channels: int = N_CHANNELS,
    ):
        super().__init__()
        bf = base_feat

        self.net = nn.Sequential(
            # No BN on first layer (common practice)
            nn.utils.spectral_norm(
                nn.Conv2d(n_channels, bf, 4, 2, 1, bias=False)  # → bf × 32 × 32
            ),
            nn.LeakyReLU(0.2, inplace=True),

            _conv_block_d(bf,     bf * 2),   # → (bf*2) × 16 × 16
            _conv_block_d(bf * 2, bf * 4),   # → (bf*4) × 8  × 8
            _conv_block_d(bf * 4, bf * 8),   # → (bf*8) × 4  × 4

            # Output a single scalar per image
            nn.utils.spectral_norm(
                nn.Conv2d(bf * 8, 1, 4, 1, 0, bias=False)  # → 1 × 1 × 1
            ),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)   # Flatten to [batch]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_noise(batch_size: int, latent_dim: int = LATENT_DIM, device="cpu") -> torch.Tensor:
    """Sample a batch of random noise vectors z ~ N(0, 1)."""
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)

    print(f"Generator     parameters: {count_parameters(G):,}")
    print(f"Discriminator parameters: {count_parameters(D):,}")

    z    = make_noise(4, device=device)
    fake = G(z)
    score = D(fake)

    print(f"\nNoise shape  : {z.shape}")
    print(f"Fake image   : {fake.shape}   range [{fake.min():.2f}, {fake.max():.2f}]")
    print(f"D score      : {score.shape}  → {score.detach().cpu().numpy().round(3)}")
