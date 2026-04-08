"""
GAN architecture for Model 2 — trained on real flower photographs.
64x64 RGB output (matches Model 1 resolution).

Its Discriminator is reused as a "color critic" in colorize.py.
"""

import torch
import torch.nn as nn

LATENT_DIM   = 128
IMAGE_SIZE   = 64
N_CHANNELS   = 3
BASE_FEAT_G  = 64
BASE_FEAT_D  = 64


def _up(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _down(in_ch, out_ch):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


class PhotoGenerator(nn.Module):
    """
    Generates 64x64 RGB flower photographs from noise.
    Architecture: noise -> 4x4 -> 8 -> 16 -> 32 -> 64
    Mirrors model.py Generator but outputs 3 channels (RGB).
    """
    def __init__(self, latent_dim=LATENT_DIM, base_feat=BASE_FEAT_G):
        super().__init__()
        bf = base_feat
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, bf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(bf*8),
            nn.ReLU(inplace=True),
            _up(bf*8, bf*4),
            _up(bf*4, bf*2),
            _up(bf*2, bf),
            nn.ConvTranspose2d(bf, N_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def forward(self, z):
        return self.net(z)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)


class PhotoDiscriminator(nn.Module):
    """
    Discriminator for real vs generated flower photos.
    Reused as a color critic in colorize.py via extract_features().
    """
    def __init__(self, base_feat=BASE_FEAT_D):
        super().__init__()
        bf = base_feat

        self.block1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(N_CHANNELS, bf, 4, 2, 1, bias=False)   # 32x32
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = _down(bf,    bf*2)   # 16x16
        self.block3 = _down(bf*2,  bf*4)   #  8x8
        self.block4 = _down(bf*4,  bf*8)   #  4x4
        self.head   = nn.utils.spectral_norm(
            nn.Conv2d(bf*8, 1, 4, 1, 0, bias=False)
        )
        self._init_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x).view(-1)

    def extract_features(self, x):
        """
        Returns intermediate feature maps at multiple scales.
        Used as a perceptual color loss in colorize.py.
        """
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return [f1, f2, f3]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)


def make_noise(batch_size, latent_dim=LATENT_DIM, device="cpu"):
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = PhotoGenerator().to(device)
    D = PhotoDiscriminator().to(device)

    z     = make_noise(2, device=device)
    imgs  = G(z)
    score = D(imgs)
    feats = D.extract_features(imgs)

    print(f"Device          : {device}")
    print(f"Generated shape : {imgs.shape}")    # [2, 3, 64, 64]
    print(f"D score shape   : {score.shape}")
    print(f"Feature scales  : {[f.shape for f in feats]}")
