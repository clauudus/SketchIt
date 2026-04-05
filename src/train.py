"""
Training loop for the Flower Sketch DCGAN.

Run:
    python src/train.py

Checkpoints and sample images are saved automatically.
Resume training from a checkpoint with --resume.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import FlowerSketchDataset, IMAGE_SIZE
from model import Generator, Discriminator, make_noise, LATENT_DIM, N_CHANNELS


# ── Default config ────────────────────────────────────────────────────────────

DEFAULTS = dict(
    data_dir      = "data/processed",
    output_dir    = "output",
    epochs        = 2000,       # With ~100 images, run long — GANs need it
    batch_size    = 16,         # Small batch works well for small datasets
    lr_g          = 2e-4,       # Learning rate Generator
    lr_d          = 1e-4,       # Discriminator learns a bit slower → stability
    beta1         = 0.5,        # Adam β₁ (0.5 is standard for GANs)
    latent_dim    = LATENT_DIM,
    n_channels    = N_CHANNELS,
    save_every    = 100,        # Save checkpoint every N epochs
    sample_every  = 50,         # Save a grid of generated images every N epochs
    n_samples     = 16,         # Images in the sample grid
    noise_std     = 0.05,       # Instance noise added to D inputs (reduces overfitting)
    noise_decay   = 0.9999,     # Noise anneals to 0 over training
    seed          = 42,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_instance_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Add small Gaussian noise to Discriminator inputs. Annealed over time."""
    if std > 0:
        return x + torch.randn_like(x) * std
    return x


def save_samples(G, fixed_z, path, n_row=4):
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z)          # [-1, 1]
        imgs = (imgs + 1) / 2      # [0, 1]
    vutils.save_image(imgs, path, nrow=n_row, padding=2, normalize=False)
    G.train()


def gradient_penalty(D, real, fake, device):
    """WGAN-GP gradient penalty for extra training stability (optional path)."""
    alpha  = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out  = D(interp)
    grads  = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    grads  = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Flower Sketch GAN  |  device: {device}")
    print(f"{'='*55}\n")

    # ── Data
    dataset = FlowerSketchDataset(cfg["data_dir"], image_size=IMAGE_SIZE)
    loader  = DataLoader(
        dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = 0,          # Set to 2-4 if your system supports it
        drop_last   = True,       # Keep batch sizes consistent
    )
    print(f"Dataset: {len(dataset)} images  |  "
          f"Batches per epoch: {len(loader)}\n")

    # ── Models
    G = Generator(cfg["latent_dim"], n_channels=cfg["n_channels"]).to(device)
    D = Discriminator(n_channels=cfg["n_channels"]).to(device)

    # ── Optimisers  (Adam with β₁=0.5 is the GAN standard)
    opt_G = optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(cfg["beta1"], 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(cfg["beta1"], 0.999))

    # ── LR schedulers — gently decay after 60% of training
    def lr_lambda(epoch):
        warmup = int(cfg["epochs"] * 0.6)
        return 1.0 if epoch < warmup else max(0.2, 1.0 - (epoch - warmup) / cfg["epochs"])

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # ── Loss
    criterion = nn.BCEWithLogitsLoss()

    # ── Fixed noise for consistent sample grids
    fixed_z = make_noise(cfg["n_samples"], cfg["latent_dim"], device)

    # ── Output dirs
    samples_dir = os.path.join(cfg["output_dir"], "samples")
    ckpt_dir    = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir,    exist_ok=True)

    # ── Resume from checkpoint
    start_epoch = 0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    noise_std = cfg["noise_std"]

    # ── Main loop
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        loss_d_sum = loss_g_sum = 0.0

        for real in loader:
            real  = real.to(device)
            bsize = real.size(0)

            real_labels = torch.ones(bsize,  device=device) * 0.9   # Label smoothing
            fake_labels = torch.zeros(bsize, device=device) + 0.1   # Prevents D becoming too confident

            # ────────────────────────────────────────────
            # Train Discriminator
            # ────────────────────────────────────────────
            D.zero_grad()

            # Real images
            real_noisy = add_instance_noise(real, noise_std)
            d_real     = D(real_noisy)
            loss_d_real = criterion(d_real, real_labels)

            # Fake images
            z    = make_noise(bsize, cfg["latent_dim"], device)
            fake = G(z).detach()        # detach → don't backprop into G yet
            fake_noisy = add_instance_noise(fake, noise_std)
            d_fake     = D(fake_noisy)
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()

            # ────────────────────────────────────────────
            # Train Generator  (every step — small dataset)
            # ────────────────────────────────────────────
            G.zero_grad()

            z    = make_noise(bsize, cfg["latent_dim"], device)
            fake = G(z)
            d_fake_for_g = D(fake)

            # Generator wants D to think fakes are real → target = 1
            loss_g = criterion(d_fake_for_g, torch.ones(bsize, device=device))
            loss_g.backward()
            opt_G.step()

            loss_d_sum += loss_d.item()
            loss_g_sum += loss_g.item()

        # Anneal instance noise
        noise_std *= cfg["noise_decay"]

        sched_G.step()
        sched_D.step()

        # ── Logging
        n_batches = len(loader)
        elapsed   = time.time() - t0
        print(
            f"Epoch {epoch+1:>4}/{cfg['epochs']}  |  "
            f"D: {loss_d_sum/n_batches:.4f}  "
            f"G: {loss_g_sum/n_batches:.4f}  |  "
            f"{elapsed:.1f}s"
        )

        # ── Save sample images
        if (epoch + 1) % cfg["sample_every"] == 0:
            path = os.path.join(samples_dir, f"epoch_{epoch+1:04d}.png")
            save_samples(G, fixed_z, path)
            print(f"  ✓ Samples saved → {path}")

        # ── Save checkpoint
        if (epoch + 1) % cfg["save_every"] == 0:
            path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch,
                "G":     G.state_dict(),
                "D":     D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, path)
            print(f"  ✓ Checkpoint saved → {path}")

    print("\nTraining complete!")
    # Save final model
    torch.save(G.state_dict(), os.path.join(cfg["output_dir"], "generator_final.pt"))
    print(f"Final generator saved → {cfg['output_dir']}/generator_final.pt")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Flower Sketch GAN")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    args = p.parse_args()
    train(vars(args))
