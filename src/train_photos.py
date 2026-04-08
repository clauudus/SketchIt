"""
Trains Model 2 on real flower photographs.
This model learns color distributions from nature to understant what 
flowers have (color-wise), how they shade, natural color relationships.

Its trained Discriminator is saved separately and reused in
colorize.py as a color critic.

Usage:
    python src/train_photos.py

Resume:
    python src/train_photos.py --resume output/photos/checkpoints/ckpt_epoch_1000.pt
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

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import FlowerSketchDataset
from model_photo import PhotoGenerator, PhotoDiscriminator, make_noise, LATENT_DIM


DEFAULTS = dict(
    data_dir     = "data/processed_photos",
    output_dir   = "output/photos",
    epochs       = 2000,
    batch_size   = 16,
    lr_g         = 2e-4,
    lr_d         = 1e-4,
    beta1        = 0.5,
    latent_dim   = LATENT_DIM,
    save_every   = 200,
    sample_every = 100,
    n_samples    = 16,
    noise_std    = 0.05,
    noise_decay  = 0.9999,
    seed         = 42,
)


def add_instance_noise(x, std):
    return x + torch.randn_like(x) * std if std > 0 else x


def save_samples(G, fixed_z, path, n_row=4):
    G.eval()
    with torch.no_grad():
        imgs = (G(fixed_z) + 1) / 2
    vutils.save_image(imgs, path, nrow=n_row, padding=2)
    G.train()


def train(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Flower Photo GAN  |  128×128 RGB  |  device: {device}")
    print(f"{'='*55}\n")

    dataset = FlowerSketchDataset(
        cfg["data_dir"],
        image_size = 64,
        augment    = True,
        color      = True,
    )
    loader = DataLoader(
        dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = 0,
        drop_last   = True,
    )
    print(f"Dataset: {len(dataset)} photos  |  Batches/epoch: {len(loader)}\n")

    G = PhotoGenerator(cfg["latent_dim"]).to(device)
    D = PhotoDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(cfg["beta1"], 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(cfg["beta1"], 0.999))

    def lr_lambda(epoch):
        warmup = int(cfg["epochs"] * 0.6)
        return 1.0 if epoch < warmup else max(0.1, 1.0 - (epoch - warmup) / cfg["epochs"])

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    criterion = nn.BCEWithLogitsLoss()
    fixed_z   = make_noise(cfg["n_samples"], cfg["latent_dim"], device)

    samples_dir = os.path.join(cfg["output_dir"], "samples")
    ckpt_dir    = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir,    exist_ok=True)

    start_epoch = 0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}\n")

    noise_std = cfg["noise_std"]

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        loss_d_sum = loss_g_sum = 0.0

        for real in loader:
            real  = real.to(device)
            bsize = real.size(0)

            real_labels = torch.ones(bsize,  device=device) * 0.9
            fake_labels = torch.zeros(bsize, device=device) + 0.1

            D.zero_grad()
            loss_d_real = criterion(D(add_instance_noise(real, noise_std)), real_labels)
            z           = make_noise(bsize, cfg["latent_dim"], device)
            fake        = G(z).detach()
            loss_d_fake = criterion(D(add_instance_noise(fake, noise_std)), fake_labels)
            loss_d      = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()

            G.zero_grad()
            z      = make_noise(bsize, cfg["latent_dim"], device)
            fake   = G(z)
            loss_g = criterion(D(fake), torch.ones(bsize, device=device))
            loss_g.backward()
            opt_G.step()

            loss_d_sum += loss_d.item()
            loss_g_sum += loss_g.item()

        noise_std *= cfg["noise_decay"]
        sched_G.step()
        sched_D.step()

        n_batches = len(loader)
        elapsed   = time.time() - t0
        print(
            f"Epoch {epoch+1:>4}/{cfg['epochs']}  |  "
            f"D: {loss_d_sum/n_batches:.4f}  "
            f"G: {loss_g_sum/n_batches:.4f}  |  "
            f"{elapsed:.1f}s"
        )

        if (epoch + 1) % cfg["sample_every"] == 0:
            path = os.path.join(samples_dir, f"epoch_{epoch+1:04d}.png")
            save_samples(G, fixed_z, path)
            print(f"  ✓ Samples → {path}")

        if (epoch + 1) % cfg["save_every"] == 0:
            path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, path)
            print(f"  ✓ Checkpoint → {path}")

    # Save final model AND the discriminator separately
    # (discriminator is reused as color critic in colorize.py)
    torch.save(G.state_dict(),
               os.path.join(cfg["output_dir"], "photo_generator_final.pt"))
    torch.save(D.state_dict(),
               os.path.join(cfg["output_dir"], "photo_discriminator_final.pt"))

    print("\nPhoto training complete!")
    print(f"Generator  → {cfg['output_dir']}/photo_generator_final.pt")
    print(f"Discriminator (color critic) → {cfg['output_dir']}/photo_discriminator_final.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train flower photo GAN")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(vars(args))
