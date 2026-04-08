"""
The bridge between Model 1 (sketch) and Model 2 (photo).
All images are 64x64.

Training:
    python src/colorize.py --train

Colorize a generated sketch:
    python src/colorize.py --sketch_seed 42

Colorize an existing image file:
    python src/colorize.py --image path/to/sketch.png
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import Generator, make_noise, LATENT_DIM
from model_photo import PhotoDiscriminator
from preprocess import FlowerSketchDataset

IMAGE_SIZE = 64

DEFAULTS = dict(
    sketch_ckpt    = "output/checkpoints/ckpt_epoch_2000.pt",
    photo_d_ckpt   = "output/photos/photo_discriminator_final.pt",
    colorizer_ckpt = "output/colorizer/colorizer_final.pt",
    sketch_data    = "data/processed",
    photo_data     = "data/processed_photos",
    output_dir     = "output/colorizer",
    epochs         = 1000,
    batch_size     = 8,
    lr             = 1e-4,
    lambda_color   = 10.0,
    lambda_l1      = 1.0,
    save_every     = 200,
    sample_every   = 50,
    seed           = 42,
)


# ── U-Net Colorizer ───────────────────────────────────────────────────────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        conv   = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False) if down \
            else nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        layers = [conv]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.LeakyReLU(0.2) if down else nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Colorizer(nn.Module):
    """
    U-Net: grayscale sketch [1, 64, 64] -> color image [3, 64, 64].
    Skip connections preserve sketch line structure while the
    decoder fills in color guided by the photo color critic.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = UNetBlock(1,   64,  down=True, use_bn=False)  # 32x32
        self.e2 = UNetBlock(64,  128, down=True)                 # 16x16
        self.e3 = UNetBlock(128, 256, down=True)                 #  8x8
        self.e4 = UNetBlock(256, 512, down=True)                 #  4x4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),            #  2x2
            nn.ReLU(),
        )

        # Decoder with skip connections
        self.d1 = UNetBlock(512,  512, down=False, dropout=True)  #  4x4
        self.d2 = UNetBlock(1024, 256, down=False)                #  8x8
        self.d3 = UNetBlock(512,  128, down=False)                # 16x16
        self.d4 = UNetBlock(256,  64,  down=False)                # 32x32

        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),     # 64x64
            nn.Tanh(),
        )
        self._init_weights()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        bn = self.bottleneck(e4)
        d1 = self.d1(bn)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        return self.out(torch.cat([d4, e1], dim=1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)


# ── Paired dataset ────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, size=IMAGE_SIZE):
        self.size = size
        self.sketches = sorted([
            os.path.join(sketch_dir, f)
            for f in os.listdir(sketch_dir) if f.endswith(".png")
        ])
        self.photos = sorted([
            os.path.join(photo_dir, f)
            for f in os.listdir(photo_dir) if f.endswith(".png")
        ])
        if not self.sketches:
            raise RuntimeError(f"No sketches in '{sketch_dir}'")
        if not self.photos:
            raise RuntimeError(f"No photos in '{photo_dir}'")
        print(f"Colorizer dataset: {len(self.sketches)} sketches, {len(self.photos)} photos")

    def __len__(self):
        return max(len(self.sketches), len(self.photos))

    def __getitem__(self, idx):
        sketch = Image.open(self.sketches[idx % len(self.sketches)]).convert("L") \
                      .resize((self.size, self.size), Image.LANCZOS)
        photo  = Image.open(self.photos[idx % len(self.photos)]).convert("RGB") \
                      .resize((self.size, self.size), Image.LANCZOS)
        return TF.to_tensor(sketch) * 2 - 1, TF.to_tensor(photo) * 2 - 1


# ── Color critic loss ─────────────────────────────────────────────────────────

def color_critic_loss(D_photo, colorized, real_photos, lambda_color):
    feats_col  = D_photo.extract_features(colorized)
    feats_real = D_photo.extract_features(real_photos)
    loss = sum(nn.functional.l1_loss(fc, fr.detach())
               for fc, fr in zip(feats_col, feats_real))
    return loss * lambda_color


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Colorizer Training  |  64x64  |  device: {device}")
    print(f"{'='*55}\n")

    D_photo = PhotoDiscriminator().to(device)
    if os.path.exists(cfg["photo_d_ckpt"]):
        D_photo.load_state_dict(torch.load(cfg["photo_d_ckpt"], map_location=device))
        print(f"Color critic loaded from {cfg['photo_d_ckpt']}")
    else:
        print(f"No photo discriminator found — train Model 2 first:")
        print(f"  python src/train_photos.py")

    for p in D_photo.parameters():
        p.requires_grad = False
    D_photo.eval()

    colorizer = Colorizer().to(device)
    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        colorizer.load_state_dict(torch.load(cfg["resume"], map_location=device))
        print(f"Resumed from {cfg['resume']}")

    opt     = optim.Adam(colorizer.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    dataset = SketchPhotoDataset(cfg["sketch_data"], cfg["photo_data"])
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"],
                         shuffle=True, num_workers=0, drop_last=True)

    samples_dir = os.path.join(cfg["output_dir"], "samples")
    os.makedirs(samples_dir,        exist_ok=True)
    os.makedirs(cfg["output_dir"],  exist_ok=True)

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        total_loss = 0.0

        for sketches, photos in loader:
            sketches  = sketches.to(device)
            photos    = photos.to(device)
            colorized = colorizer(sketches)

            l1   = nn.functional.l1_loss(colorized, photos) * cfg["lambda_l1"]
            cc   = color_critic_loss(D_photo, colorized, photos, cfg["lambda_color"])
            loss = l1 + cc

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1:>4}/{cfg['epochs']}  |  "
            f"Loss: {total_loss/len(loader):.4f}  |  "
            f"{time.time()-t0:.1f}s"
        )

        if (epoch + 1) % cfg["sample_every"] == 0:
            colorizer.eval()
            with torch.no_grad():
                s   = sketches[:4]
                col = (colorizer(s) + 1) / 2
                sk  = (s.expand(-1, 3, -1, -1) + 1) / 2
                vutils.save_image(
                    torch.cat([sk, col], dim=0),
                    os.path.join(samples_dir, f"epoch_{epoch+1:04d}.png"),
                    nrow=4, padding=2
                )
            colorizer.train()
            print(f"  Samples -> {samples_dir}/epoch_{epoch+1:04d}.png")

        if (epoch + 1) % cfg["save_every"] == 0:
            path = os.path.join(cfg["output_dir"], f"colorizer_epoch_{epoch+1:04d}.pt")
            torch.save(colorizer.state_dict(), path)
            print(f"  Checkpoint -> {path}")

    torch.save(colorizer.state_dict(), cfg["colorizer_ckpt"])
    print(f"\nColorizer saved -> {cfg['colorizer_ckpt']}")


# ── Inference ─────────────────────────────────────────────────────────────────

def colorize_sketch(cfg, sketch_seed=None, image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output/colorizer/generated", exist_ok=True)

    G_sketch = Generator().to(device)
    state    = torch.load(cfg["sketch_ckpt"], map_location=device)
    G_sketch.load_state_dict(state["G"] if "G" in state else state)
    G_sketch.eval()

    colorizer = Colorizer().to(device)
    colorizer.load_state_dict(torch.load(cfg["colorizer_ckpt"], map_location=device))
    colorizer.eval()

    with torch.no_grad():
        if image_path:
            img    = Image.open(image_path).convert("L") \
                         .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            sketch = TF.to_tensor(img).unsqueeze(0) * 2 - 1
            sketch = sketch.to(device)
            label  = os.path.splitext(os.path.basename(image_path))[0]
        else:
            if sketch_seed is not None:
                torch.manual_seed(sketch_seed)
            z      = make_noise(1, device=device)
            sketch = G_sketch(z)
            label  = f"seed_{sketch_seed or 'random'}"

        colored     = colorizer(sketch)
        sketch_rgb  = (sketch.expand(-1, 3, -1, -1) + 1) / 2
        colored_out = (colored + 1) / 2

        out_path = f"output/colorizer/generated/{label}_colored.png"
        vutils.save_image(
            torch.cat([sketch_rgb, colored_out], dim=0),
            out_path, nrow=2, padding=4
        )
        print(f"Saved: {out_path}  (left: sketch  right: colored)")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--train",       action="store_true")
    p.add_argument("--sketch_seed", type=int,  default=None)
    p.add_argument("--image",       type=str,  default=None)
    p.add_argument("--resume",      type=str,  default=None)
    args = p.parse_args()
    cfg  = vars(args)

    if args.train:
        train(cfg)
    elif args.sketch_seed is not None or args.image:
        colorize_sketch(cfg, sketch_seed=args.sketch_seed, image_path=args.image)
    else:
        print("Usage:")
        print("  Train:               python src/colorize.py --train")
        print("  Colorize new sketch: python src/colorize.py --sketch_seed 42")
        print("  Colorize image:      python src/colorize.py --image path/to/sketch.png")
