"""
Generate new flower sketches from a trained Generator.

Usage:
    # Generate 16 images into output/generated/
    python src/generate.py

    # Generate 64 images from a specific checkpoint
    python src/generate.py --checkpoint output/checkpoints/ckpt_epoch_1000.pt --n 64

    # Generate and interpolate between two random flowers (cool for demos!)
    python src/generate.py --interpolate --steps 10
"""

import os
import sys
import argparse
import torch
import torchvision.utils as vutils
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import Generator, make_noise, LATENT_DIM, N_CHANNELS


# ── Core generation ──────────────────────────────────────────────────────────

def load_generator(checkpoint_path: str, device) -> Generator:
    """Load a trained Generator from a checkpoint or state dict file."""
    G = Generator().to(device)

    state = torch.load(checkpoint_path, map_location=device)

    # Handle both full checkpoint dicts and bare state dicts
    if "G" in state:
        G.load_state_dict(state["G"])
        print(f"Loaded Generator from checkpoint (epoch {state.get('epoch', '?')+1})")
    else:
        G.load_state_dict(state)
        print("Loaded Generator state dict")

    G.eval()
    return G


def generate_grid(G, n: int, device, nrow: int = 4, out_path: str = "generated.png"):
    """Generate n images and save as a grid."""
    z    = make_noise(n, device=device)
    with torch.no_grad():
        imgs = G(z)
        imgs = (imgs + 1) / 2    # [-1,1] → [0,1]

    vutils.save_image(imgs, out_path, nrow=nrow, padding=4, normalize=False)
    print(f"Saved {n} images → {out_path}")
    return imgs


def generate_single(G, device, out_path: str = "flower.png"):
    """Generate one image and save it."""
    z = make_noise(1, device=device)
    with torch.no_grad():
        img = G(z)[0]
        img = (img + 1) / 2             # [0,1]
        img = img.squeeze().cpu().numpy()
        img = (img * 255).astype("uint8")
    pil = Image.fromarray(img, mode="L")
    pil.save(out_path)
    print(f"Saved single flower → {out_path}")
    return pil


def interpolate(G, device, steps: int = 10, out_path: str = "interpolation.png"):
    """
    Linear interpolation between two random latent vectors.
    This produces a smooth 'morphing' sequence between two flowers —
    great for visualising the latent space and for demos/presentations.
    """
    z1 = make_noise(1, device=device)
    z2 = make_noise(1, device=device)

    alphas = torch.linspace(0, 1, steps, device=device)
    frames = []

    with torch.no_grad():
        for a in alphas:
            z    = (1 - a) * z1 + a * z2
            img  = G(z)
            img  = (img + 1) / 2
            frames.append(img)

    grid = torch.cat(frames, dim=0)
    vutils.save_image(grid, out_path, nrow=steps, padding=4)
    print(f"Interpolation ({steps} steps) → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate flower sketches")
    p.add_argument("--checkpoint",   type=str, default="output/generator_final.pt")
    p.add_argument("--out_dir",      type=str, default="output/generated")
    p.add_argument("--n",            type=int, default=16,   help="Number of images")
    p.add_argument("--nrow",         type=int, default=4,    help="Images per row in grid")
    p.add_argument("--interpolate",  action="store_true",    help="Generate interpolation")
    p.add_argument("--steps",        type=int, default=10,   help="Interpolation steps")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    G = load_generator(args.checkpoint, device)

    if args.interpolate:
        interpolate(G, device, args.steps,
                    out_path=os.path.join(args.out_dir, "interpolation.png"))
    else:
        generate_grid(G, args.n, device, args.nrow,
                      out_path=os.path.join(args.out_dir, "grid.png"))
