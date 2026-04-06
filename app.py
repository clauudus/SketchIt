"""
Interactive flower generator GUI.
Run from the project root:

    python app.py

Controls:
  - Sketch button -> generate a black & white flower
  - Color buttons -> generate a flower tinted that color (phase 2)
  - New seed button -> randomise without changing color
  - Escape / ✕ -> close the window
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Try to import the model; graceful fallback if not trained yet ─────────────
try:
    from model import Generator, make_noise, LATENT_DIM
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_TITLE  = "Flower Generator"
CANVAS_SIZE   = 320          # Display size of the generated image
IMAGE_SIZE    = 64           # Must match training IMAGE_SIZE
CHECKPOINT    = "output/generator_final.pt"   # Default checkpoint path

COLORS = [
    ("Sketch",  None,      "#F5F5F5", "#222222"),   # label, hue, btn_bg, btn_fg
    ("Red",     "red",     "#FFEDED", "#CC2222"),
    ("Blue",    "blue",    "#EDF2FF", "#2255CC"),
    ("Yellow",  "yellow",  "#FFFBEA", "#AA8800"),
    ("Pink",    "pink",    "#FFECF5", "#CC3388"),
    ("Green",   "green",   "#EDFFF0", "#228833"),
    ("Purple",  "purple",  "#F3EDFF", "#6633CC"),
]

# Hue rotation amounts (degrees) for each color
HUE_SHIFT = {
    "red":    0,
    "blue":   200,
    "yellow": 55,
    "pink":   320,
    "green":  120,
    "purple": 270,
}


# ── Color tinting ─────────────────────────────────────────────────────────────

def apply_color(pil_img: Image.Image, color_name: str | None) -> Image.Image:
    """
    Tint a grayscale sketch image with a color.
    Uses multiply blending: white stays white, dark lines pick up the color.
    This is a preview of what will happen when you train on colored drawings.
    """
    if color_name is None:
        return pil_img.convert("RGB")

    img_rgb = pil_img.convert("RGB")

    color_map = {
        "red":    (220,  60,  60),
        "blue":   ( 60, 110, 220),
        "yellow": (220, 180,  30),
        "pink":   (220,  80, 160),
        "green":  ( 50, 180,  80),
        "purple": (130,  60, 220),
    }
    r, g, b = color_map.get(color_name, (128, 128, 128))

    # Create a solid color layer and multiply-blend with the sketch
    color_layer = Image.new("RGB", img_rgb.size, (r, g, b))
    img_arr     = np.array(img_rgb, dtype=np.float32) / 255.0
    col_arr     = np.array(color_layer, dtype=np.float32) / 255.0

    # Multiply blend: dark lines stay dark and colored, white background stays white
    blended = img_arr * col_arr + img_arr * (1 - col_arr) * 0.6
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class FlowerModel:
    def __init__(self):
        self.G      = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load(self, checkpoint_path: str) -> bool:
        if not MODEL_AVAILABLE:
            return False
        if not os.path.exists(checkpoint_path):
            return False
        try:
            G = Generator().to(self.device)
            state = torch.load(checkpoint_path, map_location=self.device)
            G.load_state_dict(state["G"] if "G" in state else state)
            G.eval()
            self.G      = G
            self.loaded = True
            return True
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            return False

    def generate(self, seed: int | None = None) -> Image.Image:
        """Generate one flower. Returns a PIL Image in grayscale."""
        if self.loaded and self.G is not None:
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                z   = make_noise(1, device=self.device)
                img = self.G(z)[0]                    # [1, H, W]
                img = (img + 1) / 2                   # [0, 1]
                img = img.squeeze().cpu().numpy()
                img = (img * 255).astype("uint8")
            return Image.fromarray(img, mode="L")
        else:
            # Placeholder: random noise sketch (until model is trained)
            return _demo_sketch(seed)


def _demo_sketch(seed: int | None = None) -> Image.Image:
    """
    Generates a placeholder 'sketch-like' image so the UI is usable
    before training is done. Replace this automatically once you have
    a trained checkpoint.
    """
    rng = np.random.default_rng(seed)
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 240

    # Draw a few random ellipses to suggest a flower
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    n_petals = rng.integers(5, 9)
    for i in range(n_petals):
        angle   = (2 * np.pi / n_petals) * i
        px      = int(cx + 18 * np.cos(angle))
        py      = int(cy + 18 * np.sin(angle))
        for x in range(IMAGE_SIZE):
            for y in range(IMAGE_SIZE):
                d = ((x - px) ** 2 / 64 + (y - py) ** 2 / 36)
                if d < 1:
                    img[y, x] = max(0, img[y, x] - (1 - d) * 180)

    # Centre dot
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):
            if (x - cx) ** 2 + (y - cy) ** 2 < 20:
                img[y, x] = 30

    img = np.clip(img, 0, 255).astype("uint8")
    return Image.fromarray(img, mode="L")


# ── Main App ──────────────────────────────────────────────────────────────────

class FlowerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(WINDOW_TITLE)
        self.resizable(False, False)
        self.configure(bg="#FAFAF8")
        self.bind("<Escape>", lambda e: self.quit())

        self.model         = FlowerModel()
        self.current_color = None     # None = sketch (grayscale)
        self.current_seed  = None
        self._photo        = None     # keep reference so GC doesn't collect it

        self._load_model()
        self._build_ui()
        self._generate()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        ok = self.model.load(CHECKPOINT)
        if ok:
            print(f"✓ Model loaded from {CHECKPOINT}")
        else:
            print("ℹ No trained model found — using placeholder sketches.")
            print(f"  Train first with:  python src/train.py")
            print(f"  Then run this app: python app.py")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=16, pady=10)

        # ── Title bar
        header = tk.Frame(self, bg="#FAFAF8")
        header.pack(fill="x", **pad)
        tk.Label(header, text="Flower Generator", font=("Helvetica", 16, "bold"),
                 bg="#FAFAF8", fg="#1A1A1A").pack(side="left")
        tk.Label(header, text="— your drawings, your model",
                 font=("Helvetica", 11), bg="#FAFAF8", fg="#888").pack(side="left", padx=6)
        tk.Button(header, text="✕", font=("Helvetica", 13), bg="#FAFAF8",
                  fg="#888", bd=0, relief="flat", cursor="hand2",
                  command=self.quit).pack(side="right")

        # ── Canvas
        canvas_frame = tk.Frame(self, bg="#EEECE6",
                                highlightbackground="#DDDBD3", highlightthickness=1)
        canvas_frame.pack(padx=16, pady=(0, 8))
        self.canvas = tk.Label(canvas_frame, bg="#EEECE6",
                               width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.pack(padx=12, pady=12)

        # ── Status label
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var,
                 font=("Helvetica", 10), bg="#FAFAF8", fg="#888").pack()

        # ── Color buttons row
        btn_frame = tk.Frame(self, bg="#FAFAF8")
        btn_frame.pack(padx=16, pady=(10, 4), fill="x")

        self.color_btns = {}
        for label, color, bg, fg in COLORS:
            b = tk.Button(
                btn_frame,
                text=label,
                font=("Helvetica", 10, "bold"),
                bg=bg, fg=fg,
                activebackground=bg, activeforeground=fg,
                bd=0, relief="flat",
                padx=10, pady=6,
                cursor="hand2",
                command=lambda c=color, l=label: self._on_color(c, l),
            )
            b.pack(side="left", padx=3)
            self.color_btns[label] = b

        self._highlight_color_btn("Sketch")

        # ── Action buttons
        action_frame = tk.Frame(self, bg="#FAFAF8")
        action_frame.pack(padx=16, pady=(4, 14), fill="x")

        tk.Button(
            action_frame,
            text="✦  New flower",
            font=("Helvetica", 11, "bold"),
            bg="#1A1A1A", fg="#FFFFFF",
            activebackground="#333", activeforeground="#FFF",
            bd=0, relief="flat",
            padx=18, pady=8,
            cursor="hand2",
            command=self._generate,
        ).pack(side="left")

        tk.Button(
            action_frame,
            text="Same seed",
            font=("Helvetica", 10),
            bg="#EEECE6", fg="#444",
            activebackground="#E0DDD7", activeforeground="#333",
            bd=0, relief="flat",
            padx=12, pady=8,
            cursor="hand2",
            command=self._recolor,
        ).pack(side="left", padx=8)

        tk.Button(
            action_frame,
            text="Save image",
            font=("Helvetica", 10),
            bg="#EEECE6", fg="#444",
            activebackground="#E0DDD7", activeforeground="#333",
            bd=0, relief="flat",
            padx=12, pady=8,
            cursor="hand2",
            command=self._save,
        ).pack(side="left")

        tk.Label(action_frame, text="Esc to quit",
                 font=("Helvetica", 9), bg="#FAFAF8", fg="#BBB").pack(side="right")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_color(self, color: str | None, label: str):
        self.current_color = color
        self._highlight_color_btn(label)
        self._recolor()    # keep the same flower, just change tint

    def _highlight_color_btn(self, active_label: str):
        for lbl, btn in self.color_btns.items():
            relief = "solid" if lbl == active_label else "flat"
            btn.config(relief=relief, bd=1 if lbl == active_label else 0)

    def _generate(self):
        """Generate a brand-new flower with a new random seed."""
        self.current_seed = torch.randint(0, 100_000, (1,)).item()
        self._render()

    def _recolor(self):
        """Re-apply current color to the existing seed (no new generation)."""
        self._render()

    def _render(self):
        """Generate + colorise + display."""
        self.status_var.set("Generating…")
        self.update_idletasks()

        sketch = self.model.generate(seed=self.current_seed)
        colored = apply_color(sketch, self.current_color)

        # Upscale for display (nearest-neighbor keeps the sketch aesthetic)
        display = colored.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)

        self._photo = ImageTk.PhotoImage(display)
        self.canvas.config(image=self._photo)

        color_label = self.current_color.capitalize() if self.current_color else "Sketch"
        self.status_var.set(f"{color_label} flower  ·  seed {self.current_seed}")

    def _save(self):
        if self._photo is None:
            return
        os.makedirs("output/generated", exist_ok=True)
        color_label = self.current_color or "sketch"
        path = f"output/generated/{color_label}_{self.current_seed}.png"

        # Regenerate the image at full quality and save
        sketch  = self.model.generate(seed=self.current_seed)
        colored = apply_color(sketch, self.current_color)
        colored.save(path)

        self.status_var.set(f"Saved → {path}")
        print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = FlowerApp()
    app.mainloop()
