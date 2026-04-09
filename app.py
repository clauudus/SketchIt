"""
Flower Generator:

Three buttons:
  Sketch   -> generate a new grayscale flower from Model 1
  Color it -> colorize the current sketch using the trained colorizer
  Escape   -> close the window

Run from project root:
    python app.py

After colorizing the sketch, if you want to colorize again you should generate a new sketch, then will get the colorizer button operative again.
"""

import os
import sys
import math
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Config ────────────────────────────────────────────────────────────────────

CANVAS_SIZE    = 360
IMAGE_SIZE     = 64

SKETCH_CKPT    = "output/checkpoints/ckpt_epoch_2000.pt"
COLORIZER_CKPT = "output/colorizer/colorizer_final.pt"

BG         = "#F7F6F2"
BTN_DARK   = "#1A1A1A"
BTN_LIGHT  = "#ECEAE3"
TEXT_MUTED = "#999"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_sketch_model(device):
    try:
        from model import Generator, LATENT_DIM
        G = Generator().to(device)
        if os.path.exists(SKETCH_CKPT):
            state = torch.load(SKETCH_CKPT, map_location=device)
            G.load_state_dict(state["G"] if "G" in state else state)
            G.eval()
            print(f"Sketch model loaded from {SKETCH_CKPT}")
            return G, LATENT_DIM, True
        print(f"No sketch checkpoint at {SKETCH_CKPT} — using placeholder")
        return None, 128, False
    except Exception as e:
        print(f"Could not load sketch model: {e}")
        return None, 128, False


def _load_colorizer(device):
    try:
        from colorize import Colorizer
        colorizer = Colorizer().to(device)
        if os.path.exists(COLORIZER_CKPT):
            colorizer.load_state_dict(
                torch.load(COLORIZER_CKPT, map_location=device))
            colorizer.eval()
            print(f"Colorizer loaded from {COLORIZER_CKPT}")
            return colorizer, True
        print(f"No colorizer at {COLORIZER_CKPT} — train with: python src/colorize.py --train")
        return None, False
    except Exception as e:
        print(f"Could not load colorizer: {e}")
        return None, False


# ── Placeholder sketch (when model not trained yet) ───────────────────────────

def _placeholder_sketch(seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    img = (np.ones((IMAGE_SIZE, IMAGE_SIZE)) * 240).astype("float32")
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    n = rng.integers(5, 9)
    for i in range(n):
        angle = (2 * math.pi / n) * i
        px = int(cx + 18 * math.cos(angle))
        py = int(cy + 18 * math.sin(angle))
        for x in range(IMAGE_SIZE):
            for y in range(IMAGE_SIZE):
                d = (x - px)**2 / 60 + (y - py)**2 / 32
                if d < 1:
                    img[y, x] = max(0, img[y, x] - (1 - d) * 180)
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):
            if (x - cx)**2 + (y - cy)**2 < 18:
                img[y, x] = 40
    return Image.fromarray(img.clip(0, 255).astype("uint8"), mode="L")


# ── Main App ──────────────────────────────────────────────────────────────────

class FlowerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flower Generator")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.bind("<Escape>", lambda e: self.quit())

        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_seed   = None
        self.current_sketch = None   # PIL Image grayscale
        self._photo         = None   # tkinter reference

        self.G, self.latent_dim, self.sketch_ready = _load_sketch_model(self.device)
        self.colorizer, self.color_ready = _load_colorizer(self.device)

        self._build_ui()
        self._generate_sketch()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(
            header, text="Flower Generator",
            font=("Helvetica", 15, "bold"), bg=BG, fg="#1A1A1A"
        ).pack(side="left")
        tk.Button(
            header, text="✕",
            font=("Helvetica", 13), bg=BG, fg=TEXT_MUTED,
            bd=0, relief="flat", cursor="hand2",
            command=self.quit
        ).pack(side="right")

        # Canvas
        canvas_wrap = tk.Frame(
            self, bg="#E8E6DF",
            highlightbackground="#D8D6CF", highlightthickness=1
        )
        canvas_wrap.pack(padx=18, pady=12)
        self.canvas_label = tk.Label(
            canvas_wrap, bg="#E8E6DF",
            width=CANVAS_SIZE, height=CANVAS_SIZE
        )
        self.canvas_label.pack(padx=10, pady=10)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self, textvariable=self.status_var,
            font=("Helvetica", 10), bg=BG, fg=TEXT_MUTED
        ).pack()

        # Buttons
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(padx=18, pady=(10, 16), fill="x")

        tk.Button(
            btn_row,
            text="✦  Sketch",
            font=("Helvetica", 11, "bold"),
            bg=BTN_DARK, fg="#FFFFFF",
            activebackground="#333", activeforeground="#FFF",
            bd=0, relief="flat", padx=20, pady=9,
            cursor="hand2",
            command=self._generate_sketch,
        ).pack(side="left")

        self.btn_color = tk.Button(
            btn_row,
            text="Color it",
            font=("Helvetica", 11),
            bg=BTN_LIGHT, fg="#444",
            activebackground="#DDDBD4", activeforeground="#222",
            disabledforeground="#BBBBBB",
            bd=0, relief="flat", padx=20, pady=9,
            cursor="hand2" if self.color_ready else "arrow",
            command=self._colorize,
            state="normal" if self.color_ready else "disabled",
        )
        self.btn_color.pack(side="left", padx=10)

        tk.Button(
            btn_row,
            text="Escape",
            font=("Helvetica", 11),
            bg=BTN_LIGHT, fg="#444",
            activebackground="#DDDBD4", activeforeground="#222",
            bd=0, relief="flat", padx=20, pady=9,
            cursor="hand2",
            command=self.quit,
        ).pack(side="left")

        if not self.color_ready:
            tk.Label(
                self,
                text="Color it — available after: python src/colorize.py --train",
                font=("Helvetica", 9), bg=BG, fg=TEXT_MUTED
            ).pack(pady=(0, 6))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _generate_sketch(self):
        self.current_seed = torch.randint(0, 100_000, (1,)).item()
        self.status_var.set("Generating…")
        self.update_idletasks()

        if self.sketch_ready and self.G is not None:
            torch.manual_seed(self.current_seed)
            with torch.no_grad():
                from model import make_noise
                z   = make_noise(1, self.latent_dim, self.device)
                img = self.G(z)[0]
                img = (img + 1) / 2
                arr = (img.squeeze().cpu().numpy() * 255).astype("uint8")
            self.current_sketch = Image.fromarray(arr, mode="L")
        else:
            self.current_sketch = _placeholder_sketch(self.current_seed)

        self._display(self.current_sketch.convert("RGB"))
        self.status_var.set(f"Sketch  ·  seed {self.current_seed}")

        # Re-enable Color it for the new sketch
        if self.color_ready:
            self.btn_color.config(state="normal")

    def _colorize(self):
        if self.current_sketch is None or not self.color_ready:
            return

        self.status_var.set("Colorizing…")
        self.update_idletasks()

        with torch.no_grad():
            sketch_t = TF.to_tensor(
                self.current_sketch.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            ).unsqueeze(0) * 2 - 1
            sketch_t = sketch_t.to(self.device)

            colored = self.colorizer(sketch_t)[0]
            colored = (colored + 1) / 2
            arr     = (colored.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        self._display(Image.fromarray(arr, mode="RGB"))
        self.status_var.set(f"Colored  ·  seed {self.current_seed}")
        self.btn_color.config(state="disabled")   # greyed out until next sketch

    def _display(self, pil_img: Image.Image):
        display     = pil_img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(display)
        self.canvas_label.config(image=self._photo)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FlowerApp().mainloop()
