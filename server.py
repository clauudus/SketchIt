"""
Local web server.
Run from project root:

    python server.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import time
import glob
import threading
import subprocess
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

app = Flask(__name__, static_folder="web/static", template_folder="web")

# Paths

SKETCH_UPLOAD_DIR  = "data/raw"
PHOTO_UPLOAD_DIR   = "data/photos_raw"
PROCESSED_SKETCH   = "data/processed"
PROCESSED_PHOTO    = "data/processed_photos"
SKETCH_CKPT        = "output/checkpoints/ckpt_epoch_2000.pt"
COLORIZER_CKPT     = "output/colorizer/colorizer_final.pt"
GENERATED_DIR      = "output/colorizer/generated"
AGREEMENT_FLAG     = "output/.agreement_signed"

for d in [SKETCH_UPLOAD_DIR, PHOTO_UPLOAD_DIR, PROCESSED_SKETCH,
          PROCESSED_PHOTO, GENERATED_DIR,
          "output/checkpoints", "output/colorizer",
          "output/photos", "web/static"]:
    os.makedirs(d, exist_ok=True)

# Training state

training_state = {
    "running":    False,
    "phase":      "",
    "epoch":      0,
    "total":      0,
    "loss":       "",
    "message":    "",
    "error":      "",
    "start_time": None,
}


# Model detection

def _find_best_sketch_ckpt():
    """
    Returns the path to the best available sketch checkpoint.
    Prefers ckpt_epoch_2000.pt, then the highest-epoch checkpoint found,
    then generator_final.pt as a fallback.
    """
    # First choice: exact expected path
    if os.path.exists(SKETCH_CKPT):
        return SKETCH_CKPT

    # Second: scan for any checkpoint in output/checkpoints/
    pattern = "output/checkpoints/ckpt_epoch_*.pt"
    found = sorted(glob.glob(pattern))
    if found:
        # Pick the one with the highest epoch number
        def epoch_num(p):
            try:
                return int(os.path.basename(p).replace("ckpt_epoch_", "").replace(".pt", ""))
            except Exception:
                return 0
        return max(found, key=epoch_num)

    # Third: generator_final.pt
    final = "output/generator_final.pt"
    if os.path.exists(final):
        return final

    return None


def _find_colorizer_ckpt():
    """Returns the colorizer checkpoint path if any version exists."""
    if os.path.exists(COLORIZER_CKPT):
        return COLORIZER_CKPT

    # Check for intermediate checkpoints
    pattern = "output/colorizer/colorizer_epoch_*.pt"
    found = sorted(glob.glob(pattern))
    if found:
        def epoch_num(p):
            try:
                return int(os.path.basename(p)
                           .replace("colorizer_epoch_", "").replace(".pt", ""))
            except Exception:
                return 0
        return max(found, key=epoch_num)

    return None


def get_model_status():
    """
    Returns a dict describing exactly what is trained and available.
    This is the single source of truth used by both the API and generation routes.
    """
    sketch_ckpt    = _find_best_sketch_ckpt()
    colorizer_ckpt = _find_colorizer_ckpt()

    sketch_ready    = sketch_ckpt is not None
    colorizer_ready = colorizer_ckpt is not None
    fully_trained   = sketch_ready and colorizer_ready

    return {
        "sketch_ready":    sketch_ready,
        "colorizer_ready": colorizer_ready,
        "fully_trained":   fully_trained,
        "sketch_ckpt":     sketch_ckpt,
        "colorizer_ckpt":  colorizer_ckpt,
        "sketch_ckpt_name": os.path.basename(sketch_ckpt) if sketch_ckpt else None,
        "colorizer_ckpt_name": os.path.basename(colorizer_ckpt) if colorizer_ckpt else None,
    }


def agreement_signed():
    return os.path.exists(AGREEMENT_FLAG)


# Routes - pages

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


# Routes - agreement

@app.route("/api/agreement", methods=["POST"])
def sign_agreement():
    with open(AGREEMENT_FLAG, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    return jsonify({"ok": True})


@app.route("/api/agreement", methods=["GET"])
def check_agreement():
    return jsonify({"signed": agreement_signed()})


# Routes - status (replaces /api/trained)

@app.route("/api/status")
def status():
    """
    Main status endpoint. Returns everything the frontend needs to know
    about what is trained and what buttons should be unlocked.
    """
    ms = get_model_status()
    ts = dict(training_state)
    return jsonify({**ms, "training": ts})


@app.route("/api/trained")
def is_trained_route():
    ms = get_model_status()
    return jsonify({"trained": ms["fully_trained"], **ms})


# Routes - uploads

@app.route("/api/upload/sketches", methods=["POST"])
def upload_sketches():
    return _handle_upload(request.files.getlist("files"), SKETCH_UPLOAD_DIR)


@app.route("/api/upload/photos", methods=["POST"])
def upload_photos():
    return _handle_upload(request.files.getlist("files"), PHOTO_UPLOAD_DIR)


def _handle_upload(files, dest_dir):
    if not files:
        return jsonify({"error": "No files received"}), 400
    saved = []
    for f in files[:100]:
        if f.filename and f.filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            path = os.path.join(dest_dir, f.filename)
            f.save(path)
            saved.append(f.filename)
    return jsonify({"saved": len(saved), "files": saved})


@app.route("/api/count/sketches")
def count_sketches():
    return jsonify({"count": len(_image_files(SKETCH_UPLOAD_DIR))})


@app.route("/api/count/photos")
def count_photos():
    return jsonify({"count": len(_image_files(PHOTO_UPLOAD_DIR))})


def _image_files(d):
    if not os.path.exists(d):
        return []
    return [f for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]


# Routes - training

@app.route("/api/train", methods=["POST"])
def start_training():
    if training_state["running"]:
        return jsonify({"error": "Already training"}), 400
    if not agreement_signed():
        return jsonify({"error": "Agreement not signed"}), 403

    data = request.json or {}
    sketch_epochs    = int(data.get("sketch_epochs",    2000))
    photo_epochs     = int(data.get("photo_epochs",     2000))
    colorizer_epochs = int(data.get("colorizer_epochs",  500))

    t = threading.Thread(
        target=_run_training,
        args=(sketch_epochs, photo_epochs, colorizer_epochs),
        daemon=True
    )
    t.start()
    return jsonify({"ok": True})


@app.route("/api/train/status")
def training_status():
    return jsonify(dict(training_state))


@app.route("/api/train/stream")
def training_stream():
    def generate():
        last = {}
        while True:
            current = dict(training_state)
            if current != last:
                yield f"data: {json.dumps(current)}\n\n"
                last = dict(current)
            if current["phase"] in ("done", "error"):
                break
            time.sleep(0.5)
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def _update(phase="", epoch=0, total=0, loss="", message="", error=""):
    training_state.update({
        "phase": phase, "epoch": epoch, "total": total,
        "loss": loss, "message": message, "error": error,
    })


def _run_training(sketch_epochs, photo_epochs, colorizer_epochs):
    training_state["running"]    = True
    training_state["start_time"] = time.time()
    try:
        _update("sketches", message="Cleaning your sketches...")
        from preprocess import batch_clean, batch_clean_color, IMAGE_SIZE
        batch_clean(SKETCH_UPLOAD_DIR, PROCESSED_SKETCH, size=IMAGE_SIZE)

        _update("photos_prep", message="Preparing photos...")
        batch_clean_color(PHOTO_UPLOAD_DIR, PROCESSED_PHOTO, size=64)

        _update("sketches", 0, sketch_epochs, message="Training your sketch model...")
        _train_model("src/train.py", sketch_epochs, "sketches",
                     ["--data_dir", PROCESSED_SKETCH,
                      "--save_every", "500", "--sample_every", "200"])

        _update("photos", 0, photo_epochs, message="Learning colours from photos...")
        _train_model("src/train_photos.py", photo_epochs, "photos",
                     ["--data_dir", PROCESSED_PHOTO,
                      "--save_every", "500", "--sample_every", "200"])

        _update("colorizer", 0, colorizer_epochs, message="Training the colorizer...")
        _train_model("src/colorize.py", colorizer_epochs, "colorizer",
                     ["--train", "--sketch_data", PROCESSED_SKETCH,
                      "--photo_data", PROCESSED_PHOTO,
                      "--save_every", "100", "--sample_every", "25"])

        _update("done", message="Training complete! You can now generate drawings.")

    except Exception as e:
        _update("error", error=str(e), message="Training failed.")
        import traceback; traceback.print_exc()
    finally:
        training_state["running"] = False


def _train_model(script, epochs, phase_label, extra_args=None):
    cmd = [sys.executable, script, "--epochs", str(epochs)] + (extra_args or [])
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Epoch"):
            parts = line.split("/")
            try:
                epoch = int(parts[0].replace("Epoch", "").strip())
                total = int(parts[1].split()[0])
                loss  = line.split("|")[1].strip() if "|" in line else ""
                _update(phase_label, epoch, total, loss=loss,
                        message=f"Training ({phase_label})...")
            except Exception:
                pass
        print(line)
    proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"{script} exited with code {proc.returncode}")


# Routes - generation

@app.route("/api/generate/sketch", methods=["POST"])
def generate_sketch():
    ms = get_model_status()
    if not ms["sketch_ready"]:
        return jsonify({"error": "Sketch model not trained yet"}), 400
    try:
        import torch
        from model import Generator, make_noise, LATENT_DIM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G = Generator().to(device)
        state = torch.load(ms["sketch_ckpt"], map_location=device)
        G.load_state_dict(state["G"] if "G" in state else state)
        G.eval()

        seed = torch.randint(0, 100_000, (1,)).item()
        torch.manual_seed(seed)
        with torch.no_grad():
            z   = make_noise(1, LATENT_DIM, device)
            img = G(z)[0]
            img = (img + 1) / 2
            arr = (img.squeeze().cpu().numpy() * 255).astype("uint8")

        from PIL import Image
        pil = Image.fromarray(arr, mode="L").resize((256, 256), Image.NEAREST)
        fname = f"sketch_{seed}.png"
        pil.save(os.path.join(GENERATED_DIR, fname))

        return jsonify({
            "seed": seed,
            "image": f"/output/{fname}",
            "colorizer_ready": ms["colorizer_ready"],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/color", methods=["POST"])
def colorize():
    ms = get_model_status()
    if not ms["sketch_ready"]:
        return jsonify({"error": "Sketch model not ready"}), 400
    if not ms["colorizer_ready"]:
        return jsonify({"error": "Colorizer not trained yet"}), 400

    data = request.json or {}
    seed = data.get("seed")
    if seed is None:
        return jsonify({"error": "No seed provided"}), 400
    try:
        import torch
        from model import Generator, make_noise, LATENT_DIM
        from colorize import Colorizer
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        G = Generator().to(device)
        state = torch.load(ms["sketch_ckpt"], map_location=device)
        G.load_state_dict(state["G"] if "G" in state else state)
        G.eval()

        colorizer = Colorizer().to(device)
        colorizer.load_state_dict(
            torch.load(ms["colorizer_ckpt"], map_location=device))
        colorizer.eval()

        torch.manual_seed(seed)
        with torch.no_grad():
            z       = make_noise(1, LATENT_DIM, device)
            sketch  = G(z)
            colored = colorizer(sketch)
            colored = (colored + 1) / 2
            arr = (colored[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        pil = Image.fromarray(arr, mode="RGB").resize((256, 256), Image.NEAREST)
        fname = f"colored_{seed}.png"
        pil.save(os.path.join(GENERATED_DIR, fname))

        return jsonify({"seed": seed, "image": f"/output/{fname}"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(GENERATED_DIR, filename)


# Startup report

def _startup_report():
    ms = get_model_status()
    print("\n" + "="*50)
    print("  Sketchit — Web App")
    print("  Open http://localhost:5000 in your browser")
    print("="*50)
    print(f"  Sketch model  : {'✓ ' + ms['sketch_ckpt_name'] if ms['sketch_ready'] else '✗ not found'}")
    print(f"  Colorizer     : {'✓ ' + ms['colorizer_ckpt_name'] if ms['colorizer_ready'] else '✗ not found'}")
    print(f"  Agreement     : {'✓ signed' if agreement_signed() else '— not signed yet'}")
    print("="*50 + "\n")


if __name__ == "__main__":
    _startup_report()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
