# DrawingMLmodel
What if the artist consents? <br/>
A generative AI model trained exclusively on one artist's own drawings, created to explore the ethics of AI art generation. <br/>

## The Idea
AI image generation is surrounded by controversy: artists' work is scraped and used to train models without consent. This project flips that question. What does it look like when a single artist deliberately trains a model only on their own work? <br/>
The output isn't just a generator; it's a statement about authorship, consent, and what "AI-generated art" can mean when the human behind it is fully in control of the dataset. <br/>

## Quick Start
```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your scanned sketches in data/raw/
# If the folder doesn't exist, you can create it :), I'll be adding it soon

# 3. Set up and clean the data
python scripts/setup_data.py

# 4. Train the model
python src/train.py

# 5. Generate new flowers!
python src/generate.py
```

## How it works
The model is a DCGAN (Deep Convolutional Generative Adversarial Network) adapted for small datasets: <br/>

- Generator: Takes random noise → outputs a 64×64 flower sketch<br/>
- Discriminator: Learns to tell real sketches from generated ones<br/>
- They compete until the Generator gets good enough to fool the Discriminator<br/>

### Small Dataset Adaptations (around a 100 images)
- Heavy augmentations (flips, rotations, crop, etc) -> To make the 100 images "feel" like 1000 <br/>
- Spectral normalisation -> Stabilises training <br/>
- Instance noise in D inputs -> Prevents discriminator from overpowering generator <br/>
- Label smoothing -> Stops the model from becoming "overconfident" <br/>
- Long training (2000 epochs) -> Small models need more iterations <br/>

## Phases
- Phase 1: (now) Grayscale scanned sketches. <br/>
- Phase 2: (soon) Digital clean drawings with color -> Upgrade NCHANNELS = 3 <br/>
- Phase 3: Explore conditioning -> Generate flowers in a certain style (blue flower, red, etc, per example) <br> 

## Generate New Flowers
```
# Generate a 4×4 grid
python src/generate.py --n 16

# Generate an interpolation between two flowers (for demos)
python src/generate.py --interpolate --steps 12

# Load a specific checkpoint
python src/generate.py --checkpoint output/checkpoints/ckpt_epoch_1000.pt
```

## The Ethical Question
This project is an ethical experiment made tangible: <br/>

- Standard AI image models train on millions of images scraped from the web, usually without artist consent <br/>
- This model trains on exactly 100 drawings, all made by the same person, who chose to do this <br/>
- The "dataset" is a creative act in itself <br/>

Does consent change the ethics? Does scale? Does authorship of the training data change who "authored" the output?
