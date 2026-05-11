# DrawingMLmodel
What if the artist consents? <br/>
A generative AI model trained exclusively on one artist's own drawings, created to explore the ethics of AI art generation. <br/>
This tool can then be used by the artist to get inspiration to create new art when having, for example, an art-blockage. <br/>

## The Idea
AI image generation is surrounded by controversy: artists' work is scraped and used to train models without consent. This project flips that question. What does it look like when a single artist deliberately trains a model only on their own work? <br/>
The output isn't just a generator; it's a statement about authorship, consent and what "AI-generated art" can mean when the human behind it is fully in control of the dataset. <br/>
This tool trains two small generative models on your artwork:
- Model 1 learns your sketch style from your own drawings
- Model 2 learns colours from reference photographs
- A colouriser bridges both, applying colour knowledge to the generated sketches

Everything runs locally on your machine. Nothing is uploaded anywhere, your art stays yours.

## Quick Start
Before anything, you will need a dataset, the minimum dataset size that could be recommended would be around 80-100 sketches and 20-30 images. This is just a reference, if an artist is expecting a more abstarct output a smaller dataset would be perfect. <br/>

To start using this tool, there's two different posibilities.

### Using the graphic interface

On your terminal use the following commands:
```
# 1. Install dependencies
pip install -r requirements.txt
pip install flask

# 2. Launch the app
python server.py
```
Now open your prefered browser and write on your search bar the following:
```
http://localhost:5000
```
That's it! The app itself will guide you through four steps: <br/>
- Sign the Intellectual Property agreement
- Upload your sketches and reference photos (drag & drop)
- Press "Train your model" and wait (between 4 to 8 hours on CPU)
- Press "New sketch" and "Color it" to generate sketches and colour them

### Using only the terminal:

Open your terminal and follow the steps:
```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up the folder structure and clean your images
# Drop your sketches into data/raw/
# Drop your photos into data/photos_raw/

python scripts/setup_data.py # cleand sketches and saves them on data/processed/
python scripts/setup_photos.py # prepares photos and saves them on data/processed_photos/

# 3. Train Model 1 (sketch style)
python src/train.py --epochs 2000

# 4. Train Model 2 (colour knowledge)
python src/train_photos.py --epochs 2000

# 5. Train colouriser
python src/colorize.py --train --epochs 500

# 6. Generate sketches
# Single sketch
python src/generate.py --n 1

# 4x4 grid of sketches + matching colourised grid
python src/generate_grid.py

# Colourize a specific sketch by seed
python src/colorize.py --sketch_seed 42

# Interpolation between two drawings (I use this for demos)
python src/generate.py --interpolate --steps 12

```
If the training gets interrupted use the following commands:
```
python src/train.py --resume output/checkpoints/ckpt_epoch_1000.pt --epochs 2000
python src/colorize.py --train --resume output/colorizer/colorizer_epoch_300.pt
```
## How long does the training take?
| Stage | Epochs | Time |
|---|---|---|
| Model 1 (sketches) | 2,000 | ~3.5 hours |
| Model 2 (photos) | 2,000 | ~3–4 hours |
| Colouriser | 500 | ~1 hour |
| **Total** | | **~8–9 hours** |

All times are approximate on a standard consumer CPU, in this case, the experimentation was fully done with a 13th Gen Intel(R) Core(TM) i5-1335U CPU. <br/>
Training overnight is the easiest approach. Generation takes under one second once trained.

## How it works
The model is a DCGAN (Deep Convolutional Generative Adversarial Network) adapted for small datasets: <br/>

- Generator: Takes random noise -> outputs a 64×64 sketch<br/>
- Discriminator: Learns to tell real sketches from generated ones<br/>
- They compete until the Generator gets good enough to fool the Discriminator<br/>

### Small Dataset Adaptations
- Heavy augmentations (flips, rotations, crop, etc) -> To make the 100 images "feel" like 1000 <br/>
- Spectral normalisation -> Stabilises training <br/>
- Instance noise in D inputs -> Prevents discriminator from overpowering generator <br/>
- Label smoothing -> Stops the model from becoming "overconfident" <br/>
- Long training (2000 epochs) -> Small models need more iterations <br/>

## The Ethical Question
This project is an ethical experiment made tangible: <br/>

- Standard AI image models train on millions of images scraped from the web, usually without artist consent <br/>
- This model trains on exactly 100 drawings, all made by the same person, who chose to do this <br/>
- The "dataset" is a creative act in itself <br/>

Your art should not be used to train models without consent, this is why the IP agreement is shown in the interface, it is a deliberate design choice, consent is active and not asssumed.
