# HaMeR: Hand Mesh Recovery
Code repository for the paper:
**Reconstructing Hands in 3D with Transformers**
![teaser](assets/teaser.jpg)

## Installation
First you need to clone the repo:
```
git clone --recursive https://github.com/Shade5/hamer.git
cd hamer
```

We recommend using [uv](https://docs.astral.sh/uv/) for installation:
```bash
uv sync
source .venv/bin/activate
uv pip install --force-reinstall --no-binary xtcocotools xtcocotools --no-build-isolation
uv pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation --no-deps
uv pip install -e third-party/ViTPose --no-build-isolation
rsync -zaP /mnt/sunny_nas/weights/hamer_weights/ _DATA/
```
If the mano files are missing, need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## Demo
```bash
python demo_simple.py
```
