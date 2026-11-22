# Deepfake Detection

Binary classifier for **real vs AI-generated images** (`0 = fake`, `1 = real`).

The project compares EfficientNet-B0 and EfficientNetV2-S on two very different datasets, then trains a combined EfficientNet-B0 that generalises across both. There is no web app or API: training and evaluation live in notebooks and a training script; inference is a small Python helper.

## Results

Cross-dataset generalisation is the main finding. A model trained on one domain looks strong in-domain and fails on the other. A larger backbone (EfficientNetV2-S) does **not** close that gap. Training on both domains does.

| Weights | Train data | CIFAKE AUC | Faces AUC |
|---|---|---|---|
| `model_cifake.pth` (B0) | CIFAKE | 0.983 | 0.460 |
| `model_faces.pth` (B0) | Faces | 0.493 | 0.880 |
| `model_cifake_v2.pth` | CIFAKE | 0.984 | 0.445 |
| `model_faces_v2.pth` | Faces | 0.452 | 0.915 |
| **`model_combined.pth` (B0)** | CIFAKE + Faces | **0.992** | **0.972** |

Same-domain B0 vs V2 numbers come from `data/processed/results_b0_vs_v2.csv`. Combined-model numbers come from `data/processed/evaluation_results.csv`.

Caveats: most comparison runs used a single seed, 128×128 inputs (CIFAKE is upscaled from 32×32), and short training. The B0 baselines were originally trained on Kaggle and reused locally unless retrained.

## Datasets

Raw images are **not** stored in this repository (too large). Download them into `data/raw/` with the folder names below.

| Dataset | Content | Resolution | Splits in this project | Local path |
|---|---|---|---|---|
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | CIFAR-10 style objects vs AI images | 32×32 | 50k/50k train, 10k/10k test per class | `data/raw/images ai vs real/{train,test}/{FAKE,REAL}` |
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Flickr faces vs StyleGAN | 256×256 | 50k/50k train, 10k/10k val, 10k/10k test per class | `data/raw/Facesai vs real/real_vs_fake/real-vs-fake/{train,valid,test}/{fake,real}` |

Both datasets are balanced RGB JPEGs. `ImageFolder` maps fake → 0 and real → 1 (alphabetical class names).

## Layout

```
deepfake_detection/
├── data/processed/          # CSV summaries and metrics (committed)
├── data/raw/                # datasets (gitignored; download locally)
├── models/                  # trained .pth weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_efficientnetv2_vs_b0.ipynb
├── scripts/train_combined.py
└── deepfake_detection/      # installable package
    └── deepfake_detection/
        ├── config.py        # paths and seed
        └── utils.py         # load_model / predict
```

## Setup

Tested with Python 3.10, PyTorch 2.14 (CUDA 12.6), and an NVIDIA GPU. CPU inference works; training EfficientNetV2 on CPU is not practical.

```bash
cd deepfake_detection
pip install -e ./deepfake_detection
pip install torch torchvision timm pillow pandas matplotlib scikit-learn jupyter
```

Install a CUDA build of PyTorch that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/).

Point a Jupyter kernel at this environment (the comparison notebook expects a kernel named `deepfake`).

## Training

Combined EfficientNet-B0 (20k images from each dataset, JPEG compression augmentation, 5 epochs, mixed precision on GPU):

```bash
python scripts/train_combined.py
```

Weights are written to `models/model_combined.pth` after every epoch.

B0 vs EfficientNetV2-S (same subsets, 128×128, AdamW, ImageNet normalisation) is in `notebooks/02_efficientnetv2_vs_b0.ipynb`. Set `REUSE_B0 = False` in that notebook for a strictly identical local retrain of B0.

Exploratory analysis (balance, file size, colour, frequency content, possible shortcuts) is in `notebooks/01_data_exploration.ipynb`.

## Inference

```python
from deepfake_detection.utils import load_model, predict

model = load_model("model_combined.pth")
print(predict(model, "path/to/image.jpg"))
# {'fake': 0.02, 'real': 0.98}
```

Images are resized to 128×128 and normalised with ImageNet mean/std.

## License and data

Code in this repository is provided as-is for research. The image datasets remain under their original Kaggle/source licenses; download them separately.
