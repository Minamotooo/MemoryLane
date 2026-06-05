# MemoryLane.
### A one-click solution to image colorization

CSE330 — Machine Learning Sessional | Jan 2026

**Team:** Aurchi Chowdhury · Sakif Naieb Raiyan · Sayaad Muzahid Masfi

---

## Overview

MemoryLane colorizes black-and-white photos using deep learning. It works in the **CIELab color space** — the grayscale L\* (Luminance) channel goes in, and the model predicts the a\* and b\* (chrominance) channels to reconstruct a full-color image.

The core challenge is that this is an "ill-posed" problem: one grayscale value can map to many valid colors (a gray blob could be a red or green apple). Prior work tends to predict the statistical mean of all plausible colors, producing **washed-out, sepia-toned outputs**. We address this directly.

---

## Our Contributions

We build on Lin & Ferreira (2021), which used an EfficientNet-B7 encoder + U-Net decoder trained on the Places dataset with MSE loss. We make three targeted improvements:

### 1. Architecture Upgrade — ConvNeXt-Base Encoder
We replaced the outdated EfficientNet-B7 backbone with **ConvNeXt-Base**, a state-of-the-art feature extractor. Our hypothesis: richer semantic features lead to more accurate colorization of complex objects like clothing and vehicles.

### 2. Dataset Bias Fix — Colorfulness Filtering
Training on the raw Places365 dataset (~1.8M images) includes many desaturated and near-grayscale images, which biases the model toward faded predictions. We filtered the dataset by **chroma score** (threshold = 18) and finetuned on the top 28.7% most colorful images (~518K), forcing the model to learn vivid color distributions.

### 3. Multi-Component Loss — Solving the Desaturation Problem
We replaced the plain MSE objective with a weighted combination:

```
L_total = λ_mse · L_mse  +  λ_perc · L_perc  +  λ_adv · L_adv
          (1.0)              (0.1)                (0.01)
```

| Loss | Role |
|------|------|
| **Rebalanced MSE** | Upweights rare colors, prevents gray bias |
| **VGG Perceptual (relu2_2, relu3_3)** | Penalizes texture/structure mismatch, prevents blurring |
| **PatchGAN Adversarial** | 70×70 patch critic enforces local realism, prevents artifacts |

The MSE is rebalanced by computing an empirical color distribution over the (a, b) space and inversely weighting rare colors — directly countering the model's tendency to predict the statistical mean.

---

## Architecture

**Generator:** ConvNeXt-Base encoder → U-Net decoder (skip connections at each scale)  
**Discriminator:** PatchGAN (30×30 output, each cell = 70×70 receptive field)

![Model Architecture](images/architecture.png)

**Prediction Pipeline:**

![Prediction Pipeline](images/prediction-pipeline.png)

---

## Training Setup

| | Training | Finetuning |
|-|----------|------------|
| **Dataset** | Places365 (~1.8M images) | Top 28.7% by colorfulness (~518K) |
| **GPU** | H100 (80 GB) | H100 (80 GB) |
| **Epochs** | 11 | 5 (early stopped) |
| **Batch size** | 32 | — |
| **Generator LR** | 1e-4 | 2e-5 |
| **Discriminator LR** | 2e-4 | 4e-5 |
| **Optimizer** | AdamW (β1 = 0.5) | AdamW |
| **Total time** | ~35 hrs | — |

---

## Results

Mean colorfulness score (Hasler & Süsstrunk, 2003) averaged across the test set:

| Model | Score |
|-------|-------|
| Ground Truth (Original) | 36.9 |
| ConvNeXt Pretrained | 22.7 |
| EfficientNet B7 | 31.9 |
| **ConvNeXt Finetuned** | **40.9** |

![Colorfulness Comparison](images/colorfulness-comparison.png)

The finetuned model **surpasses the ground truth** colorfulness score, confirming that targeted finetuning on a colorfulness-biased subset is an effective and lightweight strategy for correcting desaturation bias in large-dataset-trained colorization models.

**Test set (Places365):**

![Test set visual comparison](images/results-test-set.png)

**Outside domain (historical photographs):**

![Outside domain visual comparison](images/results-outside-domain.png)

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `ml-project.ipynb` | Main training pipeline (ConvNeXt pretrained) |
| `bright-image-ml-project.ipynb` | Finetuning on colorfulness-filtered subset |
| `autocolorization-base-paper.ipynb` | EfficientNet-B7 baseline (Lin & Ferreira replication) |
| `final-inference-all-3-models.ipynb` | Side-by-side comparison of all 3 models |
| `inference-demo.ipynb` | Quick inference demo |

---

## Motivation

Just before we were set to submit our project ideas, a team member's parent asked if we could colorize a photo from his first year of university. Although he ended up using a commercial tool, the question sparked the idea. We wanted to build our own colorizer — one capable of breathing new life into old family photos, historical archives, and black-and-white artwork.
