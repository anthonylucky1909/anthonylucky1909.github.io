
# Understanding CLIP: Contrastive Language–Image Pretraining

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)

---

## Overview

![CLIP](https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png)

CLIP (Contrastive Language–Image Pretraining) is a model developed by OpenAI to learn **visual representations from natural language supervision**. Unlike traditional supervised models trained on fixed labels, CLIP leverages **image–text pairs** available on the web to train a **dual-encoder model** that can measure **image–text similarity**. This allows **zero-shot classification**, retrieval, and other tasks without task-specific finetuning.

The core idea is simple:

1. Encode images and text into a **shared embedding space**.
2. Use **cosine similarity** to measure how well an image and text match.
3. Train the model with a **contrastive loss** to align matching pairs while separating non-matching ones.

---

## CLIP Architecture

CLIP uses **two separate encoders**:

- **Image encoder**: $f_\\theta(I)$, maps an image $I$ to embedding $\\mathbf{z}_I \\in \\mathbb{R}^d$
- **Text encoder**: $g_\\phi(T)$, maps text $T$ to embedding $\\mathbf{z}_T \\in \\mathbb{R}^d$

Both embeddings are **L2-normalized**:

$$
\\mathbf{z}_I = \\frac{f_\\theta(I)}{\\|f_\\theta(I)\\|}, \\quad
\\mathbf{z}_T = \\frac{g_\\phi(T)}{\\|g_\\phi(T)\\|}
$$

The similarity between image and text is computed via **cosine similarity**:

$$
\\text{sim}(\\mathbf{z}_I, \\mathbf{z}_T) = \\mathbf{z}_I^\\top \\mathbf{z}_T
$$

---

## Contrastive Training

Given a batch of $N$ image–text pairs $(I_i, T_i)$, the **similarity matrix** is:

$$
S_{ij} = \\frac{\\mathbf{z}_{I_i}^\\top \\mathbf{z}_{T_j}}{\\tau}
$$

where $\\tau > 0$ is a **learnable temperature parameter** controlling distribution sharpness.

The **symmetric contrastive loss** is defined as:

$$
\\mathcal{L} = \\frac{1}{2} \\left(
-\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(S_{ii})}{\\sum_{j=1}^{N} \\exp(S_{ij})} 
- \\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(S_{ii})}{\\sum_{j=1}^{N} \\exp(S_{ji})}
\\right)
$$

This encourages **matching image–text pairs** to be close while pushing apart non-matching pairs.

---

## Zero-Shot Classification

For a classification task with $K$ classes $\\{c_1, ..., c_K\\}$:

1. Convert each class into a **text prompt** $T_k$, e.g., "a photo of a cat."
2. Encode each prompt into an embedding $\\mathbf{z}_{T_k}$.
3. Classify an image $I$ using:

$$
\\hat{y} = \\arg\\max_k \\text{sim}(\\mathbf{z}_I, \\mathbf{z}_{T_k})
$$

This allows **direct application to unseen tasks** without finetuning.

---

## Implementation Example in PyTorch

```python
import torch
import torch.nn as nn

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super(CLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, images, texts):
        z_i = self.image_encoder(images)
        z_i = z_i / z_i.norm(dim=-1, keepdim=True)
        z_t = self.text_encoder(texts)
        z_t = z_t / z_t.norm(dim=-1, keepdim=True)
        return z_i @ z_t.T  # Cosine similarity matrix
