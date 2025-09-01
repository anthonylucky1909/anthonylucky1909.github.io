# QLoRA: Efficient Low-Rank Adaptation for Large Language Models

[PDF Source](https://arxiv.org/pdf/2305.14314)

---

## Overview

QLoRA is an **upgrade of LoRA (Low-Rank Adaptation)** for large language models. Its main goal is to **enable efficient fine-tuning** of massive models (like LLaMA or GPT) under limited GPU memory by combining:

1. **4-bit quantization** of model weights.
2. **Double quantization** to reduce memory overhead further.
3. **Paged optimizer** to store some parameters on CPU instead of GPU.

---

## 1. Motivation

Training large LLMs requires **huge GPU memory**, which limits accessibility. QLoRA optimizes memory usage without significantly hurting model performance by:

* Freezing the base model weights.
* Quantizing the weights to 4-bit.
* Using LoRA adapters in low-rank matrices to fine-tune efficiently.

---

## 2. 4-bit Quantization

The original full-precision weights \$W \in \mathbb{R}^{d\_{out} \times d\_{in}}\$ are **quantized** to 4-bit using a scale:

$$
\text{scale} = \frac{\max(|W|, \text{dim}=1)}{\text{clip}}
$$

$$
W_q = \text{round}\left(\frac{W}{\text{scale}}\right) \in [-8, 7] \subset \mathbb{Z}_8
$$

Dequantization:

$$
W \approx W_q \cdot \text{scale}
$$

This allows **memory-efficient storage** while preserving approximate precision.

---

## 3. Double Quantization

To save even more memory:

* First quantize weights to 4-bit.
* Store the **scale factors** in 8-bit integers rather than 32-bit floats.

This reduces overall memory usage and makes large model training feasible on limited GPUs.

---

## 4. LoRA Integration

LoRA adds **trainable low-rank matrices** \$A\$ and \$B\$ to the frozen base weights:

$$
W_{adapted} = W_{frozen} + \alpha \cdot B A
$$

where:

* \$A \in \mathbb{R}^{r \times d\_{in}}\$
* \$B \in \mathbb{R}^{d\_{out} \times r}\$
* \$r \ll \min(d\_{in}, d\_{out})\$ is the **rank**
* \$\alpha / r\$ is a scaling factor.

This allows **fine-tuning only a small number of parameters** while keeping the main model frozen.

---

## 5. Paged Optimizer

Training huge models is still limited by GPU memory. QLoRA uses a **paged Adam optimizer**:

* Some optimizer states are **offloaded to CPU memory**.
* Only active weights remain on GPU.
* This reduces memory usage drastically and avoids out-of-memory errors.

---

## 6. Training Pipeline

The simplified QLoRA training pipeline:

1. **Freeze the base model weights**.
2. **Quantize** frozen weights to 4-bit.
3. **Scale factors** stored in 8-bit.
4. Add **LoRA adapters** \$A, B\$.
5. Train only the LoRA parameters using a **paged optimizer**.

---

## 7. Implementation Example (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simple 4-bit quantize/dequant ---
def quantize_nf4(W, clip=7.5):
    scale = W.abs().amax(dim=1, keepdim=True) / clip
    scale[scale == 0] = 1.0
    W_q = torch.clamp((W / scale).round(), -8, 7).to(torch.int8)
    return W_q, scale

def dequant(W_q, scale):
    return W_q.float() * scale

# --- Simple QLoRA Linear Layer ---
class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super().__init__()
        # Frozen base weight (quantized)
        W = torch.randn(out_features, in_features) * 0.02
        self.W_q, self.scale = quantize_nf4(W)
        self.W_q.requires_grad = False  # frozen

        # LoRA adapters (trainable)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        # Dequantize frozen base weight
        W = dequant(self.W_q, self.scale)
        base_out = F.linear(x, W)  # frozen base
        lora_out = F.linear(F.linear(x, self.A), self.B) * self.scaling  # LoRA
        return base_out + lora_out
```

---

## 8. Summary

* QLoRA allows **memory-efficient fine-tuning** of large language models.
* Combines **4-bit quantization**, **double quantization**, and **LoRA adapters**.
* **Paged optimizer** enables training without exceeding GPU memory.
* Only **a small number of trainable parameters** (LoRA) are optimized.

This makes QLoRA **practical for large-scale model adaptation** on limited hardware.

---

## References

* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)
* LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
