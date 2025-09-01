# LoRA: Low-Rank Adaptation for Efficient Fine-Tuning

[PDF Source](https://arxiv.org/pdf/2106.09685)

---

## Overview

LoRA is designed **only for fine-tuning large pre-trained models**, not for pre-training from scratch. Its main goal is to **reduce computation and time cost** by freezing the base model parameters and only learning a small set of low-rank adaptation parameters.

---

## 1. Motivation

Training large models from scratch is:

* Computation-intensive
* Time-consuming
* Inefficient for many practical tasks

LoRA addresses this by **avoiding full fine-tuning** and instead injecting trainable low-rank matrices into the model.

---

## 2. Method

Consider a linear layer in a model:

$$
Y = X W + b
$$

Instead of fine-tuning the entire weight \$W\$, LoRA adds **low-rank matrices** \$A\$ and \$B\$:

$$
Y' = Y + \Delta Y = X W + \alpha (X A B)
$$

where:

* \$A \in \mathbb{R}^{d\_{out} \times r}\$
* \$B \in \mathbb{R}^{r \times d\_{in}}\$
* \$r \ll \min(d\_{in}, d\_{out})\$ is the low-rank size
* \$\alpha\$ is a scaling factor

Effectively, the new output is:

$$
Y' = X (W + \Delta W), \quad \Delta W = \alpha A B
$$

This **keeps the original model frozen** and only trains a small number of additional parameters.

---

## 3. Implementation Example (PyTorch)

```python
import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, W, alpha, r):
        super().__init__()
        self.w = W  # frozen base weight
        # Initialize low-rank matrices
        self.A = nn.Parameter(torch.randn(W.shape[0], r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, W.shape[1]) * 0.01)
        self.alpha = alpha

    def forward(self, x):
        delta = self.alpha * (x @ self.A @ self.B)
        return x @ self.w + delta
```

---

## 4. Summary

* LoRA allows **efficient fine-tuning** of large pre-trained models.
* It **freezes the original weights** and introduces a **small trainable low-rank matrix**.
* Reduces **computation**, **memory**, and **time cost**.
* Only a **small number of parameters** need updating, making it practical for resource-limited environments.

---

## References

* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
