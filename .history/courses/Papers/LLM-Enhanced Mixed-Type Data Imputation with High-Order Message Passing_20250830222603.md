# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## 1. Motivation

Traditional vision models rely on labeled datasets (e.g., ImageNet), which are expensive and limited in diversity. CLIP leverages the abundance of natural image–text pairs by learning a **shared embedding space** for images and text, enabling **zero-shot transfer** to new tasks.

---

## 2. Model Architecture

CLIP employs a **dual-encoder architecture**:

- **Image encoder** \( f_\theta(I) \) maps an image \( I \) to an embedding \( \mathbf{z}_I \in \mathbb{R}^d \).  
- **Text encoder** \( g_\phi(T) \) maps a text sequence \( T \) to an embedding \( \mathbf{z}_T \in \mathbb{R}^d \).

Both embeddings are **L2-normalized**:

\[
\mathbf{z}_I = \frac{f_\theta(I)}{\|f_\theta(I)\|}, \quad
\mathbf{z}_T = \frac{g_\phi(T)}{\|g_\phi(T)\|}
\]

Thus, all embeddings lie on the **unit hypersphere**.

---

## 3. Similarity Function

The similarity between an image and text is measured using **cosine similarity**:

\[
\text{sim}(\mathbf{z}_I, \mathbf{z}_T) = \mathbf{z}_I^\top \mathbf{z}_T
\]

Since embeddings are normalized, this is equivalent to the cosine of the angle between the vectors.

---

## 4. Contrastive Training

Given a batch of \( N \) image–text pairs \( \{(I_i, T_i)\}_{i=1}^N \), the similarity matrix is defined as:

\[
S_{ij} = \frac{\mathbf{z}_{I_i}^\top \mathbf{z}_{T_j}}{\tau}
\]

where \( \tau > 0 \) is a learnable **temperature parameter** controlling the sharpness of the distribution.

### 4.1 Image-to-Text Loss

\[
\mathcal{L}_{\text{img}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}
\]

### 4.2 Text-to-Image Loss

\[
\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})}
\]

### 4.3 Total Loss

\[
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{img}} + \mathcal{L}_{\text{text}} \right)
\]

---

## 5. Zero-Shot Classification

For \( K \) classes \( \{c_1, \dots, c_K\} \), we create textual prompts (e.g., "a photo of a dog"). Each prompt \( T_k \) is encoded into \( \mathbf{z}_{T_k} \).  

An image \( I \) is classified as:

\[
\hat{y} = \arg\max_{k} \text{sim}(\mathbf{z}_I, \mathbf{z}_{T_k})
\]

This enables classification **without any task-specific finetuning**.

---

## 6. Key Properties

- Trained on 400M image–text pairs.  
- Generalizes well to unseen tasks.  
- Supports image–text retrieval, zero-shot classification, and similarity scoring.  
- Uses a simple contrastive loss with cosine similarity.

---

## 7. Key Equations (Summary)

1. **Embeddings**:

\[
\mathbf{z}_I = \frac{f_\theta(I)}{\|f_\theta(I)\|}, \quad
\mathbf{z}_T = \frac{g_\phi(T)}{\|g_\phi(T)\|}
\]

2. **Cosine Similarity**:

\[
\text{sim}(\mathbf{z}_I, \mathbf{z}_T) = \mathbf{z}_I^\top \mathbf{z}_T
\]

3. **Symmetric Contrastive Loss**:

\[
\mathcal{L} = \frac{1}{2} \left(
-\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}
-
\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})}
\right)
\]

---

## Reference

Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*.  
[arXiv:2103.00020](https://arxiv.org/pdf/2103.00020)
