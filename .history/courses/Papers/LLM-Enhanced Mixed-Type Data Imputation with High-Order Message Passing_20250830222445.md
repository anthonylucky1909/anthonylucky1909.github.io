# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## 1. Motivation

Conventional vision systems require supervised training on large labeled datasets (e.g., ImageNet). However, these datasets are expensive to construct, limited in diversity, and restricted to predefined classes. Meanwhile, natural language offers a virtually unlimited source of supervision because images are often paired with text on the internet.  

CLIP (Contrastive Language–Image Pretraining) leverages this by learning **joint embeddings** of images and text, enabling zero-shot transfer: given any natural language description, the model can retrieve or classify images without task-specific finetuning.  

---

## 2. Model Architecture

CLIP employs a **dual-encoder** design:

- **Image Encoder**: \( f_\theta(I) \), a ResNet or Vision Transformer mapping an image \( I \) into an embedding space \( \mathbb{R}^d \).
- **Text Encoder**: \( g_\phi(T) \), a Transformer language model mapping a text sequence \( T \) into the same embedding space \( \mathbb{R}^d \).

The encoders output:

\[
z_I = \frac{f_\theta(I)}{\| f_\theta(I) \|}, \quad
z_T = \frac{g_\phi(T)}{\| g_\phi(T) \|}
\]

where embeddings are L2-normalized to lie on the unit hypersphere.

---

## 3. Similarity Function

Similarity between an image and text is measured by the **cosine similarity**:

\[
s(z_I, z_T) = z_I^\top z_T \quad \in [-1, 1]
\]

This score quantifies semantic alignment between an image and a caption.

---

## 4. Contrastive Objective

Given a minibatch of \( N \) image–text pairs \( \{(I_i, T_i)\}_{i=1}^N \), CLIP computes the similarity matrix:

\[
S_{ij} = s(z_{I_i}, z_{T_j}) / \tau
\]

where \( \tau > 0 \) is a **learnable temperature parameter** controlling distribution sharpness.

### 4.1 Image-to-Text Loss
For each image, the correct text is treated as the positive pair, while all other texts in the batch are negatives:

\[
\mathcal{L}_{\text{img}} = -\frac{1}{N} \sum_{i=1}^N 
\log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})}
\]

### 4.2 Text-to-Image Loss
Symmetrically, for each text, the correct image is the positive, and all others are negatives:

\[
\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^N 
\log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ji})}
\]

### 4.3 Final Loss
The final training objective is the average of both directions:

\[
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{img}} + \mathcal{L}_{\text{text}} \right)
\]

This is equivalent to a **symmetric cross-entropy loss** over the similarity matrix.

---

## 5. Zero-Shot Inference

For classification with class labels \( \{c_1, \dots, c_K\} \), textual prompts are created (e.g., *“a photo of a {label}”*).  

Each prompt \( T_k \) is embedded into \( z_{T_k} \).  

An image \( I \) is classified by:

\[
\hat{y} = \arg\max_{k} \, s(z_I, z_{T_k})
\]

Thus, CLIP avoids task-specific finetuning and can generalize across unseen datasets.

---

## 6. Properties

- **Scalability**: Trained on 400M image–text pairs.
- **Generalization**: Competitive zero-shot accuracy across diverse benchmarks.
- **Flexibility**: Supports retrieval, classification, and similarity scoring for arbitrary image–text pairs.
- **Simplicity**: The loss is a straightforward contrastive cross-entropy with cosine similarity.

---

## 7. Key Equations

1. **Embeddings**  
\[
z_I = \frac{f_\theta(I)}{\| f_\theta(I) \|}, \quad 
z_T = \frac{g_\phi(T)}{\| g_\phi(T) \|}
\]

2. **Cosine Similarity**  
\[
s(z_I, z_T) = z_I^\top z_T
\]

3. **Symmetric Contrastive Loss**  
\[
\mathcal{L} = \frac{1}{2} \left( 
-\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})} \;+\;
-\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ji})}
\right)
\]

---

## 8. Reference

Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*.  
[arXiv:2103.00020](https://arxiv.org/pdf/2103.00020)
