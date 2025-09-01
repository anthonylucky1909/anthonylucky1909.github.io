# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## Motivation

Traditional vision models are trained with manually annotated datasets such as ImageNet, which are limited in scale and diversity. However, natural language supervision is abundant: billions of image–text pairs exist on the internet. The motivation of CLIP (Contrastive Language–Image Pretraining) is to **align images and text into a shared embedding space**, enabling zero-shot transfer to downstream vision tasks without task-specific training.

---

## Architecture

CLIP employs a **dual-encoder architecture**:

- **Image Encoder**: Typically a ResNet or Vision Transformer (ViT) maps an image \( I \) to a vector representation \( f(I) \in \mathbb{R}^d \).
- **Text Encoder**: A Transformer-based model maps a text description \( T \) into a vector \( g(T) \in \mathbb{R}^d \).

Both encoders project into the same **joint embedding space** of dimension \( d \).

Formally:
\[
z_I = f(I) \in \mathbb{R}^d, \quad z_T = g(T) \in \mathbb{R}^d
\]

---

## Training Objective

The objective is to **maximize similarity** between the embedding of a paired image and its corresponding text while minimizing similarity with mismatched pairs.  

Similarity is measured using the **cosine similarity**:
\[
\text{sim}(z_I, z_T) = \frac{z_I \cdot z_T}{\|z_I\|\|z_T\|}
\]

Given a batch of \( N \) image–text pairs, CLIP computes a similarity matrix:
\[
S_{ij} = \text{sim}(z_{I_i}, z_{T_j})
\]

A **symmetric cross-entropy loss** is applied:

\[
\mathcal{L}_{\text{img}} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii} / \tau)}{\sum_{j=1}^N \exp(S_{ij} / \tau)}
\]

\[
\mathcal{L}_{\text{text}} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii} / \tau)}{\sum_{j=1}^N \exp(S_{ji} / \tau)}
\]

The final training loss is:
\[
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{img}} + \mathcal{L}_{\text{text}} \right)
\]

Here, \( \tau \) is a **learnable temperature parameter** controlling the sharpness of the distribution.

---

## Zero-Shot Classification

CLIP can perform classification **without finetuning** by reformulating labels as **text prompts**.

For example, for ImageNet classification:

- Class label: *“dog”*  
- Prompt: *“a photo of a dog”*  

The text encoder embeds all class prompts, then classification is done by computing cosine similarity between the image embedding and each class embedding:

\[
\hat{y} = \arg\max_{c} \, \text{sim}(z_I, g(T_c))
\]

This allows CLIP to generalize to tasks it was never explicitly trained on.

---

## Advantages

- **Scalability**: Trained on 400M image–text pairs.
- **Generalization**: Strong zero-shot performance on multiple benchmarks.
- **Flexibility**: Works with arbitrary image–text similarity queries.
- **Simple Objective**: Contrastive learning with cosine similarity.

---

## Summary

- CLIP uses **dual encoders** (image + text) projecting into a shared embedding space.  
- Training minimizes a **contrastive loss** that aligns paired embeddings and separates unpaired ones.  
- Similarity is computed with **cosine similarity**.  
- Enables **zero-shot classification** and flexible cross-modal retrieval.

---

## Key Equations

1. Embedding vectors:
   \[
   z_I = f(I), \quad z_T = g(T)
   \]

2. Cosine similarity:
   \[
   \text{sim}(z_I, z_T) = \frac{z_I \cdot z_T}{\|z_I\|\|z_T\|}
   \]

3. Contrastive loss:
   \[
   \mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{img}} + \mathcal{L}_{\text{text}} \right)
   \]

---

## Reference

Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*.  
[arXiv:2103.00020](https://arxiv.org/pdf/2103.00020)
