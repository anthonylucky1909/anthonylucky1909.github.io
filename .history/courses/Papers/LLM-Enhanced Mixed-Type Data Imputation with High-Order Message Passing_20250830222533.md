# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## 1. Motivation

Traditional vision models are trained on labeled datasets (e.g., ImageNet), which are costly to create and limited in scope. Meanwhile, there is an abundance of natural image–text pairs available online. CLIP leverages this by learning a **shared embedding space** for images and text, enabling zero-shot transfer to new tasks.

---

## 2. Model Architecture

CLIP uses **two encoders**:

- Image encoder: f(I) → vector z_I in R^d  
- Text encoder: g(T) → vector z_T in R^d  

Both outputs are **normalized** so that the vector length is 1:

- z_I = f(I) divided by its magnitude (||f(I)||)  
- z_T = g(T) divided by its magnitude (||g(T)||)

All embeddings therefore lie on the **unit sphere**.

---

## 3. Similarity Function

Similarity between an image and text is computed with **cosine similarity**:

- sim(z_I, z_T) = dot product of z_I and z_T  
  (since both are normalized, this is equivalent to cos(theta))

This gives a value between -1 and 1 indicating semantic alignment.

---

## 4. Contrastive Training

We train on a batch of N image–text pairs.  
Define the similarity between each image i and text j as:

- S[i][j] = dot product of z_I[i] and z_T[j] divided by temperature τ  

where τ > 0 is a learnable parameter controlling sharpness.

### Image-to-Text Loss

For each image, the correct text is treated as positive, all other texts in the batch are negatives:

- L_img = -(1/N) * sum over i of log( exp(S[i][i]) / sum over j of exp(S[i][j]) )

### Text-to-Image Loss

Symmetric for texts:

- L_text = -(1/N) * sum over i of log( exp(S[i][i]) / sum over j of exp(S[j][i]) )

### Final Loss

The total loss is the average of both directions:

- L = 0.5 * (L_img + L_text)

---

## 5. Zero-Shot Classification

For K classes {c1, c2, ..., cK}, we create textual prompts (e.g., "a photo of a dog"). Each prompt T_k is encoded into z_Tk.  

Given an image I, we classify it as:

- predicted_class = argmax over k of sim(z_I, z_Tk)

This allows classification **without any task-specific finetuning**.

---

## 6. Key Properties

- Trained on 400 million image–text pairs.  
- Generalizes well to unseen datasets.  
- Supports image-text retrieval, zero-shot classification, and similarity scoring.  
- Simple contrastive loss with cosine similarity as the only similarity measure.

---

## 7. Key Equations (Readable Version)

1. Embeddings:

- Image: z_I = f(I) / magnitude of f(I)  
- Text: z_T = g(T) / magnitude of g(T)

2. Cosine similarity:

- sim(z_I, z_T) = dot product of z_I and z_T

3. Symmetric contrastive loss:

- L = 0.5 * (L_img + L_text)  

Where:

- L_img = -(1/N) * sum over i of log( exp(S[i][i]) / sum over j of exp(S[i][j]) )  
- L_text = -(1/N) * sum over i of log( exp(S[i][i]) / sum over j of exp(S[j][i]) )

---

## Reference

Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*.  
[arXiv:2103.00020](https://arxiv.org/pdf/2103.00020)
