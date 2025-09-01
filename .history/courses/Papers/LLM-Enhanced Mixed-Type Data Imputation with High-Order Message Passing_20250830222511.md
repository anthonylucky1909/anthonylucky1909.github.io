# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## 1. Motivation

Traditional vision models are trained on labeled datasets (e.g., ImageNet), which are costly to create and limited in scope. Meanwhile, there is an abundance of natural image–text pairs available online. CLIP leverages this by learning a **shared embedding space** for images and text, enabling zero-shot transfer to new tasks.

---

## 2. Model Architecture

CLIP uses **two encoders**:

- Image encoder: f(I) → vector z_I in R^d  
- Text encoder: g(T) → vector z_T in R^d  

Both outputs are normalized:

- z_I = f(I) / || f(I) ||  
- z_T = g(T) / || g(T) ||

Thus, all embeddings lie on the unit hypersphere.

---

## 3. Similarity Function

Similarity between an image and text is computed with **cosine similarity**:

- sim(z_I, z_T) = (z_I · z_T) / (||z_I|| · ||z_T||)  

Since embeddings are normalized, this reduces to the dot product.

---

## 4. Contrastive Training

We train on a batch of N image–text pairs.  
Define the similarity matrix:

- S_ij = (z_Ii · z_Tj) / τ  

where τ is a learnable **temperature parameter**.

### Image-to-Text Loss
For each image, the correct text is positive, others are negatives:

- L_img = -(1/N) Σ_i log( exp(S_ii) / Σ_j exp(S_ij) )

### Text-to-Image Loss
Symmetric version:

- L_text = -(1/N) Σ_i log( exp(S_ii) / Σ_j exp(S_ji) )

### Final Loss
The overall loss is:

- L = 0.5 * (L_img + L_text)

---

## 5. Zero-Shot Classification

For K classes {c1, …, cK}, we create natural language prompts (e.g., “a photo of a dog”). Each prompt is encoded by the text encoder into z_Tk.  

An image I is classified by:

- ŷ = argmax_k sim(z_I, z_Tk)

This allows classification without finetuning.

---

## 6. Key Properties

- Trained on 400M image–text pairs.  
- Generalizes well to unseen tasks.  
- Enables retrieval, classification, and similarity scoring.  
- Uses a simple contrastive loss with cosine similarity.

---

## 7. Key Equations (Summary)

1. Embeddings:  
   z_I = f(I) / ||f(I)||  
   z_T = g(T) / ||g(T)||  

2. Cosine similarity:  
   sim(z_I, z_T) = z_I · z_T  

3. Contrastive loss:  
   L = 0.5 * (L_img + L_text)  

---

## Reference

Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*.  
[arXiv:2103.00020](https://arxiv.org/pdf/2103.00020)
