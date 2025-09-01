# Understanding LLMs for Tabular Data Generation

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

---

## Overview

The paper *Language Models are Realistic Tabular Data Generators* introduces an innovative method to use **Large Language Models (LLMs)**, specifically **GPT-2**, for generating **realistic synthetic tabular data**.  

Traditional approaches for tabular data generation or imputation (e.g., GANs, Variational Autoencoders, statistical models) often face limitations:
- They may not generalize well across **heterogeneous data types** (numerical, categorical, textual).  
- They often require **task-specific modeling assumptions**.  
- Privacy-preserving data generation remains challenging.  

This work reframes the problem: instead of designing task-specific models, it leverages the **general-purpose language modeling ability** of LLMs. By converting **tables into sequences of tokens**, LLMs can learn dependencies between attributes in much the same way they learn dependencies between words in sentences.  

Applications include:
- **Data augmentation** to improve machine learning models.  
- **Privacy-preserving synthetic datasets** for data sharing.  
- **Imputation of missing values** in incomplete rows.  
- **Stress-testing ML systems** with diverse synthetic data.  

---

## Core Workflow

The pipeline is designed to translate **tabular data → text → tokens → LLM → synthetic tokens → text → tabular data**.

1. **Serialization**  
   Each table row is converted into a structured text string (e.g., `"age: 34, gender: M, income: 55000"`).  
   This ensures that both numerical and categorical attributes are represented in a format the LLM can process.  

2. **Tokenization**  
   The serialized string is tokenized using a subword tokenizer (e.g., BPE from GPT-2).  
   This breaks down values and attribute names into a consistent vocabulary of tokens.  

3. **Model Training**  
   GPT-2 is fine-tuned in an **auto-regressive** fashion: it learns to predict the next token given all previous tokens.  
   Formally, it maximizes the likelihood:  

   $$
   P(t_1, t_2, ..., t_n) = \prod_{i=1}^n P(t_i | t_1, t_2, ..., t_{i-1})
   $$

4. **Sampling with Temperature**  
   During generation, tokens are sampled from the learned probability distribution.  
   Temperature \( \tau \) adjusts randomness:  

   $$
   P'(t) = \frac{\exp(\log P(t)/\tau)}{\sum_j \exp(\log P(t_j)/\tau)}
   $$

   - Low \(\tau\): conservative, deterministic outputs.  
   - High \(\tau\): diverse, more exploratory outputs.  

5. **Deserialization**  
   Generated text is parsed back into a tabular format, restoring numerical and categorical attributes.  

---

## Capabilities

- **Mixed-Type Data**: Works on both numerical and categorical attributes.  
- **Partial Row Completion**: Can infer missing values by conditioning on observed attributes.  
- **Flexible Control**: Temperature allows trade-off between diversity and accuracy.  
- **Data Privacy**: Produces synthetic data with similar statistical properties, reducing risks of exposing raw sensitive data.  

---

## Evaluation Strategy

To validate realism and utility of synthetic data:

1. Datasets are split into **80% training and 20% testing**.  
2. Synthetic data is generated from the LLM.  
3. ML models are trained on **synthetic vs. real data**, then evaluated on the same test set.  
4. Compared models include:  
   - **Linear Regression**  
   - **Logistic Regression**  
   - **Decision Trees**  

Results demonstrate that models trained on synthetic data achieve **similar predictive performance** to those trained on real data, proving that **statistical dependencies and correlations** are preserved.  

---

## Implementation Example (Pseudo PyTorch Code)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example row serialization
row = {"age": 34, "gender": "M", "income": 55000}
text = "age: 34, gender: M, income: 55000"

# Tokenization
inputs = tokenizer(text, return_tensors="pt")

# Forward pass (auto-regression loss)
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

# Generation with temperature sampling
gen_tokens = model.generate(inputs["input_ids"],
                            max_length=50,
                            temperature=0.8,
                            do_sample=True)
gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```


---

## References

* Borisov, V. et al. (2022). *Language Models are Realistic Tabular Data Generators*. [arXiv:2210.06280](https://arxiv.org/abs/2210.06280)  
* HuggingFace Transformers Documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)  

---

## License

This project is licensed under the MIT License.
