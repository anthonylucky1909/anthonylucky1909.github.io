from pathlib import Path

# Descriptive Markdown content with Mermaid diagram included
tabular_md_with_diagram = """# Understanding LLMs for Tabular Data Generation

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

---
## Overview

The paper *Language Models are Realistic Tabular Data Generators* introduces an innovative method that leverages Large Language Models (LLMs), specifically GPT-2, to generate realistic synthetic tabular data. Traditional approaches such as GANs, Variational Autoencoders, or statistical models often face limitations: they may not generalize well across heterogeneous data types (numerical, categorical, textual), they typically require task-specific modeling assumptions, and ensuring privacy-preserving data generation remains challenging. This work reframes the problem by treating tabular data generation as a language modeling task, where tables are converted into sequences of tokens and the LLM learns dependencies between attributes in a manner similar to how it learns relationships between words in sentences. The resulting approach enables a range of applications, including data augmentation to improve machine learning models, the creation of privacy-preserving synthetic datasets for sharing, the imputation of missing values in incomplete rows, and stress-testing machine learning systems with diverse synthetic data.


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
   P(t_1, t_2, ..., t_n) = \\prod_{i=1}^n P(t_i | t_1, t_2, ..., t_{i-1})
   $$

4. **Sampling with Temperature**  
   During generation, tokens are sampled from the learned probability distribution.  
   Temperature \\( \\tau \\) adjusts randomness:  

   $$
   P'(t) = \\frac{\\exp(\\log P(t)/\\tau)}{\\sum_j \\exp(\\log P(t_j)/\\tau)}
   $$

   - Low \\(\\tau\\): conservative, deterministic outputs.  
   - High \\(\\tau\\): diverse, more exploratory outputs.  

5. **Deserialization**  
   Generated text is parsed back into a tabular format, restoring numerical and categorical attributes.  

---

## Capabilities

- **Mixed-Type Data**: Works on both numerical and categorical attributes.  
- **Partial Row Completion**: Can infer missing values by conditioning on observed attributes.  
- **Flexible Control**: Temperature allows trade-off between diversity and accuracy.  
- **Data Privacy**: Produces synthetic data with similar statistical properties, reducing risks of exposing raw sensitive data.  

## Evaluation Strategy

To validate the realism and utility of the synthetic data, the authors split datasets into 80% training and 20% testing. Synthetic data is then generated using the LLM, and machine learning models are trained on both the real and synthetic datasets. The models are subsequently evaluated on the same test set to provide a fair comparison. The experiments involve Linear Regression, Logistic Regression, and Decision Trees as benchmark models. Results demonstrate that models trained on synthetic data achieve performance comparable to those trained on real data, indicating that the generated tables successfully preserve the statistical dependencies and correlations present in the original datasets.


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
