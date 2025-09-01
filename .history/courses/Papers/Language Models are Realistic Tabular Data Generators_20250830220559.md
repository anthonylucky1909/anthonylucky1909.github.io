# Understanding LLMs for Tabular Data Generation

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

---

## Overview

This paper demonstrates that **Large Language Models (LLMs)**, such as **GPT-2**, can generate **realistic synthetic tabular data**.  

The key idea is to **serialize tabular rows into text sequences** and train the model in an **auto-regressive manner**. The model learns token probabilities conditioned on previous tokens, enabling it to generate new rows sequentially. After generation, the text is **converted back to tabular format**.

Applications include:
- **Data augmentation** for ML tasks  
- **Privacy-preserving data sharing**  
- **Testing and benchmarking ML pipelines**  

---

## Core Workflow

1. **Serialization**: Convert each table row into a text sequence.  
2. **Tokenization**: Apply subword tokenization (e.g., BPE).  
3. **Model Training**: Train GPT-2 to predict the next token given previous ones.  
4. **Sampling with Temperature**: Control randomness/diversity in generation.  
   - Low temperature → conservative, more deterministic samples.  
   - High temperature → more diverse but potentially noisier samples.  
5. **Deserialization**: Convert generated text back into table format.  

---

## Example Pipeline

**Table → Text → Tokens → GPT-2 → Tokens → Text → Table**

Mathematically:

$$
\text{Table Row} \xrightarrow{\text{encoder}} \text{Text Sequence} \xrightarrow{\text{tokenizer}} \{t_1, t_2, ..., t_n\}
$$

The model learns:

$$
P(t_i | t_1, t_2, ..., t_{i-1})
$$

Generation with temperature scaling:

$$
P'(t) = \frac{\exp(\log P(t) / \tau)}{\sum_j \exp(\log P(t_j) / \tau)}
$$

where $\tau$ = temperature.

---

## Capabilities

- **Heterogeneous tables**: Handles categorical + numerical features.  
- **Row completion**: Can impute missing values in partially observed rows.  
- **Flexible sampling**: Adjust diversity via temperature scaling.  

---

## Evaluation Strategy

Datasets split into **80% train / 20% test**.  
Evaluation compares models trained on **synthetic vs. real data**:

- **Linear Regression**  
- **Logistic Regression**  
- **Decision Trees**  

Results show that synthetic data preserves **statistical properties and correlations**, yielding comparable performance to real data.

---

## Implementation Example (Pseudo PyTorch Code)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Serialize table row into text
row = {"age": 34, "gender": "M", "income": 55000}
text = "age: 34, gender: M, income: 55000"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Train with auto-regression
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

# Generate new row with temperature sampling
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
