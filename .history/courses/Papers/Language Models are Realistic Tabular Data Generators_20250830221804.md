# Summary of Language Models are Realistic Tabular Data Generators

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

---

## Overview

The paper *Language Models are Realistic Tabular Data Generators* introduces **GReaT (Generation of Realistic Tabular Data)**, a framework that adapts auto-regressive language models such as GPT-2 for the task of generating realistic synthetic tabular data. Instead of relying on task-specific architectures like GANs, VAEs, or statistical methods, which often struggle with heterogeneous feature types or require heavy domain assumptions, this approach reframes tabular data generation as a text modeling task. Each table row is serialized into a structured textual sequence that includes both feature names and values, allowing the model to capture dependencies between attributes in the same way it models dependencies between words in sentences. This design ensures that both categorical and numerical features are handled seamlessly, contextual relationships between features are preserved, and missing values can be imputed effectively. By randomizing the order of features during training, GReaT further enables arbitrary conditioning, meaning it can generate or complete rows given any subset of observed features. The resulting synthetic tables not only preserve statistical dependencies and correlations but also offer practical benefits for data augmentation, privacy-preserving data sharing, missing value imputation, and stress-testing of machine learning pipelines.

---

## Workflow

The GReaT workflow follows a straightforward but powerful pipeline. First, each tabular row is serialized into a textual form where features and their values are explicitly represented. These serialized rows are then tokenized using subword methods such as Byte Pair Encoding (BPE), ensuring consistent representation of both words and numbers. GPT-2 is fine-tuned on this tokenized data in an auto-regressive manner, learning to predict each token conditioned on all previous tokens. During generation, new rows are sampled sequentially from the learned probability distribution. A temperature parameter is introduced to control randomness in token sampling: low temperatures encourage deterministic, conservative outputs, while higher temperatures allow more diverse but potentially noisier generations. Finally, the generated sequences are deserialized back into tabular form, restoring structured rows of categorical and numerical values.

---

## Evaluation

To validate the realism and utility of the generated data, datasets are split into 80% training and 20% testing. The LLM is trained on the training data, then used to produce synthetic rows, which are evaluated indirectly by training machine learning models on them. These models, including Linear Regression, Logistic Regression, and Decision Trees, are then tested on the same held-out test set. The results show that models trained on synthetic data achieve predictive performance that is highly comparable to models trained on real data. This confirms that the generated tables retain the statistical properties and correlations of the original datasets and can therefore serve as reliable substitutes in downstream tasks.

---

## Visualization

```mermaid
flowchart LR
    A[📊 Table Row] --> B[📝 Serialization into Text]
    B --> C[🔡 Tokenization]
    C --> D[🤖 GPT-2 Fine-Tuning]
    D --> E[🎲 Sampling with Temperature]
    E --> F[🔄 Deserialization]
    F --> G[📊 Synthetic Table]
