# Understanding Logistic Regression in Machine Learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg)](https://pytorch.org/) [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)

---

## Overview

Logistic Regression is a fundamental statistical and machine learning model used for **binary and multiclass classification tasks**. Unlike Linear Regression, which predicts continuous outcomes, Logistic Regression predicts the probability of categorical outcomes using the **logistic (sigmoid) function**.
It is widely used in areas such as medicine, social sciences, and natural language processing due to its **simplicity, interpretability, and probabilistic output**.

---

## Mathematical Formulation

The logistic function maps any real-valued number into the interval \[0,1]:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

where

$z = w^T x + b$

* `w` is the weight vector
* `x` is the input feature vector
* `b` is the bias term

The decision rule for binary classification:

$\hat{y} = \begin{cases} 1 & \text{if } \sigma(w^T x + b) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$

The **odds** of an event y=1 given x are:

$\text{odds} = \frac{P(y=1|x)}{P(y=0|x)}$

Taking the natural logarithm gives the **logit function**:

$\text{logit}(P) = \log\frac{P(y=1|x)}{1-P(y=1|x)} = w^T x + b$

Thus, Logistic Regression is essentially a **linear model on the log-odds**.

---

## Model Evaluation Metrics

To evaluate performance, Logistic Regression uses metrics derived from the **confusion matrix**:

* **Accuracy**: Fraction of correctly predicted instances.
* **Precision**: Reliability of positive predictions (`TP / (TP + FP)`).
* **Recall (Sensitivity)**: Fraction of actual positives correctly predicted (`TP / (TP + FN)`).
* **F1 Score**: Harmonic mean of Precision and Recall.

These metrics provide a comprehensive view of model performance, especially on **imbalanced datasets**.

---

## Implementation in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model, loss, optimizer
model = LogisticRegression(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    y_true = y_test.numpy()
    y_pred_class = y_pred_class.numpy()

    acc = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

---

## Visualization

![Logistic Regression GIF](https://facultystaff.richmond.edu/~tmattson/INFO303/images/logisticregressionanimatedgif.gif)

---

## References

* [Understanding Logistic Regression](https://medium.com/@anthonyhuang1909/understanding-logistic-regression-in-machine-learning)
* [PyTorch Documentation](https://pytorch.org/)
* [Scikit-learn Documentation](https://scikit-learn.org/)

---

## License

This project is licensed under the Apache 2.0 License.
