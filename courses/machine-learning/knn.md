# Understanding K-Nearest Neighbors (KNN)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg)](https://pytorch.org/)

---

## Introduction

K-Nearest Neighbors (KNN) is a supervised learning algorithm commonly used for both classification, where it predicts the class of a data point based on the majority class of its nearest neighbors, and regression, where it predicts a continuous value by averaging the values of its nearest neighbors. KNN is considered a lazy learner because it does not construct an explicit model during training, but rather stores the training data and performs computations only at prediction time.

---

## Algorithm Steps

The KNN algorithm follows these steps: first, choose the value of k, which determines the number of neighbors to consider. Next, compute the distance between the test point and all training points. Then, identify the k nearest neighbors, i.e., the points with the smallest distances. Finally, make a prediction: for classification, select the class with the majority vote among the neighbors; for regression, compute the average of the neighbors’ values.

---

## Distance Metrics

The most common distance metric is the **Euclidean distance**:

$$
d(x, y) = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}
$$

Other distance functions include: Manhattan Distance, Minkowski Distance,Cosine Similarity. The choice of distance metric can impact model performance.

---

## Choosing k

Selecting the right k is critical: one approach is cross-validation, where multiple k values are tested and the one yielding the best performance is chosen; another is the elbow method, which involves plotting error versus k and selecting the point where the error curve begins to level off; additionally, using an odd k for classification helps avoid ties in majority voting.

---

## Visualization

![KNN GIF](https://miro.medium.com/v2/resize\:fit:1100/format\:webp/1*oOWUXrzZjhg-QXFHUf4uzg.gif)

*The GIF illustrates classification based on nearest neighbors. Different colors indicate classes, and the background shows decision boundaries.*

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import KFold

class KNN(nn.Module):
    def __init__(self, k: int):
        super(KNN, self).__init__()
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y

    def forward(self, x_test):
        # Compute Euclidean distance
        distance = torch.sqrt(torch.sum((self.X_train - x_test) ** 2, dim=1))
        k_eff = min(self.k, len(self.X_train))
        knn_index = torch.topk(distance, k_eff, largest=False).indices
        knn_labels = self.Y_train[knn_index]

        # Return the most common label
        most_common = Counter(knn_labels.tolist()).most_common(1)
        return most_common[0][0]

    def predict_batch(self, X_test):
        return torch.tensor([self.forward(x) for x in X_test])

# Cross-validation
def cross_validation(X, y, k_values, num_folds=5):
    results = {}
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for k in k_values:
        acc_scores = []
        model = KNN(k=k)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict_batch(X_val)
            acc = (y_pred == y_val).float().mean().item()
            acc_scores.append(acc)

        results[k] = sum(acc_scores) / len(acc_scores)
    return results

# Toy example
X_toy = torch.tensor([[1., 2.], [2., 3.], [3., 4.],
                      [6., 7.], [7., 8.], [8., 8.], [9., 9.]])
y_toy = torch.tensor([0, 0, 0, 1, 1, 1, 1])

k_values = list(range(1, 8))
results_toy = cross_validation(X_toy, y_toy, k_values, num_folds=3)
print(results_toy)
```

---

## References

* [K-Nearest Neighbors (KNN) Algorithm Overview](https://medium.com/@anthonyhuang1909/understanding-knn-algorithm-in-machine-learning)
* [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)

---

## License

This work is licensed under the **Apache 2.0 License**.
