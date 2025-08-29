# Understanding Linear Regression in Machine Learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg)](https://pytorch.org/) [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)

---

## ML Linear Regression

Linear Regression is a foundational supervised learning algorithm used to predict a continuous output variable based on one or more input features. It assumes a linear relationship between independent variables (features) and the dependent variable (target).

---

## 1. Linear Regression Model

### Single Feature (Univariate)

For one input feature \$x\$, the linear regression model is:

$$
\hat{y} = w \cdot x + b
$$

Where:

* \$x\$ = input feature
* \$w\$ = weight (coefficient)
* \$b\$ = bias (intercept)
* \$\hat{y}\$ = predicted output

### Multiple Features (Multivariate)

For \$n\$ features:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

Where \$x\_i\$ are input features and \$w\_i\$ their corresponding weights.

---

## 2. Cost Function (Mean Squared Error)

The **Mean Squared Error (MSE)** measures the average squared difference between actual and predicted values:

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

Where:

* \$m\$ = number of samples
* \$y\_i\$ = actual value
* \$\hat{y}\_i\$ = predicted value

---

## 3. Gradient Descent Update

To minimize the cost function, weights and bias are updated iteratively using gradient descent:

$$
w := w - \alpha \frac{\partial \text{MSE}}{\partial w}, \quad b := b - \alpha \frac{\partial \text{MSE}}{\partial b}
$$

Where \$\alpha\$ is the learning rate.

---

## 4. Objective Function

The goal of linear regression is to minimize the prediction error:

$$
\min_{w,b} \text{MSE}(w, b) = \frac{1}{m} \sum_{i=1}^{m} \big(y_i - (w^T x_i + b)\big)^2
$$

---

## 5. Advantages

* Simple and easy to understand
* Fast to compute
* Can be improved using regularization (L1 or L2)
* Supports L2 regularization (weight decay)
* Implementable in PyTorch or Scikit-learn
* Visualization helps to understand regression fit

---

## 6. Limitations

* Can only model linear relationships
* Sensitive to outliers
* Struggles with high-dimensional data without proper regularization

---

## 7. Installation

```bash
pip install torch scikit-learn
```

---

## 8. PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Generate synthetic data
x = torch.randn((5, 1))
y = 2 * x + 1 + 0.1 * torch.randn(x.size())

# Initialize model
input_dim = x.shape[1]
model = LinearRegression(input_dim=input_dim, output_dim=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0)  # L2 regularization optional

epochs = 100

for epoch in range(epochs):
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1} | Loss: {loss.item():.4f}')
```

---

## 9. Visualization

![Linear Regression GIF](https://miro.medium.com/v2/resize\:fit:1400/format\:webp/1*OKjRI4lvO1SDUmRoV4wkBQ.gif)

---

## References

* [Understanding Linear Regression in Machine Learning](https://medium.com/@anthonyhuang1909/understanding-linear-regression-in-machine-learning-e90d157ec1dd)

---

## License

This project is licensed under the Apache 2.0 License.
