# 📊 Supervised Machine Learning — Regression

Welcome to the **Supervised Regression** section of my AI learning journey!  
This folder covers the complete theory and hands-on implementation of regression models using Python, from the very basics to regularization and model evaluation. 🚀

---

## 📚 Topics Covered

Each topic below is backed with Python scripts, Jupyter notebooks, or Markdown notes.  
You can explore, run, or modify the code to understand how supervised regression truly works.

---

### ✅ Basics of Regression

- **`simple-linear-regression.py`**  
  ### **Linear Regression**

- Linear Regression is a supervised machine learning algorithm used to model the relationship between a **dependent variable** (target) and an **independent variable** (feature).
- In **single variable linear regression**, there is only **one input feature** and **one output variable**.
- The model aims to fit a **line of best fit** using the equation:Y=mX+c

   ![Regression Graph](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)

    
    Y = m X + c
    
    Where:
    
    - Y: Predicted output (target)
    - X: Input feature
    - m: Slope of the line (coefficient)
    - c: Y-intercept (constant)
 
  

## 📘 Multiple Linear Regression

**Multiple Linear Regression** is a statistical technique that models the relationship between one **dependent variable** and two or more **independent variables**.

### 🧮 Equation:
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

![Image Description](https://miro.medium.com/v2/resize:fit:560/1*HLN6FxlXrzDtYN0KAlom4A.png)

### 📌 Where:
- **y**: Dependent variable  
- **x₁, x₂, ..., xₙ**: Independent variables  
- **β₀**: Intercept  
- **β₁, β₂, ..., βₙ**: Coefficients of the independent variables  
- **ε**: Error term

  
---

### ⚙️ Math Behind the Learning

### 📊 **Cost Function in Regression**

In regression problems, the **cost function** (also known as **loss function**) measures the difference between the predicted values and the actual values. The goal is to minimize this cost function during model training to make the model's predictions as accurate as possible.

The most common cost functions used in regression are **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

---

#### 1. **Mean Squared Error (MSE)**

MSE is the most commonly used cost function in regression. It measures the average of the squared differences between the predicted values and the actual values.

##### Formula:
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- y_i = Actual value
- y_hat_i = Predicted value
- n = Number of data points

---

#### 2. **Mean Absolute Error (MAE)**

MAE measures the average of the absolute differences between the predicted values and the actual values. It is less sensitive to outliers compared to MSE.

##### Formula:
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- y_i = Actual value
- y_hat_i = Predicted value
- n = Number of data points

---

### ✨ **Choosing the Right Cost Function**
- **MSE** is preferred when you want to heavily penalize large errors, as it squares the errors.
- **MAE** is preferred when you want to treat all errors equally and are more robust to outliers.


  
# Derivatives and Chain Rule in Machine Learning

In machine learning, the objective is to minimize a **cost function** (or loss function), which measures how far the model's predictions are from the actual values. To achieve this, we compute the **derivative** of the cost function with respect to each model parameter (such as weights and biases). These derivatives help determine how to update the parameters to reduce prediction error.

This concept is especially important in optimization algorithms like **gradient descent**, where the model parameters are adjusted iteratively using the gradient (i.e., the derivative) of the cost function.

---

## 🔹 Example: Derivatives in Linear Regression

Consider a simple linear regression model:
ŷ = w₁x + w₀


Where:
- `ŷ` is the predicted value,
- `x` is the input feature,
- `w₁` is the weight (slope),
- `w₀` is the bias (intercept).

The **Mean Squared Error (MSE)** is commonly used as the cost function:

MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

Where:
- `yᵢ` is the actual value,
- `ŷᵢ` is the predicted value.

To minimize the MSE, we compute the partial derivatives with respect to each parameter. For instance, the derivative of MSE with respect to `w₁` is:

∂MSE/∂w₁ = -(2/n) * Σ xᵢ(yᵢ - ŷᵢ)
---

This **gradient** tells us the direction and magnitude to update `w₁` in order to reduce error, which is repeatedly applied in gradient descent.

---

## 🔹 Chain Rule in Machine Learning

The **chain rule** in calculus is used to compute the derivative of **composite functions**, which is common in machine learning when multiple functions are nested (e.g., in neural networks).

In linear regression, the prediction `ŷᵢ` is a function of the weights, and the cost function is a function of these predictions. To compute the derivative of the cost function with respect to weights, we use the chain rule.

### Example using Chain Rule:

Given:

ŷᵢ = w₁xᵢ + w₀
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²


To compute the gradient with respect to `w₁`, apply the chain rule:

∂MSE/∂w₁ = ∂MSE/∂ŷᵢ * ∂ŷᵢ/∂w₁


Where:
- `∂MSE/∂ŷᵢ` is the derivative of the cost with respect to the predicted value,
- `∂ŷᵢ/∂w₁` is how the prediction changes with the weight `w₁`.

By breaking the derivative into smaller parts using the chain rule, we simplify the computation and make it easier to apply gradient descent.

---

### 🧠 Gradient Descent

- **`gradient-descent-theory.md`**  
  An intuitive guide to understanding how gradient descent finds the optimal solution step-by-step.

- **`gradient-descent-implementation.py`**  
  A clean Python implementation of gradient descent. Learn how to write it from scratch and visualize the learning process.

---

### 🤔 Key Insights

- **`why-mse-not-mae.md`**  
  Explore the **mathematical reasoning** behind using MSE over MAE during model training and optimization.

---

### 🧪 Model Evaluation

- **`train-test-split.py`**  
  Learn how to properly split your dataset for training and testing. Prevents overfitting and ensures fair evaluation.

- **`model-evaluation-metrics.py`**  
  Understand and calculate key metrics:  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score

---

### 🔧 Data Preprocessing

- **`data-preprocessing.py`**  
  - **One-Hot Encoding** for categorical features  
  - **StandardScaler** for normalization  
  These preprocessing steps ensure your model learns effectively and fairly.

---

### 🧮 Advanced Regression

- **`polynomial-regression.ipynb`**  
  Learn how to fit **non-linear data** using polynomial features. Understand when and how to use this approach.

---

### ⚠️ Overfitting vs Underfitting

- **`overfitting-underfitting.md`**  
  Know the signs of underfitting (too simple) and overfitting (too complex). Learn to diagnose your model’s behavior.

- **`overfitting-remedies.py`**  
  Practical techniques to solve both problems — from cross-validation to regularization and adding more data.

---

### 📏 Regularization Techniques

- **`l1-l2-regularization.py`**  
  Apply **L1 (Lasso)** and **L2 (Ridge)** regularization in your regression models. Learn how these help prevent overfitting.

---

### ⚖️ Bias-Variance Tradeoff

- **`bias-variance-tradeoff.md`**  
  Explore one of the most important concepts in machine learning:  
  **Bias (error due to underfitting) vs. Variance (error due to overfitting)** — and how to balance them.

---

## 📂 Folder Structure
