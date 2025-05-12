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

## Overview

This repository contains Python code that demonstrates the concepts of **derivatives**, **partial derivatives**, and the **chain rule** in the context of **machine learning (ML) regression models**. These concepts are fundamental for understanding optimization algorithms, particularly in training models via **gradient descent**.

In ML, the objective is to minimize a **cost function** (or **loss function**) that measures how far the model's predictions are from the true values. By calculating the derivative of the cost function with respect to each parameter (such as weights in a linear regression model), we determine the direction to adjust these parameters to reduce the error.

---

### derivative-partial-derivative

This script explains the importance of derivatives and partial derivatives in machine learning regression models, particularly in optimization. These derivatives help in adjusting model parameters iteratively to minimize errors during training, typically through **gradient descent**.

### Example in Machine Learning Regression:

Consider a simple linear regression model:

\[
y = w_1 x + w_0
\]

Where:
- \( y \) is the predicted value,
- \( x \) is the input feature,
- \( w_1 \) is the weight (slope),
- \( w_0 \) is the bias (intercept).

To train the model, we use **Mean Squared Error (MSE)** as the cost function:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Where:
- \( y_i \) is the actual value,
- \( \hat{y}_i \) is the predicted value.

In **gradient descent**, the derivative of the **MSE** with respect to the model parameters (weights) is calculated. The **partial derivatives** guide us to adjust the weights in the direction that minimizes the error.

For example, the partial derivative of **MSE** with respect to \( w_1 \) (slope) is:

\[
\frac{\partial MSE}{\partial w_1} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)
\]

This derivative provides the gradient, which is used to update \( w_1 \) in the direction of decreasing error, and the process repeats iteratively during model training.

---

### chain-rule

The **chain rule** is a fundamental concept in calculus used to compute the derivative of a composite function. In machine learning, the chain rule is particularly helpful when there are multiple steps or functions involved in the model.

In simple linear regression or other ML models, the chain rule aids in computing the derivative of the cost function with respect to the model's parameters (weights). This is essential for optimization, particularly in **gradient descent**.

### Example in Machine Learning Regression (Chain Rule):

Consider the same linear regression model as before:

\[
y = w_1 x + w_0
\]

The cost function is **Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

To minimize the **MSE**, we need to compute the gradients of the MSE with respect to the weights \( w_0 \) (bias) and \( w_1 \) (slope). The chain rule allows us to break down these gradients into simpler components.

For example, to find the gradient with respect to \( w_1 \), we apply the chain rule:

\[
\frac{\partial MSE}{\partial w_1} = \frac{\partial MSE}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial w_1}
\]

Here:
- \( \frac{\partial MSE}{\partial \hat{y}_i} \) represents the change in error with respect to the predicted value \( \hat{y}_i \),
- \( \frac{\partial \hat{y}_i}{\partial w_1} \) shows how the predicted value \( \hat{y}_i \) changes with respect to \( w_1 \).

By applying the chain rule, we can break down the derivative into parts, making it easier to compute gradients and update the model's parameters through gradient descent.

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
