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
\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
\]

![Multiple Linear Regression](https://upload.wikimedia.org/wikipedia/commons/8/86/Multiple_regression.svg)

### 📌 Where:
- **y**: Dependent variable  
- **x₁, x₂, ..., xₙ**: Independent variables  
- **β₀**: Intercept  
- **β₁, β₂, ..., βₙ**: Coefficients of the independent variables  
- **ε**: Error term

  
---

### ⚙️ Math Behind the Learning

- **`cost-function.py`**  
  Introduction to **Mean Squared Error (MSE)** — how models measure error and why it's essential for optimization.

- **`derivative-partial-derivative.py`**  
  Grasp the importance of derivatives in machine learning. You'll learn how models use calculus to update weights.

- **`chain-rule.py`**  
  Dive into nested function differentiation. Learn how complex learning models use the chain rule to compute gradients.

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
