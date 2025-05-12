# 📊 Supervised Machine Learning — Regression

Welcome to the **Supervised Regression** section of my AI learning journey!  
This folder covers the complete theory and hands-on implementation of regression models using Python, from the very basics to regularization and model evaluation. 🚀

---

## 📚 Topics Covered

Each topic below is backed with Python scripts, Jupyter notebooks, or Markdown notes.  
You can explore, run, or modify the code to understand how supervised regression truly works.

---

### ✅ Basics of Regression


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
- yᵢ = Actual value
- ŷᵢ = Predicted value
- n = Number of data points

---

#### 2. **Mean Absolute Error (MAE)**

MAE measures the average of the absolute differences between the predicted values and the actual values. It is less sensitive to outliers compared to MSE.

##### Formula:
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- yᵢ = Actual value
- ŷᵢ = Predicted value
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

# 📉 Gradient Descent in Machine Learning

**Gradient Descent** is an optimization algorithm used to minimize a **cost function** by iteratively updating model parameters (like weights and biases) in the direction of the steepest descent — i.e., the **negative gradient** of the cost function.

It is the backbone of optimization in machine learning and deep learning models.

---

## 🧮 Mathematical Intuition

In a supervised learning setup like **linear regression**, we try to fit a line:

ŷ = w₁x + w₀

Where:
- `ŷ` is the predicted output,
- `w₁` is the weight (slope),
- `w₀` is the bias (intercept),
- `x` is the input feature.

We define the **cost function** to measure the difference between predicted values and actual values. A common choice is **Mean Squared Error (MSE)**:

J(w₁, w₀) = (1/n) * Σ (yᵢ - ŷᵢ)²
= (1/n) * Σ (yᵢ - (w₁xᵢ + w₀))²


---

## 🔁 Gradient Descent Algorithm

To minimize `J(w₁, w₀)`, we compute the **partial derivatives**:


∂J/∂w₁ = -(2/n) * Σ xᵢ (yᵢ - ŷᵢ)
∂J/∂w₀ = -(2/n) * Σ (yᵢ - ŷᵢ)


We then update the weights using a small constant called the **learning rate** `α`:

w₁ := w₁ - α * ∂J/∂w₁
w₀ := w₀ - α * ∂J/∂w₀


Repeat the updates until the model converges — i.e., the cost stops decreasing significantly.

---

## 🧠 Intuition

- If the gradient is **positive** → decrease the weight to reduce error.
- If the gradient is **negative** → increase the weight to reduce error.
- The learning rate `α` controls how large a step we take during updates.

---

## ✅ Applications

Gradient Descent allows machine learning models to **learn from data** by minimizing the cost function. It’s a foundational technique used in:

- 🔹 Linear Regression  
- 🔹 Logistic Regression  
- 🔹 Neural Networks  
- 🔹 Deep Learning  

---

> 🔁 Gradient Descent is like rolling a ball downhill — you take steps proportional to the slope of the hill (gradient) until you reach the bottom (minimum error).


---

## 🧪 Model Evaluation

Evaluating your machine learning model's performance is critical to ensure it's learning correctly and generalizing well to unseen data. This section provides scripts and explanations for splitting data and evaluating regression models using standard metrics.

---

### `train-test-split`

**Purpose**:  
Split your dataset into **training** and **testing** sets using `train_test_split` from `sklearn.model_selection`.

**Why?**  
- Prevents **overfitting** by evaluating on data the model hasn't seen during training.
- Helps assess how well the model will perform in real-world scenarios.

**Example Code**:
```python
from sklearn.model_selection import train_test_split

# X = features, y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 80% for training, 20% for testing
```

## 📊 Regression Evaluation Metrics

Evaluate the performance of your regression model using the following metrics:

---

### 1️⃣ Mean Absolute Error (MAE)

- **Formula**: MAE = (1/n) × Σ | yᵢ - ŷᵢ |
- **Intuition**: Measures the average magnitude of errors in a set of predictions, without considering their direction.
- **Interpretation**: Lower MAE indicates better model performance.
- **Unit**: Same as the target variable.

---

### 2️⃣ Mean Squared Error (MSE)

- **Formula**: MSE = (1/n) × Σ ( yᵢ - ŷᵢ )²
- **Intuition**: Like MAE, but squares the errors to penalize larger mistakes.
- **Interpretation**: Lower MSE indicates better accuracy.
- **Unit**: Squared unit of the target variable.

---

### 3️⃣ Root Mean Squared Error (RMSE)

- **Formula**: RMSE = √MSE
- **Intuition**: Square root of MSE; easier to interpret since it's in the same units as the target variable.
- **Interpretation**: Lower RMSE indicates better performance.
- **Unit**: Same as the target variable.

---

### 4️⃣ R² Score (Coefficient of Determination)

- **Formula**: R² = root(MSE)
- **Intuition**: Measures how well the regression predictions approximate the actual data.
- **Range**:  
  - 1.0: Perfect predictions  
  - 0.0: No better than predicting the mean  
  - < 0: Worse than predicting the mean  
- **Interpretation**: Higher R² indicates better model fit.
- **Unit**: Unitless

---

✅ These metrics give you insight into how accurate and reliable your regression model is.

---

## 🧪 Example Code

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Example data
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Evaluation Metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# Results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
```

---

# 📊 Data Preprocessing

Data preprocessing is an essential step in machine learning to clean, transform, and prepare your raw data for effective model training. This step includes techniques such as **One-Hot Encoding**, **Standard Scaling**, and handling **Missing Data** to ensure that the model receives high-quality input data.

---

## 🛠️ Key Steps in Data Preprocessing

---

### 1️⃣ **One-Hot Encoding**

**Purpose**:  
One-Hot Encoding is used to convert **categorical variables** into a numerical format that machine learning models can understand. It creates binary columns for each category, where a `1` indicates the presence of the category and `0` indicates its absence.

**Why It's Important**:  
Categorical data, like text or labels, cannot be directly fed into machine learning algorithms. One-Hot Encoding avoids the issue of creating arbitrary numeric labels, which might introduce unintended ordering into non-ordinal categories.

**Example**:  
For a column "City" with values such as `["Delhi", "Mumbai", "Chennai"]`, One-Hot Encoding would create separate binary columns:

| City   | City_Delhi | City_Mumbai | City_Chennai |
|--------|------------|-------------|--------------|
| Delhi  | 1          | 0           | 0            |
| Mumbai | 0          | 1           | 0            |
| Chennai| 0          | 0           | 1            |

---

### 2️⃣ **Standard Scaling (Normalization)**

**Purpose**:  
Standard Scaling is used to transform numerical data so that it has a **mean of 0** and a **standard deviation of 1**. This ensures that the model treats all features equally, regardless of their original scales.

**Why It's Important**:  
Different features might have different units or ranges (e.g., one feature might be in thousands, and another in tens), and some machine learning models are sensitive to these differences. Scaling ensures that all features contribute equally to the learning process.

**Example**:  
For a column "Age" with values `[25, 35, 45]`, Standard Scaling would transform them into values that have a mean of 0 and a standard deviation of 1.

---

### 3️⃣ **Handling Missing Data**

**Purpose**:  
Handling missing data is crucial since most machine learning algorithms can't process incomplete datasets. There are different ways to handle missing data, including **imputation** (filling missing values with the mean, median, or mode) or **deletion** (removing rows or columns with missing values).

**Why It's Important**:  
Ignoring missing data can lead to inaccurate model predictions. By either filling or removing missing values, we ensure the model works with complete information.

**Common Approaches**:
- **Imputation**: Replace missing values with the mean, median, or mode of the column.
- **Deletion**: Drop rows or columns that contain missing values.

---

### 4️⃣ **Label Encoding**

**Purpose**:  
Label Encoding is used to convert **binary or ordinal categorical** features into numeric values. This technique assigns a unique integer to each category, making it easier for machine learning models to interpret the data.

**Why It's Important**:  
It’s a simpler alternative to One-Hot Encoding when dealing with binary or ordinal features, as it does not require multiple columns.

**Example**:  
For a column "Gender" with values `["Male", "Female"]`, Label Encoding would convert it into numeric values:
- "Male" → 0
- "Female" → 1

---

### 5️⃣ **Min-Max Scaling**

**Purpose**:  
Min-Max Scaling transforms feature values to a fixed range, typically `[0, 1]`. It’s useful when you need features on the same scale and when features need to be bounded between a specific range.

**Why It's Important**:  
Some machine learning models, like neural networks, perform better when features are scaled to a specific range. Min-Max scaling helps ensure that no feature dominates others due to differences in range.

**Example**:  
For a column "Age" with values `[20, 30, 40, 50]`, Min-Max Scaling would transform them to values between 0 and 1, where:
- Min = 20, Max = 50
- For "Age" = 20:  
   = (20 - 20)/(50 - 20) = 0
- For "Age" = 50:  
   = (50 - 20)/(50 - 20) = 1
After Min-Max Scaling, the values would be `[0, 0.33, 0.66, 1]`.

---

## 🧰 Summary of Preprocessing Steps

| Step                  | Purpose                              | When to Use                                   |
|-----------------------|--------------------------------------|-----------------------------------------------|
| **One-Hot Encoding**   | Convert categorical data to binary   | For nominal categorical features              |
| **Standard Scaling**   | Normalize numerical data             | When features have different ranges/units     |
| **Label Encoding**     | Convert categorical labels to numbers | For binary or ordinal categorical data       |
| **Missing Data**       | Handle incomplete data               | When data is missing (fill or drop)           |
| **Min-Max Scaling**    | Normalize to a fixed range [0, 1]    | When you need bounded feature range           |

---

# Advanced Regression Techniques

This project covers various advanced regression techniques, including handling non-linear data, preventing overfitting/underfitting, regularization methods, and understanding the bias-variance tradeoff. The following sections explain each concept in detail.

---

## 🧮 Advanced Regression

**Purpose**:  
 Polynomial regression allows the model to learn non-linear relationships by adding higher-degree polynomial features.

**Key Concepts**:
- Polynomial features are added to the original dataset, such as quadratic (x²), cubic (x³), etc.
- **When to Use**: Use when you suspect a non-linear relationship between variables.

**Example**:  
- Fitting a quadratic curve to a dataset: \( y = a + b_1x + b_2x^2 \)

### ⚠️ Overfitting vs Underfitting

Overfitting and underfitting are common issues in machine learning that affect model performance.

**Key Concepts**:
- **Overfitting**: The model is too complex and learns the noise in the data, leading to poor generalization.
- **Underfitting**: The model is too simple and cannot capture the underlying patterns in the data.

**How to Diagnose**:
- **Cross-validation**: Use cross-validation techniques to understand how well the model generalizes.
- **Training and Test Error**: A large gap between training and test error indicates overfitting, while both errors being high suggests underfitting.

### `overfitting-remedies`

**Remedies**:
1. **Cross-Validation**: Helps evaluate model performance across multiple subsets of the data.
2. **Regularization**: L1 (Lasso) and L2 (Ridge) regularization help prevent overfitting.
3. **Adding More Data**: More data can help the model generalize better.
4. **Pruning**: Reducing complexity in decision trees or other models.

---

### 📏 Regularization Techniques


# L1 (Lasso) and L2 (Ridge) Regularization for Regression Models

# Regularization methods help in reducing the model complexity and avoid fitting noise in the data, especially in the presence of multicollinearity or when the number of features is large.

- **Lasso (L1)**: It penalizes the absolute value of coefficients and can drive some coefficients to zero, effectively performing feature selection.
- **Ridge (L2)**: It penalizes the squared value of the coefficients and shrinks them evenly, preventing them from becoming too large but does not eliminate any feature entirely.

## Key Concepts

- **Lasso (L1 Regularization)**:
    - Lasso applies a penalty proportional to the absolute value of the coefficients.
    - Can reduce some coefficients to zero, which helps in feature selection.
    - Suitable when you expect that only a few features are significant.

    **Mathematical Formula**:
    The objective function for Lasso regression is:
    \[
    \min_{\beta_0, \beta} \left( \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right)
    \]
    where:
    - \(y_i\) is the actual value,
    - \(\hat{y}_i\) is the predicted value,
    - \(\beta_j\) is the coefficient for feature \(j\),
    - \(\lambda\) is the regularization parameter controlling the strength of regularization.

- **Ridge (L2 Regularization)**:
    - Ridge applies a penalty proportional to the square of the coefficients.
    - Reduces the size of coefficients but doesn’t eliminate them.
    - Suitable when you believe all features have some impact but want to control their magnitude.

    **Mathematical Formula**:
    The objective function for Ridge regression is:
    \[
    \min_{\beta_0, \beta} \left( \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right)
    \]
    where:
    - \(y_i\) is the actual value,
    - \(\hat{y}_i\) is the predicted value,
    - \(\beta_j\) is the coefficient for feature \(j\),
    - \(\lambda\) is the regularization parameter controlling the strength of regularization.

## How It Works
- Both Lasso and Ridge regression are variations of linear regression that add a regularization term to the loss function.
- **Lasso**: Adds a penalty term of the form \( \lambda \sum |\beta_j| \) to the linear regression loss function, where \( \lambda \) is a tuning parameter.
- **Ridge**: Adds a penalty term of the form \( \lambda \sum \beta_j^2 \) to the linear regression loss function.

## When to Use
- **Lasso**: Use Lasso when you expect only a few features to be significant and want to eliminate others entirely.
- **Ridge**: Use Ridge when you believe all features have some impact but want to reduce their magnitude to avoid overfitting.

---
### ⚖️ Bias-Variance Tradeoff


**Key Concepts**:
- **Bias**: The error due to overly simplistic assumptions about the data (underfitting).
- **Variance**: The error due to the model's sensitivity to small fluctuations in the data (overfitting).
- **Goal**: Find a balance between bias and variance to build an optimal model.

**Ideal Scenario**:
- Low bias and low variance result in good generalization.

**Tradeoff**:
- **High Bias, Low Variance**: Often leads to underfitting.
- **Low Bias, High Variance**: Leads to overfitting.
- **Balanced Bias and Variance**: Leads to optimal performance.

---



