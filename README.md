# Logistic Regression - Cumulative Lab

## Introduction

In this cumulative lab, you will walk through a complete machine learning workflow with logistic regression, including data preparation, modeling (including hyperparameter tuning), and final model evaluation.

## Objectives

You will be able to:

* Practice identifying and applying appropriate preprocessing steps
* Perform an iterative modeling process, starting from a baseline model
* Practice model validation
* Practice choosing a final logistic regression model and evaluating its performance

## Your Task: Complete an End-to-End ML Process with Logistic Regression on the Credit Card Fraud Dataset

![credit card](images/credit_card.jpg)

<span>Photo by <a href="https://unsplash.com/@markuswinkler?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Markus Winkler</a> on <a href="https://unsplash.com/collections/70397509/credit-card-payment%2C-mobile-payment%2C-online-payment?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Business and Data Understanding

Here will be using the credit card fraud dataset:


```python
# Run this cell without changes
import pandas as pd

df = pd.read_csv('data/creditcard.csv.gz', compression='gzip')  
df
```

As you can see, we have over 280,000 rows, each with 30 feature columns and 1 target column:

* `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset
* `V1` - `V28`: Anonymized variables related to this transaction
* `Amount`: Transaction amount
* `Class` (target column): 1 for fraudulent transactions, 0 otherwise

This is also a highly imbalanced dataset, since fraudulent transactions are relatively rare:


```python
# Run this cell without changes
print("Raw Counts")
print(df["Class"].value_counts())
print()
print("Percentages")
print(df["Class"].value_counts(normalize=True))
```

If we had a model that *always* said that a transaction was not fraudulent (class 0), what accuracy score would we get?


```python
# Replace None with appropriate text
"""
None
"""
```

You will need to take this into account when working through this problem.

### Requirements

#### 1. Perform a Train-Test Split

For a complete end-to-end ML process, we need to create a holdout set that we will use at the very end to evaluate our final model's performance.

#### 2. Build and Evaluate a Baseline Model

Without performing any preprocessing or hyperparameter tuning, build and evaluate a vanilla logistic regression model using log loss and `cross_val_score`.

#### 3. Write a Custom Cross Validation Function

Because we are using preprocessing techniques that differ for train and validation data, we will need a custom function rather than simply preprocessing the entire `X_train` and using `cross_val_score` from scikit-learn.

#### 4. Build and Evaluate Additional Logistic Regression Models

Using the function created in the previous step, build multiple logistic regression models with different hyperparameters in order to minimize log loss.

#### 5. Choose and Evaluate a Final Model

Preprocess the full training set and test set appropriately, then evaluate the final model with various classification metrics in addition to log loss.


```python

```
