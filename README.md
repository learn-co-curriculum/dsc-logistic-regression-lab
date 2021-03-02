# Logistic Regression - Cumulative Lab

## Introduction

In this cumulative lab, you will walk through a complete machine learning workflow with logistic regression, including data preparation, modeling (including hyperparameter tuning), and final model evaluation.

## Objectives

You will be able to:

* Practice identifying and applying appropriate preprocessing steps
* Perform an iterative modeling process, starting from a baseline model
* Practice model validation
* Practice choosing a final logistic regression model and evaluating its performance

## Your Task: Complete an End-to-End ML Process with Logistic Regression on the Forest Cover Dataset

![forest road](images/forest_road.jpg)

<span>Photo by <a href="https://unsplash.com/@von_co?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Ivana Cajina</a> on <a href="https://unsplash.com/s/photos/forest-satellite?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Business and Data Understanding

Here we will be using an adapted version of the forest cover dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/covertype). Each record represents a 30 x 30 meter cell of land within Roosevelt National Forest in northern Colorado, which has been labeled as `Cover_Type` 1 for "Cottonwood/Willow" and `Cover_Type` 0 for "Ponderosa Pine". (The original dataset contained 7 cover types but we have simplified it.)

The task is to predict the `Cover_Type` based on the available cartographic variables:


```python
# Run this cell without changes
import pandas as pd

df = pd.read_csv('data/forest_cover.csv')  
df
```

As you can see, we have over 38,000 rows, each with 54 feature columns and 1 target column:

* `Elevation`: Elevation in meters
* `Aspect`: Aspect in degrees azimuth
* `Slope`: Slope in degrees
* `Horizontal_Distance_To_Hydrology`: Horizontal dist to nearest surface water features in meters
* `Vertical_Distance_To_Hydrology`: Vertical dist to nearest surface water features in meters
* `Horizontal_Distance_To_Roadways`: Horizontal dist to nearest roadway in meters
* `Hillshade_9am`: Hillshade index at 9am, summer solstice
* `Hillshade_Noon`: Hillshade index at noon, summer solstice
* `Hillshade_3pm`: Hillshade index at 3pm, summer solstice
* `Horizontal_Distance_To_Fire_Points`: Horizontal dist to nearest wildfire ignition points, meters
* `Wilderness_Area_x`: Wilderness area designation (4 columns)
* `Soil_Type_x`: Soil Type designation (40 columns)
* `Cover_Type`: 1 for cottonwood/willow, 0 for ponderosa pine

This is also an imbalanced dataset, since cottonwood/willow trees are relatively rare in this forest:


```python
# Run this cell without changes
print("Raw Counts")
print(df["Cover_Type"].value_counts())
print()
print("Percentages")
print(df["Cover_Type"].value_counts(normalize=True))
```

If we had a model that *always* said that the cover type was ponderosa pine (class 0), what accuracy score would we get?


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

## 1. Perform a Train-Test Split

This process should be fairly familiar by now. In the cell below, use the variable `df` (that has already been created) in order to create `X` and `y`, then training and test sets using `train_test_split` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)).

We'll use a random state of 42 and `stratify=y` (to ensure an even balance of fraud/not fraud rows) in the train-test split. Recall that the target is `Cover_Type`.


```python
# Replace None with appropriate code

# Import the relevant function
None

# Split df into X and y
X = None
y = None

# Perform train-test split with random_state=42 and stratify=y
X_train, X_test, y_train, y_test = None
```

Check that you have the correct data shape before proceeding:


```python
# Run this cell without changes

# X and y training data should have the same number of rows
assert X_train.shape[0] == y_train.shape[0] and X_train.shape[0] == 28875

# X and y testing data should have the same number of rows
assert X_test.shape[0] == y_test.shape[0] and X_test.shape[0] == 9626

# Both X should have 33 columns
assert X_train.shape[1] == X_test.shape[1] and X_train.shape[1] == 54

# Both y should have 1 column
assert len(y_train.shape) == len(y_test.shape) and len(y_train.shape) == 1
```

Also, we should have roughly equal percentages of cottonwood/willow trees for train vs. test targets:


```python
# Run this cell without changes
print("Train percent cottonwood/willow:", y_train.value_counts(normalize=True)[1])
print("Test percent cottonwood/willow: ", y_test.value_counts(normalize=True)[1])
```

## 2. Build and Evaluate a Baseline Model

Using scikit-learn's `LogisticRegression` model, instantiate a classifier with `random_state=42`. Then use `cross_val_score` with `scoring="neg_log_loss"` to find the average cross-validated log loss for this model on `X_train` and `y_train`.

* [`LogisticRegression` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [`cross_val_score` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

(Similar to RMSE, the internal implementation of `cross_val_score` requires that we use "negative log loss" instead of just log loss. The code provided negates the result for you.)

**The code below should produce a warning** but not an error. Because we have not scaled the data, we expect to get a `ConvergenceWarning` five times (once for each fold of cross validation).


```python
# Replace None with appropriate code

# Import relevant class and function
None
None

# Instantiate a LogisticRegression with random_state=42
baseline_model = None

# Use cross_val_score with scoring="neg_log_loss" to evaluate the model
# on X_train and y_train
baseline_neg_log_loss_cv = None

baseline_log_loss = -(baseline_neg_log_loss_cv.mean())
baseline_log_loss
```

Ok, so we are getting the `ConvergenceWarning`s we expected, and log loss of around 0.171 with our baseline model.

Is that a "good" log loss? That's hard to say — log loss is not particularly interpretable. 

If we had a model that just chose 0 (the majority class) every time, this is the log loss we would get:


```python
# Run this cell without changes
from sklearn.metrics import log_loss
import numpy as np

log_loss(y_train, np.zeros(len(y_train)))
```

Loss is a metric where lower is better, so our baseline model is clearly an improvement over just guessing the majority class every time.

Even though it is difficult to interpret, the 0.171 value will be a useful baseline as we continue modeling, to see if we are actually making improvements or just getting slightly better performance by chance.

We will also use other metrics at the last step in order to describe the final model's performance in a more user-friendly way.

## 3. Write a Custom Cross Validation Function

### Conceptual Considerations

First, consider: which preprocessing steps should be taken with this dataset? Recall that our data is imbalanced, and that it caused a `ConvergenceWarning` for our baseline model.


```python
# Replace None with appropriate text
"""
None
"""
```

As you likely noted above, we should use some kind of resampling technique to address the large class imbalance. Let's use `SMOTE` (synthetic minority oversampling, [documentation here](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)), which creates synthetic examples of the minority class to help train the model.

Does SMOTE work just like a typical scikit-learn transformer, where you fit the transformer on the training data then transform both the training and the test data?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION
"""
No, SMOTE does not work like that. We never want to oversample the
minority class in the test data, because then we are generating
metrics based on synthetic data and not actual data.

Instead, we only want to fit and transform the training data, and
leave the testing data alone.
"""
```

As you also likely noted above, we should use some transformer to normalize the data. Let's use a `StandardScaler` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)).

Does `StandardScaler` work just like a typical scikit-learn transformer, where you fit the transformer on the training data then transform both the training and the test data?


```python
# Replace None with appropriate text
"""
None
"""
```

(At this point it's a good idea to double-check your answers against the `solution` branch to make sure you understand the setup.)

### Using `StratifiedKFold`

As you can see from the `cross_val_score` documentation linked above, "under the hood" it is using `StratifiedKFold` for classification tasks.

Essentially `StratifiedKFold` is just providing the information you need to make 5 separate train-test splits inside of `X_train`. Then there is other logic within `cross_val_score` to fit and evaluate the provided model.

So, if our original code looked like this:

```python
baseline_model = LogisticRegression(random_state=42)
baseline_neg_log_loss_cv = cross_val_score(baseline_model, X_train, y_train, scoring="neg_log_loss")
baseline_log_loss = -(baseline_neg_log_loss_cv.mean())
baseline_log_loss
```

The equivalent of the above code using `StratifiedKFold` would look something like this:


```python
# Run this cell without changes
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Negative log loss doesn't exist as something we can import,
# but we can create it
neg_log_loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

# Instantiate the model (same as previous example)
baseline_model = LogisticRegression(random_state=42)

# Create a list to hold the score from each fold
kfold_scores = np.ndarray(5)

# Instantiate a splitter object and loop over its result
kfold = StratifiedKFold()
for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
    # Extract train and validation subsets using the provided indices
    X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Clone the provided model and fit it on the train subset
    temp_model = clone(baseline_model)
    temp_model.fit(X_t, y_t)
    
    # Evaluate the provided model on the validation subset
    neg_log_loss_score = neg_log_loss(temp_model, X_val, y_val)
    kfold_scores[fold] = neg_log_loss_score
    
-(kfold_scores.mean())
```

As you can see, this produced the same result as our original cross validation (including the `ConvergenceWarning`s):


```python
# Run this cell without changes
print(baseline_neg_log_loss_cv)
print(kfold_scores)
```

So, what is the point of doing it this way, instead of the much-shorter `cross_val_score` approach?

**Using `StratifiedKFold` "by hand" allows us to customize what happens inside of that loop.**

Therefore we can apply these preprocessing techniques appropriately:

1. Fit a `SMOTE` object and transform only the training subset
2. Fit a `StandardScaler` object on the training subset (not the full training data) and transform both the train and test subsets

### Writing a Custom Cross Validation Function with `StratifiedKFold`

In the cell below, we have set up a function `custom_cross_val_score` that has an interface that resembles the `cross_val_score` function from scikit-learn.

Most of it is set up for you already, all you need to do is add the `SMOTE` and `StandardScaler` steps described above.


```python
# Replace None with appropriate code

# Import relevant sklearn and imblearn classes
None
None

def custom_cross_val_score(estimator, X, y):
    # Create a list to hold the score from each fold
    kfold_scores = np.ndarray(5)

    # Instantiate a splitter object and loop over its result
    kfold = StratifiedKFold(n_splits=5)
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        # Extract train and validation subsets using the provided indices
        X_t, X_val = X.iloc[train_index], X.iloc[val_index]
        y_t, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Instantiate StandardScaler
        scaler = None
        # Fit and transform X_t
        X_t_scaled = None
        # Transform X_val
        X_val_scaled = None
        
        # Instantiate SMOTE with random_state=42 and sampling_strategy=0.28
        sm = None
        # Fit and transform X_t_scaled and y_t using sm
        X_t_oversampled, y_t_oversampled = None
        
        # Clone the provided model and fit it on the train subset
        temp_model = clone(estimator)
        temp_model.fit(X_t_oversampled, y_t_oversampled)
        
        # Evaluate the provided model on the validation subset
        neg_log_loss_score = neg_log_loss(temp_model, X_val_scaled, y_val)
        kfold_scores[fold] = neg_log_loss_score
        
    return kfold_scores
        
model_with_preprocessing = LogisticRegression(random_state=42, class_weight={1: 0.28})
preprocessed_neg_log_loss_cv = custom_cross_val_score(model_with_preprocessing, X_train, y_train)
- (preprocessed_neg_log_loss_cv.mean())
```

The output you get should be about 0.132, and there should no longer be a `ConvergenceWarning`.

If you're not getting the right output, double check that you are applying the correct transformations to the correct variables:

1. `X_t` should be scaled to create `X_t_scaled`, then `X_t_scaled` should be resampled to create `X_t_oversampled`, then `X_t_oversampled` should be used to fit the model
2. `X_val` should be scaled to create `X_val_scaled`, then `X_val_scaled` should be used to evaluate `neg_log_loss`
3. `y_t` should be resampled to create `y_t_oversampled`, then `y_t_oversampled` should be used to fit the model
4. `y_val` should not be transformed in any way. It should just be used to evaluate `neg_log_loss`

Another thing to check is that you used `sampling_strategy=0.28` when you instantiated the `SMOTE` object.

If you are getting the right output, great!  Let's compare that to our baseline log loss:


```python
# Run this cell without changes
print(-baseline_neg_log_loss_cv.mean())
print(-preprocessed_neg_log_loss_cv.mean())
```

Looks like our preprocessing with `StandardScaler` and `SMOTE` has provided some improvement over the baseline! Let's move on to Step 4.
