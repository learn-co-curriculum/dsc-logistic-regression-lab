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


```python
# __SOLUTION__
import pandas as pd

df = pd.read_csv('data/forest_cover.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>...</th>
      <th>Soil_Type_31</th>
      <th>Soil_Type_32</th>
      <th>Soil_Type_33</th>
      <th>Soil_Type_34</th>
      <th>Soil_Type_35</th>
      <th>Soil_Type_36</th>
      <th>Soil_Type_37</th>
      <th>Soil_Type_38</th>
      <th>Soil_Type_39</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2553</td>
      <td>235</td>
      <td>17</td>
      <td>351</td>
      <td>95</td>
      <td>780</td>
      <td>188</td>
      <td>253</td>
      <td>199</td>
      <td>1410</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>344</td>
      <td>17</td>
      <td>313</td>
      <td>29</td>
      <td>404</td>
      <td>183</td>
      <td>211</td>
      <td>164</td>
      <td>300</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>24</td>
      <td>13</td>
      <td>391</td>
      <td>42</td>
      <td>509</td>
      <td>212</td>
      <td>212</td>
      <td>134</td>
      <td>421</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2038</td>
      <td>50</td>
      <td>17</td>
      <td>408</td>
      <td>71</td>
      <td>474</td>
      <td>226</td>
      <td>200</td>
      <td>102</td>
      <td>283</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>341</td>
      <td>27</td>
      <td>351</td>
      <td>34</td>
      <td>390</td>
      <td>152</td>
      <td>188</td>
      <td>168</td>
      <td>190</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38496</th>
      <td>2396</td>
      <td>153</td>
      <td>20</td>
      <td>85</td>
      <td>17</td>
      <td>108</td>
      <td>240</td>
      <td>237</td>
      <td>118</td>
      <td>837</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38497</th>
      <td>2391</td>
      <td>152</td>
      <td>19</td>
      <td>67</td>
      <td>12</td>
      <td>95</td>
      <td>240</td>
      <td>237</td>
      <td>119</td>
      <td>845</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38498</th>
      <td>2386</td>
      <td>159</td>
      <td>17</td>
      <td>60</td>
      <td>7</td>
      <td>90</td>
      <td>236</td>
      <td>241</td>
      <td>130</td>
      <td>854</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38499</th>
      <td>2384</td>
      <td>170</td>
      <td>15</td>
      <td>60</td>
      <td>5</td>
      <td>90</td>
      <td>230</td>
      <td>245</td>
      <td>143</td>
      <td>864</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38500</th>
      <td>2383</td>
      <td>165</td>
      <td>13</td>
      <td>60</td>
      <td>4</td>
      <td>67</td>
      <td>231</td>
      <td>244</td>
      <td>141</td>
      <td>875</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>38501 rows × 55 columns</p>
</div>



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


```python
# __SOLUTION__
print("Raw Counts")
print(df["Cover_Type"].value_counts())
print()
print("Percentages")
print(df["Cover_Type"].value_counts(normalize=True))
```

    Raw Counts
    0    35754
    1     2747
    Name: Cover_Type, dtype: int64
    
    Percentages
    0    0.928651
    1    0.071349
    Name: Cover_Type, dtype: float64


If we had a model that *always* said that the cover type was ponderosa pine (class 0), what accuracy score would we get?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
We would get an accuracy score of 0.928651, i.e. about 92.9% accuracy

This is because about 92.9% of areas are covered by ponderosa pine
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


```python
# __SOLUTION__

# Import the relevant function
from sklearn.model_selection import train_test_split

# Split df into X and y
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Perform train-test split with random_state=42 and stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
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


```python
# __SOLUTION__

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


```python
# __SOLUTION__
print("Train percent cottonwood/willow:", y_train.value_counts(normalize=True)[1])
print("Test percent cottonwood/willow: ", y_test.value_counts(normalize=True)[1])
```

    Train percent cottonwood/willow: 0.07134199134199135
    Test percent cottonwood/willow:  0.0713692083939331


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


```python
# __SOLUTION__

# Import relevant class and function
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Instantiate a LogisticRegression with random_state=42
baseline_model = LogisticRegression(random_state=42)

# Use cross_val_score with scoring="neg_log_loss" to evaluate the model
# on X_train and y_train
baseline_neg_log_loss_cv = cross_val_score(baseline_model, X_train, y_train, scoring="neg_log_loss")

baseline_log_loss = -(baseline_neg_log_loss_cv.mean())
baseline_log_loss
```

    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    0.17116279160065645



Ok, so we are getting the `ConvergenceWarning`s we expected, and log loss of around 0.171 with our baseline model.

Is that a "good" log loss? That's hard to say — log loss is not particularly interpretable. 

If we had a model that just chose 0 (the majority class) every time, this is the log loss we would get:


```python
# Run this cell without changes
from sklearn.metrics import log_loss
import numpy as np

log_loss(y_train, np.zeros(len(y_train)))
```


```python
# __SOLUTION__
from sklearn.metrics import log_loss
import numpy as np

log_loss(y_train, np.zeros(len(y_train)))
```




    2.4640650865286937



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


```python
# __SOLUTION__
"""
Because of class imbalance, we should add some kind of resampling
step. Specifically we'll use SMOTE.

We are getting a ConvergenceWarning, which means that the gradient
descent algorithm within the logistic regression is failing to find
an optimized answer. We can also see from looking at the dataset that
some of our variables are quite small (0 or 1) while others range in
the thousands. This indicates that we should add scaling to help
"flatten" the landscape being iterated over by normalizing the
various units of the different columns.
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


```python
# __SOLUTION__
"""
Yes, StandardScaler is a scikit-learn transformer so it does work
this way. We transform both the train and test data, using the fit
from the training data only.

Often we are a bit sloppy and we fit and transform the entire
X_train prior to performing cross validation, meaning we have some
risk of leakage. Ideally every train split within the cross
validation would be fit with its own scaler, something that is 
achieved more easily with pipelines.
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


```python
# __SOLUTION__
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

    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    0.17116279160065645



As you can see, this produced the same result as our original cross validation (including the `ConvergenceWarning`s):


```python
# Run this cell without changes
print(baseline_neg_log_loss_cv)
print(kfold_scores)
```


```python
# __SOLUTION__
print(baseline_neg_log_loss_cv)
print(kfold_scores)
```

    [-0.17158243 -0.17213642 -0.1629862  -0.17678272 -0.17232618]
    [-0.17158243 -0.17213642 -0.1629862  -0.17678272 -0.17232618]


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


```python
# __SOLUTION__

# Import relevant sklearn and imblearn classes
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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
        scaler = StandardScaler()
        # Fit and transform X_t
        X_t_scaled = scaler.fit_transform(X_t)
        # Transform X_val
        X_val_scaled = scaler.transform(X_val)
        
        # Instantiate SMOTE with random_state=42 and sampling_strategy=0.28
        sm = SMOTE(random_state=42, sampling_strategy=0.28)
        # Fit and transform X_t_scaled and y_t using sm
        X_t_oversampled, y_t_oversampled = sm.fit_resample(X_t_scaled, y_t)
        
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




    0.13235904968769785



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


```python
# __SOLUTION__
print(-baseline_neg_log_loss_cv.mean())
print(-preprocessed_neg_log_loss_cv.mean())
```

    0.17116279160065645
    0.13235904968769785


Looks like our preprocessing with `StandardScaler` and `SMOTE` has provided some improvement over the baseline! Let's move on to Step 4.
