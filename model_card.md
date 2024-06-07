# Model Card

This model provides a runtime time estimator that can be used in the optimal scheduling of calibration tasks (see the
README.md file for a detailed description of the calibration task). A calibration task consists of the calculation of
all second-order partial derivatives of the corresponding structured product with respect to all relevant risk factors.

# Model Description
**Input:** The input variables to the model are as follows:

* `category`: An integer variable from the range of 0 - 3 specifying the structured product type
  * 0: class label for barrier reverse convertibles,
  * 1: class label for auto-callables,
  * 2: class label for callable barrier reverse convertible,
  * 3: Class label for accumulators and decumulators.
* `num_underlyings`: An non-negative integer variable specifying the number of underlyings of the corresponding
  structured product.
* `time_to_expiry`: A positive float variable specifying the time-to-expiry for the corresponding structured
   product (measured in years). 
* `num_paths`: A non-negative integer variable specifying the number of paths used in the Monte-Carlo simulation
   of each pricing call of the corresponding structured product. 
* `num_exercise_events`: A non-negative integer variable specifying the number of exercise events of the corresponding
  structured product. 

**Output:**

A non-negative float which provides an estimate for the calibration time used by the corresponding calibration task
(measured in seconds).

**Model Architecture:**

The model consist of a `RandomForestRegressor` from the package `scklearn.ensemble` with the following hyperparameters:

* `n_estimators`: The number of trees in the forest (default=500).
* `max_depth`: The maximum depth of the tree (default=8).
* `criterion`: The function to measure the quality of a split (default=`mse`).
* `min_samples_split`: The minimum number of samples required to split an internal node (default=2).
* `min_samples_leaf`: The minimum number of samples in newly created leaves (default=1).
* `min_weight_fraction_leaf`: The minimum weighted fraction of the input samples required to be at a leaf node (default=0).
* `max_features`: The number of features to consider when looking for the best split (default=`auto`).
* `max_leaf_nodes`: Grow trees with max_leaf_nodes in best-first fashion (default=None).
* `bootstrap`: Whether bootstrap samples are used when building trees (default=True).
* `oob_score`: Whether to use out-of-bag samples to estimate the generalization error (default=False).

see the following [page](https://scikit-learn.org/0.16/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn-ensemble-randomforestregressor)
for a detailed description.

# Performance

* training MSE: 5.2745
* training score: 0.9943
* test MSE: 44.8724
* test score: 0.9637
