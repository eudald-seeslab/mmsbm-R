# mmsbm <img src="man/figures/logo.png" align="right" height="139" alt="mmsbm logo" />

<!-- badges: start -->
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
<!-- badges: end -->

An R implementation of Mixed Membership Stochastic Block Models (MMSBM) for recommendation systems, based on the work by Godoy-Lorite et al. (2016). This package provides an efficient, vectorized implementation using base R array operations for the hot path and parallel processing via the [future](https://future.futureverse.org/) framework.

This is an R port of the [Python mmsbm package](https://github.com/eudald-seeslab/mmsbm).

## Features

- Fast, vectorized EM implementation.
- Support for both simple and cross-validated fitting.
- Parallel processing for multiple sampling runs.
- Tidymodels-style API: `predict()` returns tibbles, `metrics()` returns yardstick-style output.

## Installation

From GitHub:

```r
# install.packages("devtools")
devtools::install_github("eudald-seeslab/mmsbm-R")
```

## Usage

### Data Format

The input data should be a data.frame with exactly 3 columns: users, items, and ratings.

```r
library(mmsbm)

train <- data.frame(
  users   = paste0("user", sample(0:4, 100, replace = TRUE)),
  items   = paste0("item", sample(0:9, 100, replace = TRUE)),
  ratings = sample(1:5, 100, replace = TRUE)
)

test <- data.frame(
  users   = paste0("user", sample(0:4, 50, replace = TRUE)),
  items   = paste0("item", sample(0:9, 50, replace = TRUE)),
  ratings = sample(1:5, 50, replace = TRUE)
)
```

### Model Configuration

```r
model <- mmsbm(
  user_groups = 2,     # Number of user groups
  item_groups = 4,     # Number of item groups
  iterations  = 500,   # Number of EM iterations
  sampling    = 5,     # Number of parallel EM runs; the best run is kept
  seed        = 1      # Random seed for reproducibility
)
```

> **Note on `sampling`**: Setting `sampling` to a value greater than 1 launches that many independent EM optimizations in parallel, each starting from a different random initialization. Once all runs finish, the one with the highest training likelihood is selected. This increases the chances of finding a better solution at the cost of extra computation time.

### Parallel Backend

Before fitting, set up the parallel backend. By default, computations run sequentially. To enable parallelism:

```r
library(future)
plan(multisession)  # use multiple R sessions

# Now fit the model -- sampling runs will execute in parallel
model <- fit(model, train)

plan(sequential) # Go back to sequential after fitting
```

### Training Methods

#### Simple Fit

```r
model <- fit(model, train)
```

#### Cross-Validation Fit

```r
model <- cv_fit(model, train, folds = 5)
model$cv_results
#> # A tibble: 5 × 4
#>    fold accuracy one_off_accuracy   mae
#>   <int>    <dbl>            <dbl> <dbl>
#> 1     1    0.25             0.65  0.75
#> ...
```

### Making Predictions

`predict()` returns a tibble following tidymodels conventions:

```r
# Predicted rating classes
predict(model, test)
#> # A tibble: 50 × 1
#>   .pred_class
#>   <chr>
#> 1 3
#> 2 5
#> ...

# Rating probability distributions
predict(model, test, type = "prob")
#> # A tibble: 50 × 5
#>   .pred_1 .pred_2 .pred_3 .pred_4 .pred_5
#>     <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
#> 1  0.102   0.153   0.398   0.245   0.102
#> ...
```

### Augmenting Data

`augment()` appends predictions to the original data:

```r
augment(model, test)
#> # A tibble: 50 × 8
#>   users items ratings .pred_class .pred_1 .pred_2 .pred_3 .pred_4 .pred_5
#>   <chr> <chr>   <int> <chr>         <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
#> 1 user2 item3       4 3             0.102   0.153   0.398   0.245   0.102
#> ...
```

### Model Evaluation

`metrics()` returns a yardstick-style tibble:

```r
metrics(model, test)
#> # A tibble: 5 × 3
#>   .metric          .estimator .estimate
#>   <chr>            <chr>          <dbl>
#> 1 accuracy         multiclass     0.35
#> 2 one_off_accuracy multiclass     0.72
#> 3 mae              standard       0.65
#> 4 s2               standard      45
#> 5 s2pond           standard      32.1
```

### Unseen Users and Items

MMSBM is a bipartite graph model: it learns latent group memberships for each user (theta) and each item (eta) from the observed interactions in the training data. This means that **predictions are only possible for users and items that appeared during training**. If `predict()`, `augment()`, or `metrics()` encounter unseen users or items, the affected rows are dropped with a warning:

```r
# If test data contains "user_new" not present in train:
predict(model, test)
#> Warning: Dropping 3 observation(s) with unseen users not present in the
#> training set: user_new. The model has no learned group memberships (theta)
#> for unseen users, because MMSBM estimates memberships from observed
#> interactions only. To predict for new users, include them in the training
#> data and refit the model.
```

The same applies to unseen items (no learned eta) and unseen rating levels (not covered by the learned probability tensor). To avoid dropped rows, ensure that all users, items, and rating levels in the test set also appear in the training set.

### Model Parameters

After fitting, model parameters are available directly:

```r
model$theta  # User group memberships
model$eta    # Item group memberships
model$pr     # Rating probabilities
```

## Tidy

You can go all the way tidy:

```r
metrics_df <- mmsbm(
  user_groups = 2,
  item_groups = 4,
  iterations  = 500,
  sampling    = 5,
  seed        = 1
) |>
  fit(train) |>
  metrics(test)
```

## Running Tests

```r
devtools::test()
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
