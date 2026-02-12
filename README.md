# Mixed Membership Stochastic Block Models for R

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

An R implementation of Mixed Membership Stochastic Block Models (MMSBM) for recommendation systems, based on the work by Godoy-Lorite et al. (2016). This package provides an efficient, vectorized implementation using base R array operations for the hot path and parallel processing via the [future](https://future.futureverse.org/) framework.

This is an R port of the [Python mmsbm package](https://github.com/eudald-seeslab/mmsbm).

## Features

- Fast, vectorized EM implementation using base R arrays (`rowsum()`, matrix multiplication).
- Support for both simple and cross-validated fitting.
- Parallel processing for multiple sampling runs via [future](https://future.futureverse.org/) / [future.apply](https://future.apply.futureverse.org/).
- Progress bars via [progressr](https://progressr.futureverse.org/).
- Comprehensive model statistics and evaluation metrics.

## Installation

Install directly from GitHub:

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

> **Note on `sampling`**: Setting `sampling` to a value greater than 1 launches that many independent EM optimizations in parallel, each starting from a different random initialization. Once all runs finish, the one with the highest accuracy is selected. This increases the chances of finding a better solution at the cost of extra computation time.

### Parallel Backend

Before fitting, set up the parallel backend. By default, computations run sequentially. To enable parallelism:

```r
library(future)
plan(multisession)  # use multiple R sessions

# Now fit the model -- sampling runs will execute in parallel
model <- fit(model, train)
```

### Training Methods

#### Simple Fit

```r
model <- fit(model, train)
```

#### Cross-Validation Fit

```r
accuracies <- cv_fit(model, train, folds = 5)
cat(sprintf("Mean accuracy: %.3f ± %.3f\n", mean(accuracies), sd(accuracies)))
```

### Making Predictions

```r
model <- predict(model, test)
```

### Model Evaluation

> **Note**: you need to call `predict()` before calling `score()`.

```r
results <- score(model)

# Access various metrics
results$stats$accuracy
results$stats$mae

# Access model parameters
results$objects$theta  # User group memberships
results$objects$eta    # Item group memberships
results$objects$pr     # Rating probabilities
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
