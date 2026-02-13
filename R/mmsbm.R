# Main MMSBM interface: S3 constructor and generic methods.
#
# The mmsbm object is a list with a class attribute. It carries configuration,
# encoding state, and (after fitting) the trained model parameters.


# ── S3 generics for non-base methods ──────────────────────────────────────

#' Fit a model
#' @param model A model object.
#' @param ... Additional arguments.
#' @export
fit <- function(model, ...) UseMethod("fit")

#' Compute model performance metrics
#' @param model A fitted model object.
#' @param ... Additional arguments.
#' @export
metrics <- function(model, ...) UseMethod("metrics")

#' Cross-validated model fitting
#' @param model A model object.
#' @param ... Additional arguments.
#' @export
cv_fit <- function(model, ...) UseMethod("cv_fit")

#' Augment data with predictions
#' @param x A fitted model object.
#' @param ... Additional arguments.
#' @export
augment <- function(x, ...) UseMethod("augment")


#' Create a Mixed Membership Stochastic Block Model
#'
#' @param user_groups Integer: number of user groups.
#' @param item_groups Integer: number of item groups.
#' @param iterations Integer: EM iterations per sampling run. Default 400.
#' @param sampling Integer: number of parallel EM runs. Default 1.
#' @param seed Integer or NULL: seed for reproducibility.
#' @param debug Logical: verbose output and reduced iterations. Default FALSE.
#' @return An S3 object of class `"mmsbm"`.
#' @export
mmsbm <- function(user_groups,
                  item_groups,
                  iterations = 400L,
                  sampling   = 1L,
                  seed       = NULL,
                  debug      = FALSE) {
  obj <- list(
    user_groups = as.integer(user_groups),
    item_groups = as.integer(item_groups),
    iterations  = as.integer(iterations),
    sampling    = as.integer(sampling),
    seed        = seed,
    debug       = debug,
    # Populated during fit
    encoding     = NULL,
    train        = NULL,
    ratings      = NULL,
    results      = NULL,
    norm_factors = NULL,
    best_run     = NULL,
    theta        = NULL,
    eta          = NULL,
    pr           = NULL,
    likelihood   = NULL,
    # Populated during cv_fit
    cv_results   = NULL
  )
  class(obj) <- "mmsbm"
  obj
}


#' @export
print.mmsbm <- function(x, ...) {
  fitted <- if (!is.null(x$results)) "fitted" else "unfitted"
  cat(sprintf(
    "MMSBM model (%s): %d user groups, %d item groups, %d iterations, %d sampling runs\n",
    fitted, x$user_groups, x$item_groups, x$iterations, x$sampling
  ))
  if (!is.null(x$likelihood)) {
    cat(sprintf("  Training likelihood: %s\n", format(x$likelihood, digits = 4)))
  }
  if (!is.null(x$cv_results)) {
    cat(sprintf(
      "  CV accuracy: %s +/- %s (%d folds)\n",
      format(mean(x$cv_results$accuracy), digits = 4),
      format(stats::sd(x$cv_results$accuracy), digits = 4),
      nrow(x$cv_results)
    ))
  }
  invisible(x)
}


# ── Internal: prepare objects after data encoding ─────────────────────────

prepare_objects <- function(model, train) {
  ratings <- sort(unique(train[, 3L]))
  n_users <- max(train[, 1L])
  n_items <- max(train[, 2L])

  # Pre-compute normalization factors:
  # For each user u, norm_factors$user[u, ] = rep(degree_u, K)
  user_degrees <- tabulate(train[, 1L], nbins = n_users)
  user_degrees <- pmax(user_degrees, 1L)
  norm_user <- matrix(
    rep(user_degrees, model$user_groups),
    nrow = n_users, ncol = model$user_groups
  )

  item_degrees <- tabulate(train[, 2L], nbins = n_items)
  item_degrees <- pmax(item_degrees, 1L)
  norm_item <- matrix(
    rep(item_degrees, model$item_groups),
    nrow = n_items, ncol = model$item_groups
  )

  model$train    <- train
  model$ratings  <- ratings
  model$norm_factors <- list(user = norm_user, item = norm_item)
  model$config <- list(
    user_groups  = model$user_groups,
    item_groups  = model$item_groups,
    iterations   = model$iterations,
    n_users      = n_users,
    n_items      = n_items,
    n_ratings    = length(ratings),
    norm_factors = model$norm_factors,
    debug        = model$debug
  )

  model
}


# ── fit ───────────────────────────────────────────────────────────────────

#' Fit the MMSBM model
#'
#' Runs multiple parallel EM optimisation runs. The best run is selected by
#' training likelihood and its parameters (theta, eta, pr) are stored on the
#' model for use by [predict()], [augment()], and [metrics()].
#'
#' @param model An `mmsbm` object.
#' @param data A data.frame with three columns: users, items, ratings.
#' @param silent Logical: suppress progress output. Default FALSE.
#' @param ... Ignored (for S3 generic compatibility).
#' @return The fitted `mmsbm` object.
#' @export
fit.mmsbm <- function(model, data, silent = FALSE, ...) {
  if (!silent) {
    n_workers <- future::nbrOfWorkers()
    message(sprintf(
      "Running %d runs of %d iterations (%d worker(s)).",
      model$sampling, model$iterations, n_workers
    ))
  }

  # Encode data
  model$encoding <- create_encoding()
  train <- format_train_data(data, model$encoding)
  model <- prepare_objects(model, train)

  # Generate independent seeds for each sampling run
  if (!is.null(model$seed)) set.seed(model$seed)
  seeds <- sample.int(.Machine$integer.max, model$sampling)

  config <- model$config

  # Parallel execution with progress reporting
  progressr::handlers(progressr::handler_txtprogressbar())
  model$results <- progressr::with_progress({
    p <- progressr::progressor(steps = model$sampling * config$iterations)
    future.apply::future_lapply(
      seq_len(model$sampling),
      function(i) {
        run_one_sampling(train, config, seed = seeds[i], index = i, p = p)
      },
      future.seed = TRUE
    )
  }, enable = !silent)

  # Select the best run by training likelihood
  likelihoods <- vapply(model$results, function(r) r$likelihood, numeric(1L))
  model$best_run  <- which.max(likelihoods)
  best <- model$results[[model$best_run]]

  # Store labeled parameters from the best run
  model$theta      <- return_theta_indices(best$theta, model$encoding)
  model$eta        <- return_eta_indices(best$eta, model$encoding)
  model$pr         <- return_pr_indices(best$pr, model$encoding)
  model$likelihood <- best$likelihood

  model
}


# ── predict ───────────────────────────────────────────────────────────────

#' Predict ratings for new user-item pairs
#'
#' Returns a tibble of predictions following tidymodels conventions.
#' Use `type = "class"` (default) for predicted rating classes or
#' `type = "prob"` for rating probability distributions.
#'
#' @param object A fitted `mmsbm` object.
#' @param new_data A data.frame with three columns: users, items, ratings.
#' @param type Character: `"class"` for predicted ratings, `"prob"` for
#'   probability distributions over rating levels.
#' @param ... Ignored (for S3 generic compatibility).
#' @return A tibble. For `type = "class"`: a single `.pred_class` column.
#'   For `type = "prob"`: one `.pred_{level}` column per rating level.
#' @export
predict.mmsbm <- function(object, new_data, type = "class", ...) {
  if (is.null(object$results)) {
    stop("You need to fit the model before predicting.")
  }

  test <- format_test_data(new_data, object$encoding)
  best <- object$results[[object$best_run]]
  rat  <- prod_dist(test, best$theta, best$eta, best$pr)

  inv_ratings <- invert_named_int(object$encoding$ratings_dict)

  if (type == "class") {
    pred_idx <- max.col(rat, ties.method = "first")
    tibble::tibble(.pred_class = unname(inv_ratings[pred_idx]))
  } else if (type == "prob") {
    prob_df <- tibble::as_tibble(as.data.frame(rat))
    names(prob_df) <- paste0(".pred_", inv_ratings[seq_len(ncol(rat))])
    prob_df
  } else {
    stop("type must be 'class' or 'prob'.")
  }
}


# ── augment ───────────────────────────────────────────────────────────────

#' Augment data with model predictions
#'
#' Returns the original data with prediction columns appended:
#' `.pred_class` and one `.pred_{level}` column per rating level.
#' Rows with unseen users, items, or ratings are dropped (with a warning).
#'
#' @param x A fitted `mmsbm` object.
#' @param new_data A data.frame with three columns: users, items, ratings.
#' @param ... Ignored (for S3 generic compatibility).
#' @return A tibble: the encodable rows of `new_data` with prediction columns.
#' @export
augment.mmsbm <- function(x, new_data, ...) {
  if (is.null(x$results)) {
    stop("You need to fit the model before augmenting.")
  }

  new_data_tbl <- tibble::as_tibble(new_data)

  # Identify and filter rows with unseen values (warns once here)
  keep <- find_encodable_rows(new_data_tbl, x$encoding)
  filtered <- new_data_tbl[keep, , drop = FALSE]

  # Encode and predict in a single pass (no further filtering/warnings)
  test <- suppressWarnings(format_test_data(filtered, x$encoding))
  best <- x$results[[x$best_run]]
  rat  <- prod_dist(test, best$theta, best$eta, best$pr)

  inv_ratings <- invert_named_int(x$encoding$ratings_dict)

  # Build class predictions
  pred_idx  <- max.col(rat, ties.method = "first")
  class_col <- tibble::tibble(.pred_class = unname(inv_ratings[pred_idx]))

  # Build probability predictions
  prob_df <- tibble::as_tibble(as.data.frame(rat))
  names(prob_df) <- paste0(".pred_", inv_ratings[seq_len(ncol(rat))])

  dplyr::bind_cols(filtered, class_col, prob_df)
}


# ── metrics ───────────────────────────────────────────────────────────────

#' Compute model performance metrics
#'
#' Evaluates model predictions against observed ratings. Returns a
#' yardstick-style tibble with `.metric`, `.estimator`, and `.estimate`
#' columns.
#'
#' @param model A fitted `mmsbm` object.
#' @param new_data A data.frame with three columns: users, items, ratings.
#' @param ... Ignored (for S3 generic compatibility).
#' @return A tibble with columns `.metric`, `.estimator`, `.estimate`.
#' @export
metrics.mmsbm <- function(model, new_data, ...) {
  if (is.null(model$results)) {
    stop("You need to fit the model before computing metrics.")
  }

  test <- format_test_data(new_data, model$encoding)
  best <- model$results[[model$best_run]]
  rat  <- prod_dist(test, best$theta, best$eta, best$pr)
  stats <- compute_stats(rat, test, model$ratings)

  tibble::tibble(
    .metric    = c("accuracy", "one_off_accuracy", "mae", "s2", "s2pond"),
    .estimator = c("multiclass", "multiclass", "standard", "standard", "standard"),
    .estimate  = c(stats$accuracy, stats$one_off_accuracy, stats$mae,
                   stats$s2, stats$s2pond)
  )
}


# ── cv_fit ────────────────────────────────────────────────────────────────

#' Cross-validated model fitting
#'
#' Splits data into k folds, fits the model on each training split, and
#' evaluates on the held-out fold. Returns the model with per-fold metrics
#' in `model$cv_results` and the best fold's parameters stored on the model.
#'
#' @param model An `mmsbm` object.
#' @param data A data.frame with three columns: users, items, ratings.
#' @param folds Integer: number of folds. Default 5.
#' @param ... Ignored (for S3 generic compatibility).
#' @return The `mmsbm` object with `cv_results` (a tibble of per-fold metrics)
#'   and the best fold's parameters (theta, eta, pr).
#' @export
cv_fit.mmsbm <- function(model, data, folds = 5L, ...) {
  items_per_fold <- structure_folds(data, folds)

  temp <- data
  fold_stats <- vector("list", folds)
  fold_params <- vector("list", folds)

  for (f in seq_len(folds)) {
    message(sprintf("Running fold %d of %d...", f, folds))

    # Sample test indices: for each user, pick up to items_per_fold rows
    user_groups <- split(seq_len(nrow(temp)), temp[[1L]])
    test_idx <- unlist(lapply(user_groups, get_n_per_group, n = items_per_fold))
    test_idx <- test_idx[test_idx > 0L]

    test_data  <- temp[test_idx, , drop = FALSE]
    train_data <- data[!rownames(data) %in% rownames(test_data), , drop = FALSE]
    temp       <- temp[-test_idx, , drop = FALSE]

    # Fit on the training split
    model_fold <- fit.mmsbm(model, train_data, silent = TRUE)

    # Evaluate on the test split using internal functions
    test_enc <- format_test_data(test_data, model_fold$encoding)
    best <- model_fold$results[[model_fold$best_run]]
    rat  <- prod_dist(test_enc, best$theta, best$eta, best$pr)
    stats <- compute_stats(rat, test_enc, model_fold$ratings)

    fold_stats[[f]] <- stats
    fold_params[[f]] <- list(
      theta      = model_fold$theta,
      eta        = model_fold$eta,
      pr         = model_fold$pr,
      likelihood = model_fold$likelihood,
      results    = model_fold$results,
      best_run   = model_fold$best_run,
      encoding   = model_fold$encoding,
      train      = model_fold$train,
      ratings    = model_fold$ratings
    )
  }

  # Build cv_results tibble
  accs      <- vapply(fold_stats, function(s) s$accuracy, numeric(1L))
  one_offs  <- vapply(fold_stats, function(s) s$one_off_accuracy, numeric(1L))
  maes      <- vapply(fold_stats, function(s) s$mae, numeric(1L))

  model$cv_results <- tibble::tibble(
    fold             = seq_len(folds),
    accuracy         = accs,
    one_off_accuracy = one_offs,
    mae              = maes
  )

  # Store the best fold's parameters on the model
  best_fold <- which.max(accs)
  model$theta      <- fold_params[[best_fold]]$theta
  model$eta        <- fold_params[[best_fold]]$eta
  model$pr         <- fold_params[[best_fold]]$pr
  model$likelihood <- fold_params[[best_fold]]$likelihood
  model$results    <- fold_params[[best_fold]]$results
  model$best_run   <- fold_params[[best_fold]]$best_run
  model$encoding   <- fold_params[[best_fold]]$encoding
  model$train      <- fold_params[[best_fold]]$train
  model$ratings    <- fold_params[[best_fold]]$ratings

  message(sprintf(
    "Ran %d folds with accuracies %s.",
    folds, paste(format(accs, digits = 4), collapse = ", ")
  ))
  message(sprintf(
    "Mean accuracy: %s +/- %s.",
    format(mean(accs), digits = 4),
    format(stats::sd(accs), digits = 4)
  ))

  model
}


# ── Internal helpers ──────────────────────────────────────────────────────

#' Compute all evaluation statistics from a prediction matrix
#' @keywords internal
compute_stats <- function(rat, test, ratings) {
  indicators <- compute_indicators(rat, test, ratings)
  compute_final_stats(indicators)
}


#' Compute boolean/error indicators for evaluation
#' @keywords internal
compute_indicators <- function(rat, test, ratings) {
  # Predicted = most probable rating index (1-based)
  pred <- max.col(rat, ties.method = "first")
  real <- test[, 3L]

  # Filter observations without predictions (all-zero probability rows)
  mask <- rowSums(rat) != 0
  if (!all(mask)) {
    pred <- pred[mask]
    real <- real[mask]
    rat  <- rat[mask, , drop = FALSE]
  }

  true_hit <- as.integer(pred == real)
  almost   <- as.integer(abs(pred - real) <= 1L)
  s2       <- abs(pred - real)

  # Weighted prediction using full probability distribution
  pred_pond  <- as.numeric(rat %*% ratings)
  true_pond  <- as.integer(real == round(pred_pond))
  s2pond     <- abs(pred_pond - real)

  list(
    true   = true_hit,
    almost = almost,
    s2     = s2,
    true_pond = true_pond,
    s2pond    = s2pond
  )
}


#' Aggregate indicators into final statistics
#' @keywords internal
compute_final_stats <- function(ind) {
  n <- length(ind$true)
  list(
    accuracy         = sum(ind$true) / n,
    one_off_accuracy = sum(ind$almost) / n,
    mae              = 1 - sum(ind$true_pond) / n,
    s2               = sum(ind$s2),
    s2pond           = sum(ind$s2pond)
  )
}
