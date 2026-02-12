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

#' Score a fitted model
#' @param model A fitted model object.
#' @param ... Additional arguments.
#' @export
score <- function(model, ...) UseMethod("score")

# Ensure S3 methods have ... for generic consistency
# (the actual named arguments are listed before ...)


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
    encoding          = NULL,
    train             = NULL,
    ratings           = NULL,
    results           = NULL,
    norm_factors      = NULL,
    # Populated during predict
    test              = NULL,
    theta             = NULL,
    eta               = NULL,
    pr                = NULL,
    likelihood        = NULL,
    prediction_matrix = NULL,
    start_time        = Sys.time()
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
#' Runs multiple parallel EM optimisation runs and stores all results.
#'
#' @param model An `mmsbm` object.
#' @param data A data.frame with three columns: users, items, ratings.
#' @param silent Logical: suppress progress output. Default FALSE.
#' @param ... Ignored (for S3 generic compatibility).
#' @return The fitted `mmsbm` object (updated in place via reassignment).
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

  model
}


# ── predict ───────────────────────────────────────────────────────────────

#' Predict ratings for new user-item pairs
#'
#' @param object A fitted `mmsbm` object.
#' @param newdata A data.frame with three columns: users, items, ratings.
#' @param ... Ignored (for S3 generic compatibility).
#' @return A numeric matrix (N x R) of rating probability distributions.
#' @export
predict.mmsbm <- function(object, newdata, ...) {
  if (is.null(object$results)) {
    stop("You need to fit the model before predicting.")
  }

  test <- format_test_data(newdata, object$encoding)
  object$test <- test

  # Compute predictions for all runs
  rats <- lapply(object$results, function(res) {
    prod_dist(test, res$theta, res$eta, res$pr)
  })

  # Choose best run based on accuracy
  best <- choose_best_run(rats, test, object$ratings)

  # Store best-run parameters with original labels
  object$theta      <- return_theta_indices(object$results[[best]]$theta, object$encoding)
  object$eta        <- return_eta_indices(object$results[[best]]$eta, object$encoding)
  object$pr         <- return_pr_indices(object$results[[best]]$pr, object$encoding)
  object$likelihood <- object$results[[best]]$likelihood

  # Prediction matrix is the average across all runs
  rat_array <- array(unlist(rats), dim = c(nrow(rats[[1L]]), ncol(rats[[1L]]), length(rats)))
  object$prediction_matrix <- rowMeans(rat_array, dims = 2L)

  object
}


# ── score ─────────────────────────────────────────────────────────────────

#' Compute model performance metrics
#'
#' @param model A fitted `mmsbm` object (after `predict`).
#' @param silent Logical: suppress logging. Default FALSE.
#' @param ... Ignored (for S3 generic compatibility).
#' @return A list with `stats` (accuracy, one_off_accuracy, mae, s2, s2pond,
#'   likelihood) and `objects` (theta, eta, pr).
#' @export
score.mmsbm <- function(model, silent = FALSE, ...) {
  if (is.null(model$prediction_matrix)) {
    stop("You need to predict before computing the goodness of fit parameters.")
  }

  stats <- compute_stats(model$prediction_matrix, model$test, model$ratings)
  stats$likelihood <- model$likelihood

  if (!silent) {
    elapsed <- as.numeric(difftime(Sys.time(), model$start_time, units = "mins"))
    message(sprintf("Done %d runs in %.2f minutes.", model$sampling, elapsed))
    message(sprintf(
      "The final accuracy is %s, the one off accuracy is %s and the MAE is %s.",
      format(stats$accuracy, digits = 4),
      format(stats$one_off_accuracy, digits = 4),
      format(stats$mae, digits = 4)
    ))
  }

  list(
    stats   = stats,
    objects = list(theta = model$theta, eta = model$eta, pr = model$pr)
  )
}


# ── cv_fit ────────────────────────────────────────────────────────────────

#' Cross-validated model fitting
#'
#' Splits data into k folds and evaluates model performance on held-out data.
#'
#' @param model An `mmsbm` object.
#' @param data A data.frame with three columns: users, items, ratings.
#' @param folds Integer: number of folds. Default 5.
#' @return A numeric vector of accuracies, one per fold.
#' @export
cv_fit <- function(model, data, folds = 5L) {
  items_per_fold <- structure_folds(data, folds)

  temp <- data
  all_results <- vector("list", folds)

  for (f in seq_len(folds)) {
    message(sprintf("Running fold %d of %d...", f, folds))

    # Sample test indices: for each user, pick up to items_per_fold rows
    user_groups <- split(seq_len(nrow(temp)), temp[[1L]])
    test_idx <- unlist(lapply(user_groups, get_n_per_group, n = items_per_fold))
    test_idx <- test_idx[test_idx > 0L]

    test_data  <- temp[test_idx, , drop = FALSE]
    train_data <- data[!rownames(data) %in% rownames(test_data), , drop = FALSE]
    temp       <- temp[-test_idx, , drop = FALSE]

    # Fit and predict on this fold
    model <- fit.mmsbm(model, train_data, silent = TRUE)
    model <- predict.mmsbm(model, test_data)
    results <- score.mmsbm(model, silent = TRUE)

    all_results[[f]] <- list(
      stats   = results$stats,
      objects = list(
        theta = model$theta,
        eta   = model$eta,
        pr    = model$pr,
        rat   = model$prediction_matrix
      )
    )
  }

  # Pick best fold and store its parameters
  accuracies <- vapply(all_results, function(x) x$stats$accuracy, numeric(1L))
  best <- which.max(accuracies)
  model$theta             <- all_results[[best]]$objects$theta
  model$eta               <- all_results[[best]]$objects$eta
  model$pr                <- all_results[[best]]$objects$pr
  model$prediction_matrix <- all_results[[best]]$objects$rat

  message(sprintf("Ran %d folds with accuracies %s.", folds,
                  paste(format(accuracies, digits = 4), collapse = ", ")))
  message(sprintf("They have mean %s and sd %s.",
                  format(mean(accuracies), digits = 4),
                  format(stats::sd(accuracies), digits = 4)))

  accuracies
}


# ── Internal helpers ──────────────────────────────────────────────────────

#' Select the best run by accuracy
#' @keywords internal
choose_best_run <- function(rats, test, ratings) {
  accs <- vapply(rats, function(rat) {
    compute_stats(rat, test, ratings)$accuracy
  }, numeric(1L))
  which.max(accs)
}


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
