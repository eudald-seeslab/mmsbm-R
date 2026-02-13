# Tests for the main MMSBM model.


# ── fit ───────────────────────────────────────────────────────────────────

test_that("fit selects best run and stores labeled parameters", {
  future::plan(future::sequential)
  mm <- fit_model()

  expect_true(!is.null(mm$results))
  expect_true(!is.null(mm$best_run))
  expect_true(!is.null(mm$theta))
  expect_true(!is.null(mm$eta))
  expect_true(!is.null(mm$pr))
  expect_true(!is.null(mm$likelihood))
  expect_true(is.finite(mm$likelihood))
})


test_that("fit does not modify the original model (copy-on-modify)", {
  future::plan(future::sequential)
  original <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)

  # Capture the original state
  expect_null(original$results)
  expect_null(original$theta)
  expect_null(original$best_run)

  fitted <- fit(original, mock_data(1L), silent = TRUE)

  # Original must be unchanged

  expect_null(original$results)
  expect_null(original$theta)
  expect_null(original$best_run)
  expect_null(original$likelihood)

  # Fitted must have the new state
  expect_true(!is.null(fitted$results))
  expect_true(!is.null(fitted$theta))
})


test_that("fit with multiple sampling runs selects the best", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, sampling = 3L, seed = 42L)
  mm <- fit(mm, mock_data(1L), silent = TRUE)

  expect_length(mm$results, 3L)
  expect_true(mm$best_run >= 1L && mm$best_run <= 3L)

  # best_run should correspond to the highest training likelihood
  likelihoods <- vapply(mm$results, function(r) r$likelihood, numeric(1L))
  expect_equal(mm$best_run, which.max(likelihoods))
  expect_equal(mm$likelihood, max(likelihoods))
})


test_that("re-fitting a model overwrites previous state cleanly", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  mm <- fit(mm, mock_data(1L), silent = TRUE)

  theta_first <- mm$theta
  likelihood_first <- mm$likelihood

  # Fit again with different data
  mm <- fit(mm, mock_data(5L), silent = TRUE)

  # State should be fully replaced (different data -> likely different params)
  expect_true(!is.null(mm$theta))
  expect_true(!is.null(mm$likelihood))
  expect_true(is.finite(mm$likelihood))
})


# ── predict ───────────────────────────────────────────────────────────────

test_that("predict returns tibble with .pred_class", {
  future::plan(future::sequential)
  mm <- fit_model()
  preds <- predict(mm, mock_data(2L), type = "class")

  expect_s3_class(preds, "tbl_df")
  expect_true(".pred_class" %in% names(preds))
  expect_equal(nrow(preds), RATING_NUM)
})


test_that("predict returns tibble with probability columns", {
  future::plan(future::sequential)
  mm <- fit_model()
  probs <- predict(mm, mock_data(2L), type = "prob")

  expect_s3_class(probs, "tbl_df")
  # All columns should start with .pred_
  expect_true(all(grepl("^\\.pred_", names(probs))))
  expect_equal(nrow(probs), RATING_NUM)

  # Each row should sum to ~1 (probability distribution over ratings)
  row_sums <- rowSums(as.matrix(probs))
  expect_equal(row_sums, rep(1, RATING_NUM), tolerance = 0.01)
})


test_that("predict class and prob are consistent", {
  future::plan(future::sequential)
  mm <- fit_model()
  test_data <- mock_data(2L)

  classes <- predict(mm, test_data, type = "class")
  probs   <- predict(mm, test_data, type = "prob")

  # The class prediction should be the rating with the highest probability
  prob_mat <- as.matrix(probs)
  inv_ratings <- invert_named_int(mm$encoding$ratings_dict)
  expected_class <- unname(inv_ratings[max.col(prob_mat, ties.method = "first")])

  expect_equal(classes$.pred_class, expected_class)
})


test_that("predict without fit raises error", {
  mm <- mmsbm(2L, 2L, seed = 1L)
  expect_error(predict(mm, mock_data(0L)), "fit the model before predicting")
})


test_that("predict with invalid type raises error", {
  future::plan(future::sequential)
  mm <- fit_model()
  expect_error(predict(mm, mock_data(2L), type = "bad"), "type must be")
})


test_that("predict with unseen test values drops rows with warning", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  train_data <- data.frame(
    users   = c("u1", "u2", "u1", "u2"),
    items   = c("i1", "i2", "i2", "i1"),
    ratings = c(1L, 2L, 1L, 2L),
    stringsAsFactors = FALSE
  )
  mm <- fit(mm, train_data, silent = TRUE)

  test_with_unseen <- data.frame(
    users   = c("u1", "u_unknown", "u2"),
    items   = c("i1", "i1",        "i2"),
    ratings = c(1L,   1L,          2L),
    stringsAsFactors = FALSE
  )

  expect_warning(preds <- predict(mm, test_with_unseen), "unseen users.*u_unknown")
  # The unseen row should be dropped
  expect_equal(nrow(preds), 2L)
})


# ── augment ───────────────────────────────────────────────────────────────

test_that("augment returns new_data with prediction columns", {
  future::plan(future::sequential)
  mm <- fit_model()
  test_data <- mock_data(2L)
  augmented <- augment(mm, test_data)

  expect_s3_class(augmented, "tbl_df")
  expect_equal(nrow(augmented), nrow(test_data))
  # Should have the original columns plus .pred_class and .pred_* columns
  expect_true(".pred_class" %in% names(augmented))
  expect_true(all(names(test_data) %in% names(augmented)))
  prob_cols <- grep("^\\.pred_[^c]", names(augmented), value = TRUE)
  expect_true(length(prob_cols) > 0)
})


test_that("augment without fit raises error", {
  mm <- mmsbm(2L, 2L, seed = 1L)
  expect_error(augment(mm, mock_data(0L)), "fit the model before augmenting")
})


test_that("augment handles unseen test values without crashing", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  train_data <- data.frame(
    users   = c("u1", "u2", "u1", "u2"),
    items   = c("i1", "i2", "i2", "i1"),
    ratings = c(1L, 2L, 1L, 2L),
    stringsAsFactors = FALSE
  )
  mm <- fit(mm, train_data, silent = TRUE)

  test_with_unseen <- data.frame(
    users   = c("u1", "u_unknown", "u2"),
    items   = c("i1", "i1",        "i2"),
    ratings = c(1L,   1L,          2L),
    stringsAsFactors = FALSE
  )

  # Should warn about unseen values, not crash
  expect_warning(aug <- augment(mm, test_with_unseen), "unseen users.*u_unknown")
  # The unseen row is dropped; remaining rows should have all columns
  expect_equal(nrow(aug), 2L)
  expect_true(".pred_class" %in% names(aug))
  expect_true("users" %in% names(aug))
})


test_that("augment predictions match predict output", {
  future::plan(future::sequential)
  mm <- fit_model()
  test_data <- mock_data(2L)

  augmented <- augment(mm, test_data)
  classes   <- predict(mm, test_data, type = "class")
  probs     <- predict(mm, test_data, type = "prob")

  expect_equal(augmented$.pred_class, classes$.pred_class)
  prob_cols <- grep("^\\.pred_[^c]", names(augmented), value = TRUE)
  for (col in prob_cols) {
    expect_equal(augmented[[col]], probs[[col]])
  }
})


# ── metrics ───────────────────────────────────────────────────────────────

test_that("metrics returns yardstick-style tibble", {
  future::plan(future::sequential)
  mm <- fit_model()
  result <- metrics(mm, mock_data(2L))

  expect_s3_class(result, "tbl_df")
  expect_named(result, c(".metric", ".estimator", ".estimate"))
  expect_true("accuracy" %in% result$.metric)
  expect_true("one_off_accuracy" %in% result$.metric)
  expect_true("mae" %in% result$.metric)
  expect_true("s2" %in% result$.metric)
  expect_true("s2pond" %in% result$.metric)

  # Accuracy should be between 0 and 1
  acc <- result$.estimate[result$.metric == "accuracy"]
  expect_true(acc >= 0 && acc <= 1)
})


test_that("metrics without fit raises error", {
  mm <- mmsbm(2L, 2L, seed = 1L)
  expect_error(metrics(mm, mock_data(0L)), "fit the model before computing")
})


# ── cv_fit ────────────────────────────────────────────────────────────────

test_that("cv_fit returns model with cv_results tibble", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  result <- cv_fit(mm, mock_data(1L), folds = 2L)

  # Returns the model, not a vector
  expect_s3_class(result, "mmsbm")
  expect_true(!is.null(result$cv_results))
  expect_s3_class(result$cv_results, "tbl_df")
  expect_equal(nrow(result$cv_results), 2L)
  expect_true("accuracy" %in% names(result$cv_results))
  expect_true("one_off_accuracy" %in% names(result$cv_results))
  expect_true("mae" %in% names(result$cv_results))
  expect_true(all(result$cv_results$accuracy >= 0 & result$cv_results$accuracy <= 1))

  # Model params from best fold should be stored
  expect_true(!is.null(result$theta))
  expect_true(!is.null(result$eta))
  expect_true(!is.null(result$pr))
})


test_that("cv_fit does not modify the original model (copy-on-modify)", {
  future::plan(future::sequential)
  original <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)

  expect_null(original$cv_results)
  expect_null(original$theta)

  result <- cv_fit(original, mock_data(1L), folds = 2L)

  # Original must be unchanged
  expect_null(original$cv_results)
  expect_null(original$theta)
  expect_null(original$results)

  # Result must have the new state
  expect_true(!is.null(result$cv_results))
  expect_true(!is.null(result$theta))
})


test_that("cv_fit model can be used for predict and metrics", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  mm <- cv_fit(mm, mock_data(1L), folds = 2L)

  # Should be able to predict on new data
  test_data <- mock_data(3L, n = 20L)
  preds <- predict(mm, test_data, type = "class")
  expect_s3_class(preds, "tbl_df")
  expect_true(nrow(preds) > 0)

  # Should be able to compute metrics on new data
  met <- metrics(mm, test_data)
  expect_s3_class(met, "tbl_df")
  expect_true("accuracy" %in% met$.metric)

  # Should be able to augment
  aug <- augment(mm, test_data)
  expect_s3_class(aug, "tbl_df")
  expect_true(".pred_class" %in% names(aug))
})


# ── pipe chain ────────────────────────────────────────────────────────────

test_that("pipe chain fit |> metrics works end-to-end", {
  future::plan(future::sequential)
  result <- mmsbm(2L, 2L, iterations = 10L, seed = 1L) |>
    fit(mock_data(1L), silent = TRUE) |>
    metrics(mock_data(2L))

  expect_s3_class(result, "tbl_df")
  expect_true("accuracy" %in% result$.metric)
})


test_that("pipe chain fit |> predict works end-to-end", {
  future::plan(future::sequential)
  preds <- mmsbm(2L, 2L, iterations = 10L, seed = 1L) |>
    fit(mock_data(1L), silent = TRUE) |>
    predict(mock_data(2L), type = "class")

  expect_s3_class(preds, "tbl_df")
  expect_true(".pred_class" %in% names(preds))
})


test_that("pipe chain fit |> augment works end-to-end", {
  future::plan(future::sequential)
  aug <- mmsbm(2L, 2L, iterations = 10L, seed = 1L) |>
    fit(mock_data(1L), silent = TRUE) |>
    augment(mock_data(2L))

  expect_s3_class(aug, "tbl_df")
  expect_true(".pred_class" %in% names(aug))
  expect_true("users" %in% names(aug))
})


# ── model parameters ─────────────────────────────────────────────────────

test_that("model parameters have correct types after fit", {
  future::plan(future::sequential)
  mm <- fit_model()

  # theta and eta should be tibbles with a label column
  expect_s3_class(mm$theta, "tbl_df")
  expect_s3_class(mm$eta, "tbl_df")
  expect_true("user" %in% names(mm$theta))
  expect_true("item" %in% names(mm$eta))

  # pr should be a named list of tibbles
  expect_true(is.list(mm$pr))
  expect_true(length(mm$pr) > 0)
  for (tbl in mm$pr) {
    expect_s3_class(tbl, "tbl_df")
  }

  # likelihood should be a single finite number
  expect_true(is.numeric(mm$likelihood))
  expect_length(mm$likelihood, 1L)
  expect_true(is.finite(mm$likelihood))
})


# ── print ─────────────────────────────────────────────────────────────────

test_that("print works for unfitted model", {
  mm <- mmsbm(2L, 4L)
  expect_output(print(mm), "unfitted")
  expect_output(print(mm), "2 user groups")
  expect_output(print(mm), "4 item groups")
})


test_that("print works for fitted model", {
  future::plan(future::sequential)
  mm <- fit_model()
  expect_output(print(mm), "fitted")
  expect_output(print(mm), "Training likelihood")
})


test_that("print works for cv_fit model", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  mm <- cv_fit(mm, mock_data(1L), folds = 2L)
  expect_output(print(mm), "CV accuracy")
})


# ── internal helpers ──────────────────────────────────────────────────────

test_that("compute_likelihood returns a finite value", {
  future::plan(future::sequential)
  data_df <- mock_data(1L, n = 10L)

  mm <- mmsbm(2L, 2L, iterations = 1L, sampling = 1L, seed = 1L)
  mm$encoding <- create_encoding()
  train <- format_train_data(data_df, mm$encoding)
  mm <- prepare_objects(mm, train)

  set.seed(0L)
  theta <- matrix(runif(mm$config$n_users * mm$config$user_groups),
                  nrow = mm$config$n_users)
  theta <- theta / rowSums(theta)
  eta <- matrix(runif(mm$config$n_items * mm$config$item_groups),
                nrow = mm$config$n_items)
  eta <- eta / rowSums(eta)
  pr <- array(runif(mm$config$user_groups * mm$config$item_groups * mm$config$n_ratings),
              dim = c(mm$config$user_groups, mm$config$item_groups, mm$config$n_ratings))
  pr <- normalize_with_self(pr)

  ll <- compute_likelihood(train, theta, eta, pr)
  expect_true(is.finite(ll))
})


test_that("run_one_sampling returns expected keys and shapes", {
  future::plan(future::sequential)
  data_df <- mock_data(2L, n = 15L)

  mm <- mmsbm(2L, 2L, iterations = 1L, sampling = 1L, seed = 2L)
  mm$encoding <- create_encoding()
  train <- format_train_data(data_df, mm$encoding)
  mm <- prepare_objects(mm, train)

  result <- run_one_sampling(train, mm$config, seed = 123L, index = 1L)

  expect_true(all(c("likelihood", "pr", "theta", "eta") %in% names(result)))
  expect_equal(dim(result$pr)[1L], mm$config$user_groups)
  expect_equal(dim(result$pr)[2L], mm$config$item_groups)
})
