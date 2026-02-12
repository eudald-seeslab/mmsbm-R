# Tests for the main MMSBM model.
# These replicate the Python test_mmsbm.py test cases.

test_that("prediction matrix rows sum to approximately 1", {
  future::plan(future::sequential)
  mm <- fit_model()
  mm <- predict.mmsbm(mm, mock_data(2L))
  pred <- mm$prediction_matrix

  # Each row should sum to ~1 (probability distribution over ratings)
  expect_equal(sum(pred), RATING_NUM, tolerance = 0.01 * RATING_NUM)
})


test_that("fit and predict produce correctly shaped outputs", {
  future::plan(future::sequential)
  mm <- fit_model()
  mm <- predict.mmsbm(mm, mock_data(2L))

  expect_true(!is.null(mm$prediction_matrix))
  expect_true(!is.null(mm$theta))
  expect_true(!is.null(mm$eta))
  expect_true(!is.null(mm$pr))
  expect_true(!is.null(mm$likelihood))
})


test_that("score returns expected structure", {
  future::plan(future::sequential)
  mm <- fit_model()
  mm <- predict.mmsbm(mm, mock_data(2L))
  results <- score.mmsbm(mm, silent = TRUE)

  expect_named(results, c("stats", "objects"))
  expect_true("accuracy" %in% names(results$stats))
  expect_true("one_off_accuracy" %in% names(results$stats))
  expect_true("mae" %in% names(results$stats))
  expect_true("s2" %in% names(results$stats))
  expect_true("s2pond" %in% names(results$stats))
  expect_true("likelihood" %in% names(results$stats))
  expect_named(results$objects, c("theta", "eta", "pr"))
})


test_that("predict without fit raises error", {
  mm <- mmsbm(2L, 2L, seed = 1L)
  expect_error(predict.mmsbm(mm, mock_data(0L)), "fit the model before predicting")
})


test_that("score without predict raises error", {
  mm <- mmsbm(2L, 2L, seed = 1L)
  expect_error(score.mmsbm(mm), "predict before computing")
})


test_that("choose_best_run selects highest accuracy", {
  future::plan(future::sequential)
  mm <- fit_model()
  mm <- predict.mmsbm(mm, mock_data(2L))

  # Create fake prediction matrices with known quality
  # The one with perfect predictions should be selected
  n <- nrow(mm$test)
  R <- length(mm$ratings)

  # A "perfect" prediction matrix where argmax matches the true rating
  perfect <- matrix(0, nrow = n, ncol = R)
  for (i in seq_len(n)) {
    perfect[i, mm$test[i, 3L]] <- 1
  }
  # A "random" prediction
  random_pred <- matrix(1 / R, nrow = n, ncol = R)

  rats <- list(random_pred, perfect, random_pred)
  best <- choose_best_run(rats, mm$test, mm$ratings)
  expect_equal(best, 2L)
})


test_that("compute_likelihood returns a finite value", {
  future::plan(future::sequential)
  data_df <- mock_data(1L, n = 10L)

  mm <- mmsbm(2L, 2L, iterations = 1L, sampling = 1L, seed = 1L)
  mm$encoding <- create_encoding()
  train <- format_train_data(data_df, mm$encoding)
  mm <- prepare_objects(mm, train)

  K <- mm$config$n_users
  L <- mm$config$n_items
  R <- mm$config$n_ratings

  set.seed(0L)
  theta <- matrix(runif(mm$config$n_users * mm$config$user_groups),
                  nrow = mm$config$n_users)
  theta <- theta / rowSums(theta)
  eta <- matrix(runif(mm$config$n_items * mm$config$item_groups),
                nrow = mm$config$n_items)
  eta <- eta / rowSums(eta)
  pr <- array(runif(mm$config$user_groups * mm$config$item_groups * mm$config$n_ratings),
              dim = c(mm$config$user_groups, mm$config$item_groups, mm$config$n_ratings))
  # Normalise pr along last axis
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


test_that("score with logging executes without errors", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 1L, sampling = 1L, seed = 3L)
  mm <- fit.mmsbm(mm, mock_data(3L, n = 20L), silent = TRUE)
  mm <- predict.mmsbm(mm, mock_data(4L, n = 10L))

  expect_no_error({
    res <- score.mmsbm(mm, silent = FALSE)
  })

  expect_named(res, c("stats", "objects"))
  expect_true("accuracy" %in% names(res$stats))
})


test_that("cv_fit returns accuracies for each fold", {
  future::plan(future::sequential)
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  accuracies <- cv_fit(mm, mock_data(1L), folds = 2L)

  expect_length(accuracies, 2L)
  expect_true(all(accuracies >= 0 & accuracies <= 1))
})
