# Test helper: generate synthetic data for MMSBM tests.

RATING_NUM <- 100L

mock_data <- function(seed, n = RATING_NUM) {
  set.seed(seed)
  data.frame(
    users   = paste0("user", sample(0:4, n, replace = TRUE)),
    items   = paste0("item", sample(0:9, n, replace = TRUE)),
    ratings = sample(1:5, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

fit_model <- function() {
  mm <- mmsbm(2L, 2L, iterations = 10L, seed = 1L)
  fit.mmsbm(mm, mock_data(1L))
}
