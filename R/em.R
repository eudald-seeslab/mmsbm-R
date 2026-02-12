# EM algorithm utilities: normalization, likelihood, and the single-run driver.
# These wrap the hot-path kernels from kernels.R with the normalization and
# random-initialization logic.


#' Normalise a matrix by pre-computed factors
#'
#' Divides each row of `mat` by the corresponding row in `norm_factors`.
#'
#' @param mat Numeric matrix.
#' @param norm_factors Numeric matrix of the same shape.
#' @return Normalised matrix.
#' @keywords internal
normalize_with_d <- function(mat, norm_factors) {
  mat / norm_factors
}


#' Normalise a 3D array along its last axis so slices sum to 1
#'
#' For each (k, l) pair, normalises the rating probabilities so that
#' they sum to 1 over ratings.
#'
#' @param arr Numeric 3D array of shape K by L by R.
#' @return Normalised array of the same shape.
#' @keywords internal
normalize_with_self <- function(arr) {
  dims <- dim(arr)
  K <- dims[1L]
  L <- dims[2L]
  R <- dims[3L]

  # Reshape to (K*L) x R
  mat <- matrix(arr, nrow = K * L, ncol = R)
  rs <- rowSums(mat)
  # Avoid division by zero
  rs[rs == 0] <- 1
  mat <- mat / rs
  array(mat, dim = dims)
}


#' Compute log-likelihood of the model
#'
#' @param data Integer matrix (N x 3).
#' @param theta Numeric matrix (U x K).
#' @param eta Numeric matrix (I x L).
#' @param pr Numeric 3D array (K x L x R).
#' @return Scalar log-likelihood value.
#' @keywords internal
compute_likelihood <- function(data, theta, eta, pr) {
  omegas <- compute_omegas(data, theta, eta, pr)

  # Sum over K and L for each observation
  sum_omega <- rowSums(omegas, dims = 1L)  # length N

  eps <- .Machine$double.eps
  safe_omegas <- pmax(omegas, eps)
  safe_sums   <- pmax(sum_omega, eps)

  sum(safe_omegas * log(safe_omegas) - safe_omegas * log(safe_sums))
}


#' Execute one complete EM run with random initialisation
#'
#' @param train Integer matrix (N x 3) of training data.
#' @param config List with model configuration (user_groups, item_groups,
#'   n_users, n_items, n_ratings, norm_factors, debug).
#' @param seed Integer seed for this run's RNG.
#' @param index Integer index of this run (for progress/logging).
#' @param p A progressr progressor function, or NULL.
#' @return A list with elements: likelihood, pr, theta, eta.
#' @keywords internal
run_one_sampling <- function(train, config, seed, index, p = NULL) {
  set.seed(seed)

  n_users  <- config$n_users
  n_items  <- config$n_items
  K        <- config$user_groups
  L        <- config$item_groups
  R        <- config$n_ratings

  # Random initialisation + normalisation
  theta <- normalize_with_d(
    matrix(stats::runif(n_users * K), nrow = n_users, ncol = K),
    config$norm_factors$user
  )
  eta <- normalize_with_d(
    matrix(stats::runif(n_items * L), nrow = n_items, ncol = L),
    config$norm_factors$item
  )
  pr <- normalize_with_self(
    array(stats::runif(K * L * R), dim = c(K, L, R))
  )

  # EM iterations
  for (j in seq_len(config$iterations)) {
    updates <- update_coefficients(train, theta, eta, pr)

    theta <- normalize_with_d(updates$n_theta, config$norm_factors$user)
    eta   <- normalize_with_d(updates$n_eta, config$norm_factors$item)
    pr    <- normalize_with_self(updates$n_pr)

    if (!is.null(p)) p(message = sprintf("Run %d: iter %d/%d", index, j, config$iterations))
  }

  likelihood <- compute_likelihood(train, theta, eta, pr)

  list(
    likelihood = likelihood,
    pr         = pr,
    theta      = theta,
    eta        = eta
  )
}
