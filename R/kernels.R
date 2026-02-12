# Pure base R vectorized implementations of the three computational kernels
# used by the EM algorithm. These are the hot path -- no tidyverse here.
#
# All functions expect 1-based integer indices in `data`.
# data: integer matrix N x 3, columns = [user_idx, item_idx, rating_idx]
# theta: numeric matrix U x K  (user-group memberships)
# eta:   numeric matrix I x L  (item-group memberships)
# pr:    numeric array  K x L x R (rating probabilities per group pair)


#' Compute unnormalised responsibilities
#'
#' For every observation (u, i, r) computes
#' omega_\{n,k,l\} = theta_\{u,k\} * eta_\{i,l\} * pr_\{k,l,r\}.
#'
#' @param data Integer matrix of shape N by 3 with columns user_idx, item_idx, rating_idx.
#' @param theta Numeric matrix of shape U by K.
#' @param eta Numeric matrix of shape I by L.
#' @param pr Numeric 3D array of shape K by L by R.
#' @return Numeric 3D array of shape N by K by L.
#' @keywords internal
compute_omegas <- function(data, theta, eta, pr) {
  K <- ncol(theta)
  L <- ncol(eta)
  N <- nrow(data)

  user_idx   <- data[, 1L]
  item_idx   <- data[, 2L]
  rating_idx <- data[, 3L]

  # theta[u_n, ] for each observation -> N x K
  theta_u <- theta[user_idx, , drop = FALSE]

  # eta[i_n, ] for each observation -> N x L
  eta_i <- eta[item_idx, , drop = FALSE]

  # pr[k, l, r_n] for each observation -> rearranged to N x K x L
  # Flatten pr to (K*L) x R (column-major: K varies fastest), index by rating
  pr_flat <- matrix(pr, nrow = K * L, ncol = dim(pr)[3L])
  pr_r <- pr_flat[, rating_idx, drop = FALSE]  # (K*L) x N
  # Reshape to K x L x N then permute to N x K x L
  pr_r <- aperm(array(pr_r, dim = c(K, L, N)), c(3L, 1L, 2L))

  # Broadcast theta_u (N x K) to N x K x L via column-major recycling:
  # array() recycles the N*K elements across L slices -> theta_exp[n,k,l] = theta_u[n,k]
  theta_exp <- array(theta_u, dim = c(N, K, L))

  # Broadcast eta_i (N x L) to N x K x L:
  # First make N x L x K (recycled across K), then permute dims 2 and 3
  eta_exp <- aperm(array(eta_i, dim = c(N, L, K)), c(1L, 3L, 2L))

  theta_exp * eta_exp * pr_r
}


#' M-step parameter updates (unnormalised numerators)
#'
#' Computes the scatter-add accumulations for theta, eta, and pr using the
#' MMSBM update equations.
#'
#' @inheritParams compute_omegas
#' @return A list with elements `n_theta` (U by K), `n_eta` (I by L),
#'   `n_pr` (K by L by R).
#' @keywords internal
update_coefficients <- function(data, theta, eta, pr) {
  omegas <- compute_omegas(data, theta, eta, pr)

  K <- ncol(theta)
  L <- ncol(eta)
  R <- dim(pr)[3L]
  N <- nrow(data)

  # Normalise omegas: sum over K and L for each observation
  # rowSums with dims=1 sums dimensions 2..end -> vector of length N
  sum_omega <- rowSums(omegas, dims = 1L)
  eps <- .Machine$double.eps
  # Division recycles length-N vector along first dim (column-major) -> correct
  increments <- omegas / pmax(sum_omega, eps)

  user_idx   <- data[, 1L]
  item_idx   <- data[, 2L]
  rating_idx <- data[, 3L]

  # --- theta update: sum increments over L (dim 3), scatter-add by user ---
  # rowSums with dims=2 sums dim 3 -> N x K matrix
  inc_theta <- rowSums(increments, dims = 2L)
  n_theta <- matrix(0, nrow = nrow(theta), ncol = K)
  rs_theta <- rowsum(inc_theta, user_idx, reorder = FALSE)
  n_theta[as.integer(rownames(rs_theta)), ] <- rs_theta

  # --- eta update: sum increments over K (dim 2), scatter-add by item ---
  # Permute to (N, L, K) then sum dim 3 -> N x L matrix
  inc_eta <- rowSums(aperm(increments, c(1L, 3L, 2L)), dims = 2L)
  n_eta <- matrix(0, nrow = nrow(eta), ncol = L)
  rs_eta <- rowsum(inc_eta, item_idx, reorder = FALSE)
  n_eta[as.integer(rownames(rs_eta)), ] <- rs_eta

  # --- pr update: sum increments grouped by rating ---
  n_pr <- array(0, dim = c(K, L, R))
  for (r in seq_len(R)) {
    mask <- rating_idx == r
    if (any(mask)) {
      # colSums with dims=1 sums over dim 1 (observations) -> K x L matrix
      n_pr[, , r] <- colSums(increments[mask, , , drop = FALSE], dims = 1L)
    }
  }

  list(n_theta = n_theta, n_eta = n_eta, n_pr = n_pr)
}


#' Vectorised rating distribution p(r | u, i) for all observations
#'
#' Computes the product distribution as a single matrix multiplication.
#'
#' @inheritParams compute_omegas
#' @return Numeric matrix of shape N by R.
#' @keywords internal
prod_dist <- function(data, theta, eta, pr) {
  K <- ncol(theta)
  L <- ncol(eta)

  theta_u <- theta[data[, 1L], , drop = FALSE]  # N x K
  eta_i   <- eta[data[, 2L], , drop = FALSE]    # N x L

  # Build N x (K*L) weight matrix. R's column-major array layout means
  # matrix(pr, nrow = K*L) flattens with K varying fastest. We need the
  # weight columns in the same order: w[n, (l-1)*K + k] = theta[n,k] * eta[n,l].
  weights <- theta_u[, rep(seq_len(K), times = L), drop = FALSE] *
             eta_i[, rep(seq_len(L), each = K), drop = FALSE]

  # Flatten pr (K x L x R) to (K*L) x R (column-major: K varies fastest)
  pr_flat <- matrix(pr, nrow = K * L, ncol = dim(pr)[3L])

  # Single matrix multiply: (N x K*L) %*% (K*L x R) -> N x R
  weights %*% pr_flat
}
