# Utility functions for cross-validation and dictionary inversion.


#' Invert a named integer vector
#'
#' Swaps names and values: if input maps "a" -> 1, output maps 1 -> "a".
#'
#' @param x Named integer vector.
#' @return Named character vector indexed by the original integer values.
#' @keywords internal
invert_named_int <- function(x) {
  stats::setNames(names(x), as.character(unname(x)))
}


#' Compute the number of items per fold for cross-validation
#'
#' @param data A data.frame with at least two columns (users, items).
#' @param folds Integer number of folds.
#' @return Integer: items per fold.
#' @keywords internal
structure_folds <- function(data, folds) {
  n_items <- length(unique(data[[2L]]))
  if (folds > n_items) {
    stop(
      "Fold number can't be higher than ", n_items,
      " since this is the number of different items you have."
    )
  }
  as.integer(n_items / folds)
}


#' Sample up to n indices from a group
#'
#' Tries to sample `n` indices without replacement; if the group is too small,
#' reduces n until it works.
#'
#' @param indices Integer vector of row indices belonging to a group.
#' @param n Maximum number of items to sample.
#' @return Integer vector of sampled indices.
#' @keywords internal
get_n_per_group <- function(indices, n) {
  for (i in rev(seq_len(n))) {
    if (length(indices) >= i) {
      return(sample(indices, i))
    }
  }
  integer(0L)
}
