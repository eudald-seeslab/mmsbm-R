# Data encoding and decoding for the MMSBM model.
#
# Uses an environment as a mutable "encoding store" that format_train_data()
# populates and format_test_data() reads from. This is the one piece of
# shared mutable state in the package.


#' Create a fresh encoding environment
#'
#' @return An environment with NULL slots for obs_dict, items_dict, ratings_dict.
#' @keywords internal
create_encoding <- function() {
  env <- new.env(parent = emptyenv())
  env$obs_dict     <- NULL
  env$items_dict   <- NULL
  env$ratings_dict <- NULL
  env
}


#' Build a named integer encoding for a character vector
#'
#' Sorts unique values and maps them to 1, 2, ... Returned as a named integer
#' vector where names are the original (character) values.
#'
#' @param x Character vector.
#' @return Named integer vector.
#' @keywords internal
create_values_dict <- function(x) {
  vals <- sort(unique(x))
  stats::setNames(seq_along(vals), vals)
}


#' Apply an encoding to a character vector
#'
#' @param x Character vector.
#' @param dict Named integer vector (as returned by `create_values_dict`).
#' @return Integer vector.
#' @keywords internal
rename_values <- function(x, dict) {
  unname(dict[as.character(x)])
}


#' Format training data
#'
#' Validates the data, creates encoding dictionaries, and returns an integer
#' matrix with 1-based indices.
#'
#' @param data A data.frame (or tibble) with three columns: users, items, ratings.
#' @param encoding An encoding environment (from `create_encoding()`).
#' @return An integer matrix of shape N by 3. The encoding environment is
#'   populated as a side-effect.
#' @keywords internal
format_train_data <- function(data, encoding) {
  data <- tibble::as_tibble(data)
  # Convert everything to character for uniform handling
  data <- dplyr::mutate(data, dplyr::across(dplyr::everything(), as.character))

  # Check for missing values
  if (any(is.na(data))) {
    stop("Data contains missing values. Aborting.")
  }

  # Create encoding dictionaries (1-based)
  encoding$obs_dict     <- create_values_dict(data[[1L]])
  encoding$items_dict   <- create_values_dict(data[[2L]])
  encoding$ratings_dict <- create_values_dict(data[[3L]])

  # Apply encoding
  out <- matrix(
    c(
      rename_values(data[[1L]], encoding$obs_dict),
      rename_values(data[[2L]], encoding$items_dict),
      rename_values(data[[3L]], encoding$ratings_dict)
    ),
    ncol = 3L
  )
  storage.mode(out) <- "integer"
  out
}


#' Format test data
#'
#' Validates that test users/items/ratings were seen during training, removes
#' unseen ones with a warning, and returns an integer matrix.
#'
#' @param data A data.frame (or tibble) with three columns: users, items, ratings.
#' @param encoding An encoding environment populated by `format_train_data()`.
#' @return An integer matrix of shape N by 3.
#' @keywords internal
format_test_data <- function(data, encoding) {
  data <- tibble::as_tibble(data)
  data <- dplyr::mutate(data, dplyr::across(dplyr::everything(), as.character))

  if (any(is.na(data))) {
    stop("Data contains missing values. Aborting.")
  }

  # Check each column against training dictionaries
  col_info <- list(
    list(name = "users",   dict = encoding$obs_dict),
    list(name = "items",   dict = encoding$items_dict),
    list(name = "ratings", dict = encoding$ratings_dict)
  )

  for (i in seq_along(col_info)) {
    test_vals  <- unique(data[[i]])
    train_vals <- names(col_info[[i]]$dict)
    unseen     <- setdiff(test_vals, train_vals)

    if (length(unseen) > 0L) {
      warning(
        "The ", col_info[[i]]$name, " ",
        paste(unseen, collapse = ", "),
        " are in the test set but weren't in the train set so I'll remove them.",
        call. = FALSE
      )
      data <- dplyr::filter(data, !(data[[i]] %in% unseen))
    }
  }

  out <- matrix(
    c(
      rename_values(data[[1L]], encoding$obs_dict),
      rename_values(data[[2L]], encoding$items_dict),
      rename_values(data[[3L]], encoding$ratings_dict)
    ),
    ncol = 3L
  )
  storage.mode(out) <- "integer"
  out
}


#' Convert theta matrix back to original user labels
#'
#' @param theta Numeric matrix (U x K).
#' @param encoding Encoding environment.
#' @return A tibble with original user IDs as row names.
#' @keywords internal
return_theta_indices <- function(theta, encoding) {
  df <- tibble::as_tibble(as.data.frame(theta))
  inv <- invert_named_int(encoding$obs_dict)
  df <- tibble::add_column(df, user = inv[seq_len(nrow(theta))], .before = 1L)
  df
}


#' Convert eta matrix back to original item labels
#'
#' @param eta Numeric matrix (I x L).
#' @param encoding Encoding environment.
#' @return A tibble with original item IDs as row names.
#' @keywords internal
return_eta_indices <- function(eta, encoding) {
  df <- tibble::as_tibble(as.data.frame(eta))
  inv <- invert_named_int(encoding$items_dict)
  df <- tibble::add_column(df, item = inv[seq_len(nrow(eta))], .before = 1L)
  df
}


#' Convert pr array back to a named list of tibbles
#'
#' @param pr Numeric 3D array (K x L x R).
#' @param encoding Encoding environment.
#' @return A named list of tibbles, one per rating level.
#' @keywords internal
return_pr_indices <- function(pr, encoding) {
  inv <- invert_named_int(encoding$ratings_dict)
  R <- dim(pr)[3L]
  prs <- vector("list", R)
  for (r in seq_len(R)) {
    prs[[inv[r]]] <- tibble::as_tibble(as.data.frame(pr[, , r]))
  }
  prs
}
