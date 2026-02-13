# Tests for data encoding/decoding.
# These replicate the Python test_data_handler.py test cases.

test_that("test users subset of train does not raise errors", {
  train_df <- data.frame(
    users   = c("u1", "u2", "u3", "u1"),
    items   = c("i1", "i2", "i1", "i3"),
    ratings = c("1", "2", "3", "1"),
    stringsAsFactors = FALSE
  )
  test_df <- data.frame(
    users   = c("u1", "u2"),
    items   = c("i3", "i2"),
    ratings = c("2", "1"),
    stringsAsFactors = FALSE
  )

  enc <- create_encoding()
  train_mat <- format_train_data(train_df, enc)
  test_mat  <- format_test_data(test_df, enc)

  expect_equal(nrow(test_mat), nrow(test_df))
  expect_true(is.integer(test_mat))
})


test_that("unseen users in test are dropped with warning", {
  train_df <- data.frame(
    users   = c("u1", "u2"),
    items   = c("i1", "i2"),
    ratings = c("1", "2"),
    stringsAsFactors = FALSE
  )
  test_df <- data.frame(
    users   = c("u3", "u1"),
    items   = c("i3", "i1"),
    ratings = c("2", "1"),
    stringsAsFactors = FALSE
  )

  enc <- create_encoding()
  train_mat <- format_train_data(train_df, enc)

  # Both u3 (unseen user) and i3 (unseen item) produce warnings
  # Warnings explain why predictions are impossible for unseen nodes
  expect_warning(
    expect_warning(
      test_mat <- format_test_data(test_df, enc),
      "unseen users.*u3.*theta"
    ),
    "unseen items.*i3.*eta"
  )
  # u3 and i3 should both be dropped (same row)
  expect_equal(nrow(test_mat), 1L)
  expect_true(is.integer(test_mat))
})


test_that("unseen items in test are dropped with warning", {
  train_df <- data.frame(
    users   = c("u1", "u2"),
    items   = c("i1", "i2"),
    ratings = c("1", "2"),
    stringsAsFactors = FALSE
  )
  test_df <- data.frame(
    users   = c("u1", "u2"),
    items   = c("i3", "i2"),
    ratings = c("1", "2"),
    stringsAsFactors = FALSE
  )

  enc <- create_encoding()
  train_mat <- format_train_data(train_df, enc)

  expect_warning(
    test_mat <- format_test_data(test_df, enc),
    "unseen items.*i3.*eta"
  )
  expect_equal(nrow(test_mat), 1L)
  expect_true(is.integer(test_mat))
})


test_that("unseen ratings in test are dropped with warning", {
  train_df <- data.frame(
    users   = c("u1", "u2"),
    items   = c("i1", "i2"),
    ratings = c("1", "2"),
    stringsAsFactors = FALSE
  )
  test_df <- data.frame(
    users   = c("u1", "u2"),
    items   = c("i1", "i2"),
    ratings = c("3", "2"),
    stringsAsFactors = FALSE
  )

  enc <- create_encoding()
  train_mat <- format_train_data(train_df, enc)

  expect_warning(
    test_mat <- format_test_data(test_df, enc),
    "unseen ratings.*3.*probability tensor"
  )
  expect_equal(nrow(test_mat), 1L)
  expect_true(is.integer(test_mat))
})


test_that("encoding produces correct 1-based indices", {
  df <- data.frame(
    users   = c("b", "a", "c"),
    items   = c("y", "x", "z"),
    ratings = c("3", "1", "2"),
    stringsAsFactors = FALSE
  )

  enc <- create_encoding()
  mat <- format_train_data(df, enc)

  # Sorted unique values: users = a,b,c -> 1,2,3; items = x,y,z -> 1,2,3
  expect_equal(enc$obs_dict[["a"]], 1L)
  expect_equal(enc$obs_dict[["b"]], 2L)
  expect_equal(enc$obs_dict[["c"]], 3L)

  # Check that the matrix values are correct
  # Row 1: user "b" -> 2, item "y" -> 2, rating "3" -> 3
  expect_equal(mat[1L, ], c(2L, 2L, 3L))
})
