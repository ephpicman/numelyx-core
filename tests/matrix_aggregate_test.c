/**
 * @file matrix_aggregate_test.c
 * @brief Native regression tests for Matrix aggregate reductions such as sums, means, and related summaries.
 *
 * @details These native tests define regression coverage for the named Matrix
 * or statistical contract. Assertions are executable specifications: when
 * behavior changes intentionally, update the assertions and the corresponding
 * API documentation together. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static numelyx_core_matrix_status fail_on_row_one(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) context;

    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static int matrix_value_equals(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    double expected
)
{
    double actual = 0.0;

    return numelyx_core_matrix_get(
        matrix, row, column, &actual
    ) == NUMERUS_MATRIX_SUCCESS && actual == expected;
}

static void test_scalar_and_axis_aggregates(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *row_sums = NULL;
    numelyx_core_matrix *column_sums = NULL;
    numelyx_core_matrix *row_means = NULL;
    numelyx_core_matrix *column_means = NULL;
    double result = 0.0;

    assert(numelyx_core_matrix_create_dense(2, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_sum(root, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result == 21.0);
    assert(numelyx_core_matrix_min(root, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result == 1.0);
    assert(numelyx_core_matrix_max(root, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result == 6.0);
    assert(numelyx_core_matrix_mean(root, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result == 3.5);

    assert(numelyx_core_matrix_row_sums(root, &row_sums) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(row_sums) == 2);
    assert(numelyx_core_matrix_columns(row_sums) == 1);
    assert(matrix_value_equals(row_sums, 0, 0, 6.0));
    assert(matrix_value_equals(row_sums, 1, 0, 15.0));

    assert(numelyx_core_matrix_column_sums(root, &column_sums) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(column_sums) == 1);
    assert(numelyx_core_matrix_columns(column_sums) == 3);
    assert(matrix_value_equals(column_sums, 0, 0, 5.0));
    assert(matrix_value_equals(column_sums, 0, 1, 7.0));
    assert(matrix_value_equals(column_sums, 0, 2, 9.0));

    assert(numelyx_core_matrix_row_means(root, &row_means) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(row_means, 0, 0, 2.0));
    assert(matrix_value_equals(row_means, 1, 0, 5.0));

    assert(numelyx_core_matrix_column_means(root, &column_means) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(column_means, 0, 0, 2.5));
    assert(matrix_value_equals(column_means, 0, 1, 3.5));
    assert(matrix_value_equals(column_means, 0, 2, 4.5));

    numelyx_core_matrix_destroy(column_means);
    numelyx_core_matrix_destroy(row_means);
    numelyx_core_matrix_destroy(column_sums);
    numelyx_core_matrix_destroy(row_sums);
    numelyx_core_matrix_destroy(root);
}

static void test_nan_propagation_and_errors(void)
{
    const double nan_values[] = {NAN, 2, 3, 4};
    const double square_values[] = {1, 2, 3, 4};
    numelyx_core_matrix *nan_matrix = NULL;
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *failing = NULL;
    numelyx_core_matrix *row_sums = NULL;
    double result = 987.25;

    assert(numelyx_core_matrix_create_dense(2, 2, nan_values, &nan_matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_sum(nan_matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(result));
    assert(numelyx_core_matrix_mean(nan_matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(result));
    assert(numelyx_core_matrix_min(nan_matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(result));
    assert(numelyx_core_matrix_max(nan_matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(result));

    assert(numelyx_core_matrix_create_dense(2, 2, square_values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    result = 987.25;
    assert(numelyx_core_matrix_sum(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == 987.25);
    assert(numelyx_core_matrix_min(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == 987.25);
    assert(numelyx_core_matrix_row_sums(failing, &row_sums) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(row_sums == NULL);

    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(root);
    numelyx_core_matrix_destroy(nan_matrix);
}

int main(void)
{
    test_scalar_and_axis_aggregates();
    test_nan_propagation_and_errors();
    puts("Matrix aggregate tests passed.");
    return 0;
}
