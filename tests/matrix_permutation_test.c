/**
 * @file matrix_permutation_test.c
 * @brief Native tests for row/column permutation operations and index validation.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <stdio.h>

static numerus_matrix_status fail_on_row_one(
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
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double expected
)
{
    double actual = 0.0;

    return numerus_matrix_get(
        matrix, row, column, &actual
    ) == NUMERUS_MATRIX_SUCCESS && actual == expected;
}

static void test_row_and_column_permutations(void)
{
    const double row_values[] = {1, 2, 3, 4, 5, 6};
    const double column_values[] = {1, 2, 3, 4, 5, 6};
    const size_t row_permutation[] = {2, 0, 1};
    const size_t column_permutation[] = {2, 0, 1};
    const size_t identity[] = {0, 1, 2};
    numerus_matrix *rows = NULL;
    numerus_matrix *columns = NULL;
    numerus_matrix *permuted_rows = NULL;
    numerus_matrix *permuted_columns = NULL;
    numerus_matrix *identity_rows = NULL;

    assert(numerus_matrix_create_dense(3, 2, row_values, &rows) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_permute_rows(
        rows, row_permutation, 3, &permuted_rows
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(permuted_rows, 0, 0, 5.0));
    assert(matrix_value_equals(permuted_rows, 0, 1, 6.0));
    assert(matrix_value_equals(permuted_rows, 1, 0, 1.0));
    assert(matrix_value_equals(permuted_rows, 2, 1, 4.0));

    assert(numerus_matrix_create_permute_rows(
        rows, identity, 3, &identity_rows
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(identity_rows, 2, 0, 5.0));

    assert(numerus_matrix_create_dense(2, 3, column_values, &columns) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_permute_columns(
        columns, column_permutation, 3, &permuted_columns
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(permuted_columns, 0, 0, 3.0));
    assert(matrix_value_equals(permuted_columns, 0, 1, 1.0));
    assert(matrix_value_equals(permuted_columns, 0, 2, 2.0));
    assert(matrix_value_equals(permuted_columns, 1, 0, 6.0));
    assert(matrix_value_equals(permuted_columns, 1, 2, 5.0));

    numerus_matrix_destroy(permuted_columns);
    numerus_matrix_destroy(columns);
    numerus_matrix_destroy(identity_rows);
    numerus_matrix_destroy(permuted_rows);
    numerus_matrix_destroy(rows);
}

static void test_invalid_permutations_and_read_failure(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    const size_t duplicate[] = {0, 0, 2};
    const size_t out_of_range[] = {0, 1, 3};
    const size_t too_short[] = {0, 1};
    const size_t valid[] = {2, 1, 0};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *permuted = NULL;
    double value = 6543.25;

    assert(numerus_matrix_create_dense(3, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_permute_rows(
        root, duplicate, 3, &permuted
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(permuted == NULL);
    assert(numerus_matrix_create_permute_rows(
        root, out_of_range, 3, &permuted
    ) == NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(permuted == NULL);
    assert(numerus_matrix_create_permute_rows(
        root, too_short, 2, &permuted
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(permuted == NULL);
    assert(numerus_matrix_create_permute_columns(
        root, valid, 3, &permuted
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(permuted == NULL);
    assert(numerus_matrix_create_permute_rows(
        root, valid, 3, NULL
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 3, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_permute_rows(
        failing, valid, 3, &permuted
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(permuted, 1, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 6543.25);

    numerus_matrix_destroy(permuted);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_row_and_column_permutations();
    test_invalid_permutations_and_read_failure();
    puts("Matrix permutation tests passed.");
    return 0;
}
