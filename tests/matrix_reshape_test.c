/**
 * @file matrix_reshape_test.c
 * @brief Native tests for reshape views and preservation of row-major element order.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static numerus_matrix_status fail_on_row_one(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) column;
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

static void test_reshape_row_major_and_flatten(void)
{
    const double values[] = {
        1, 2, 3, 4, 5, 6
    };
    numerus_matrix *root = NULL;
    numerus_matrix *reshaped = NULL;
    numerus_matrix *flattened = NULL;

    assert(numerus_matrix_create_dense(2, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_reshape(root, 3, 2, &reshaped) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(reshaped) == 3);
    assert(numerus_matrix_columns(reshaped) == 2);
    assert(matrix_value_equals(reshaped, 0, 0, 1));
    assert(matrix_value_equals(reshaped, 0, 1, 2));
    assert(matrix_value_equals(reshaped, 1, 0, 3));
    assert(matrix_value_equals(reshaped, 1, 1, 4));
    assert(matrix_value_equals(reshaped, 2, 0, 5));
    assert(matrix_value_equals(reshaped, 2, 1, 6));

    assert(numerus_matrix_create_flatten(reshaped, &flattened) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(flattened) == 1);
    assert(numerus_matrix_columns(flattened) == 6);
    for (size_t column = 0; column < 6; column++) {
        assert(matrix_value_equals(flattened, 0, column, (double) column + 1));
    }

    assert(matrix_value_equals(flattened, 0, 5, 6));

    /* Views borrow their parents, so destroy the child before its parents. */
    numerus_matrix_destroy(flattened);
    numerus_matrix_destroy(reshaped);
    numerus_matrix_destroy(root);
}

static void test_validation_and_nested_reshape(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *root = NULL;
    numerus_matrix *reshaped = NULL;
    numerus_matrix *nested = NULL;

    assert(numerus_matrix_create_dense(2, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_reshape(root, 4, 2, &reshaped) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(reshaped == NULL);
    assert(numerus_matrix_create_reshape(root, 0, 6, &reshaped) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(reshaped == NULL);
    assert(numerus_matrix_create_reshape(NULL, 3, 2, &reshaped) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(reshaped == NULL);
    assert(numerus_matrix_create_reshape(root, 3, 2, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_reshape(root, 3, 2, &reshaped) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_reshape(reshaped, 2, 3, &nested) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(nested, 0, 0, 1));
    assert(matrix_value_equals(nested, 0, 2, 3));
    assert(matrix_value_equals(nested, 1, 0, 4));
    assert(matrix_value_equals(nested, 1, 2, 6));

    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(reshaped);
    numerus_matrix_destroy(root);
}

static void test_overflow_and_parent_read_failure(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *huge = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *reshaped = NULL;
    double value = 765.0;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent(
        root, SIZE_MAX, 2, &huge
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_flatten(huge, &reshaped) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(reshaped == NULL);
    assert(numerus_matrix_create_reshape(huge, 1, 1, &reshaped) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(reshaped == NULL);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_reshape(failing, 1, 4, &reshaped) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(reshaped, 0, 2, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 765.0);

    numerus_matrix_destroy(reshaped);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(huge);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_reshape_row_major_and_flatten();
    test_validation_and_nested_reshape();
    test_overflow_and_parent_read_failure();
    puts("Matrix reshape tests passed.");
    return 0;
}
