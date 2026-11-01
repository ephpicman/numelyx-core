/**
 * @file matrix_slice_test.c
 * @brief Native tests for sliced Matrix views, bounds, and parent-coordinate translation.
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

static numerus_matrix_status fail_on_row_two(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) column;
    (void) context;

    if (row == 2) {
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

static void test_slice_values_and_nested_views(void)
{
    const double values[] = {
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19
    };
    numerus_matrix *root = NULL;
    numerus_matrix *slice = NULL;
    numerus_matrix *nested = NULL;

    assert(numerus_matrix_create_dense(4, 5, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_slice(root, 1, 2, 2, 3, &slice) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(slice) == 2);
    assert(numerus_matrix_columns(slice) == 3);
    assert(matrix_value_equals(slice, 0, 0, 7.0));
    assert(matrix_value_equals(slice, 0, 1, 8.0));
    assert(matrix_value_equals(slice, 0, 2, 9.0));
    assert(matrix_value_equals(slice, 1, 0, 12.0));
    assert(matrix_value_equals(slice, 1, 1, 13.0));
    assert(matrix_value_equals(slice, 1, 2, 14.0));

    assert(numerus_matrix_create_slice(slice, 1, 1, 1, 2, &nested) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(nested) == 1);
    assert(numerus_matrix_columns(nested) == 2);
    assert(matrix_value_equals(nested, 0, 0, 13.0));
    assert(matrix_value_equals(nested, 0, 1, 14.0));
    assert(matrix_value_equals(root, 2, 3, 13.0));

    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(slice);
    numerus_matrix_destroy(root);
}

static void test_slice_validation_and_error_propagation(void)
{
    const double values[] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11
    };
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *slice = NULL;
    double value = 987.0;

    assert(numerus_matrix_create_dense(3, 4, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    slice = root;
    assert(numerus_matrix_create_slice(root, 3, 1, 0, 1, &slice) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(slice == NULL);
    assert(numerus_matrix_create_slice(root, 0, 1, 3, 2, &slice) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(slice == NULL);
    assert(numerus_matrix_create_slice(root, SIZE_MAX, 1, 0, 1, &slice) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(slice == NULL);
    assert(numerus_matrix_create_slice(root, 0, 0, 0, 1, &slice) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(slice == NULL);
    assert(numerus_matrix_create_slice(NULL, 0, 1, 0, 1, &slice) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(slice == NULL);
    assert(numerus_matrix_create_slice(root, 0, 1, 0, 1, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 3, 4, fail_on_row_two, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_slice(failing, 1, 2, 0, 2, &slice) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(slice, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 4.0);
    value = 987.0;
    assert(numerus_matrix_get(slice, 1, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 987.0);

    numerus_matrix_destroy(slice);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_slice_values_and_nested_views();
    test_slice_validation_and_error_propagation();
    puts("Matrix slice tests passed.");
    return 0;
}
