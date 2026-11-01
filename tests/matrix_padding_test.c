/**
 * @file matrix_padding_test.c
 * @brief Native tests for Matrix padding dimensions, fill values, and edge behavior.
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

static void test_constant_padding_and_nested_views(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *padded = NULL;
    numerus_matrix *nested = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_pad(
        root, 1, 2, 1, 1, -7.0, &padded
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(padded) == 5);
    assert(numerus_matrix_columns(padded) == 4);
    assert(matrix_value_equals(padded, 0, 0, -7.0));
    assert(matrix_value_equals(padded, 0, 3, -7.0));
    assert(matrix_value_equals(padded, 1, 0, -7.0));
    assert(matrix_value_equals(padded, 1, 1, 1.0));
    assert(matrix_value_equals(padded, 1, 2, 2.0));
    assert(matrix_value_equals(padded, 2, 1, 3.0));
    assert(matrix_value_equals(padded, 2, 2, 4.0));
    assert(matrix_value_equals(padded, 4, 2, -7.0));

    assert(numerus_matrix_create_pad(
        padded, 1, 0, 0, 1, 99.0, &nested
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(nested) == 6);
    assert(numerus_matrix_columns(nested) == 5);
    assert(matrix_value_equals(nested, 0, 0, 99.0));
    assert(matrix_value_equals(nested, 2, 1, 1.0));
    assert(matrix_value_equals(nested, 3, 2, 4.0));
    assert(matrix_value_equals(nested, 5, 4, 99.0));

    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(padded);
    numerus_matrix_destroy(root);
}

static void test_zero_extension_and_validation(void)
{
    const double values[] = {5, 6, 7, 8};
    numerus_matrix *root = NULL;
    numerus_matrix *extended = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_zero_extend(
        root, 1, 0, 0, 2, &extended
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(extended) == 3);
    assert(numerus_matrix_columns(extended) == 4);
    assert(matrix_value_equals(extended, 0, 0, 0.0));
    assert(matrix_value_equals(extended, 1, 0, 5.0));
    assert(matrix_value_equals(extended, 1, 1, 6.0));
    assert(matrix_value_equals(extended, 1, 2, 0.0));
    assert(matrix_value_equals(extended, 2, 0, 7.0));
    assert(matrix_value_equals(extended, 2, 1, 8.0));
    assert(matrix_value_equals(extended, 2, 3, 0.0));

    numerus_matrix_destroy(extended);
    extended = root;
    assert(numerus_matrix_create_pad(
        NULL, 1, 1, 1, 1, 0.0, &extended
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(extended == NULL);
    assert(numerus_matrix_create_pad(
        root, SIZE_MAX, 0, 0, 0, 0.0, &extended
    ) == NUMERUS_MATRIX_OVERFLOW);
    assert(extended == NULL);
    assert(numerus_matrix_create_zero_extend(
        root, 0, 0, 0, 0, NULL
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(root);
}

static void test_parent_read_failure_preserves_output(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *padded = NULL;
    double value = 1234.5;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_pad(
        failing, 1, 0, 1, 0, -1.0, &padded
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(padded, 2, 1, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 1234.5);
    assert(matrix_value_equals(padded, 0, 0, -1.0));

    numerus_matrix_destroy(padded);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_constant_padding_and_nested_views();
    test_zero_extension_and_validation();
    test_parent_read_failure_preserves_output();
    puts("Matrix padding tests passed.");
    return 0;
}
