/**
 * @file matrix_selection_test.c
 * @brief Native tests for row/column selection views, indices, and repeated selections.
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

static void test_row_selection_and_owned_indices(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t indices[] = {2, 0, 2};
    numerus_matrix *root = NULL;
    numerus_matrix *selected = NULL;

    assert(numerus_matrix_create_dense(3, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_select_rows(root, indices, 3, &selected) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(selected) == 3);
    assert(numerus_matrix_columns(selected) == 3);

    /* Index order and duplicates are preserved; the caller's array is copied. */
    indices[0] = 0;
    indices[1] = 0;
    indices[2] = 0;
    assert(matrix_value_equals(selected, 0, 0, 7.0));
    assert(matrix_value_equals(selected, 0, 2, 9.0));
    assert(matrix_value_equals(selected, 1, 0, 1.0));
    assert(matrix_value_equals(selected, 1, 2, 3.0));
    assert(matrix_value_equals(selected, 2, 1, 8.0));

    numerus_matrix_destroy(selected);
    numerus_matrix_destroy(root);
}

static void test_column_selection_and_composition(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const size_t columns[] = {2, 0};
    const size_t reordered[] = {1, 0};
    numerus_matrix *root = NULL;
    numerus_matrix *selected = NULL;
    numerus_matrix *nested = NULL;

    assert(numerus_matrix_create_dense(3, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_select_columns(
        root, columns, 2, &selected
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(selected) == 3);
    assert(numerus_matrix_columns(selected) == 2);
    assert(matrix_value_equals(selected, 0, 0, 3.0));
    assert(matrix_value_equals(selected, 0, 1, 1.0));
    assert(matrix_value_equals(selected, 2, 0, 9.0));
    assert(matrix_value_equals(selected, 2, 1, 7.0));

    assert(numerus_matrix_create_select_columns(
        selected, reordered, 2, &nested
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(nested, 0, 0, 1.0));
    assert(matrix_value_equals(nested, 0, 1, 3.0));
    assert(matrix_value_equals(nested, 2, 0, 7.0));
    assert(matrix_value_equals(nested, 2, 1, 9.0));

    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(selected);
    numerus_matrix_destroy(root);
}

static void test_validation_overflow_and_read_errors(void)
{
    const double values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const size_t invalid_index[] = {3};
    const size_t valid_index[] = {1, 2};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *selected = NULL;
    double value = 456.0;

    assert(numerus_matrix_create_dense(3, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_select_rows(
        root, invalid_index, 1, &selected
    ) == NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(selected == NULL);
    assert(numerus_matrix_create_select_columns(
        root, invalid_index, 1, &selected
    ) == NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(selected == NULL);
    assert(numerus_matrix_create_select_rows(root, valid_index, 0, &selected) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(selected == NULL);
    assert(numerus_matrix_create_select_rows(root, valid_index, SIZE_MAX, &selected) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(selected == NULL);
    assert(numerus_matrix_create_select_rows(NULL, valid_index, 1, &selected) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(selected == NULL);
    assert(numerus_matrix_create_select_columns(root, valid_index, 1, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 3, 3, fail_on_row_two, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_select_rows(
        failing, valid_index, 2, &selected
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(selected, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 3.0);
    value = 456.0;
    assert(numerus_matrix_get(selected, 1, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 456.0);

    numerus_matrix_destroy(selected);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_row_selection_and_owned_indices();
    test_column_selection_and_composition();
    test_validation_overflow_and_read_errors();
    puts("Matrix selection tests passed.");
    return 0;
}
