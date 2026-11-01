/**
 * @file matrix_repeat_test.c
 * @brief Native tests for repeated/tiled Matrix construction and output dimensions.
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

static void test_repeat_tiles_parent(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *repeated = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_repeat(root, 2, 3, &repeated) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(repeated) == 4);
    assert(numerus_matrix_columns(repeated) == 6);
    assert(matrix_value_equals(repeated, 0, 0, 1.0));
    assert(matrix_value_equals(repeated, 0, 2, 1.0));
    assert(matrix_value_equals(repeated, 1, 5, 4.0));
    assert(matrix_value_equals(repeated, 2, 0, 1.0));
    assert(matrix_value_equals(repeated, 3, 5, 4.0));

    numerus_matrix_destroy(repeated);
    numerus_matrix_destroy(root);
}

static void test_block_diagonal_and_rectangular_parent(void)
{
    const double values[] = {1, 2, 3, 4};
    const double rectangular_values[] = {8, 9};
    numerus_matrix *root = NULL;
    numerus_matrix *block = NULL;
    numerus_matrix *rectangular = NULL;
    numerus_matrix *rectangular_block = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_block_diagonal(root, 2, &block) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(block) == 4);
    assert(numerus_matrix_columns(block) == 4);
    assert(matrix_value_equals(block, 0, 0, 1.0));
    assert(matrix_value_equals(block, 1, 1, 4.0));
    assert(matrix_value_equals(block, 0, 2, 0.0));
    assert(matrix_value_equals(block, 1, 3, 0.0));
    assert(matrix_value_equals(block, 2, 0, 0.0));
    assert(matrix_value_equals(block, 2, 2, 1.0));
    assert(matrix_value_equals(block, 3, 3, 4.0));

    assert(numerus_matrix_create_dense(
        1, 2, rectangular_values, &rectangular
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_block_diagonal(
        rectangular, 2, &rectangular_block
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(rectangular_block) == 2);
    assert(numerus_matrix_columns(rectangular_block) == 4);
    assert(matrix_value_equals(rectangular_block, 0, 0, 8.0));
    assert(matrix_value_equals(rectangular_block, 0, 2, 0.0));
    assert(matrix_value_equals(rectangular_block, 1, 0, 0.0));
    assert(matrix_value_equals(rectangular_block, 1, 2, 8.0));
    assert(matrix_value_equals(rectangular_block, 1, 3, 9.0));

    numerus_matrix_destroy(rectangular_block);
    numerus_matrix_destroy(rectangular);
    numerus_matrix_destroy(block);
    numerus_matrix_destroy(root);
}

static void test_validation_overflow_and_read_failure(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *repeated = NULL;
    numerus_matrix *block = NULL;
    double value = 1234.5;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_repeat(root, 0, 1, &repeated) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(repeated == NULL);
    assert(numerus_matrix_create_repeat(
        root, SIZE_MAX, 1, &repeated
    ) == NUMERUS_MATRIX_OVERFLOW);
    assert(repeated == NULL);
    assert(numerus_matrix_create_block_diagonal(
        root, 0, &block
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(block == NULL);
    assert(numerus_matrix_create_repeat(root, 1, 1, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_repeat(
        failing, 2, 2, &repeated
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(repeated, 3, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 1234.5);

    numerus_matrix_destroy(repeated);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_repeat_tiles_parent();
    test_block_diagonal_and_rectangular_parent();
    test_validation_overflow_and_read_failure();
    puts("Matrix repeat and block-diagonal tests passed.");
    return 0;
}
