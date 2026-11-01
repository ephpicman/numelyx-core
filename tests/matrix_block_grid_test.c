/**
 * @file matrix_block_grid_test.c
 * @brief Native tests for composing block grids and validating block dimensions.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <stdint.h>
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

static void test_assemble_heterogeneous_block_grid(void)
{
    const double a_values[] = {1, 2};
    const double b_values[] = {3};
    const double c_values[] = {4, 5, 6, 7};
    const double d_values[] = {8, 9};
    numelyx_core_matrix *a = NULL;
    numelyx_core_matrix *b = NULL;
    numelyx_core_matrix *c = NULL;
    numelyx_core_matrix *d = NULL;
    numelyx_core_matrix *grid = NULL;
    numelyx_core_matrix *blocks[] = {a, b, c, d};

    assert(numelyx_core_matrix_create_dense(1, 2, a_values, &a) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 1, b_values, &b) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(2, 2, c_values, &c) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(2, 1, d_values, &d) ==
        NUMERUS_MATRIX_SUCCESS);
    blocks[0] = a;
    blocks[1] = b;
    blocks[2] = c;
    blocks[3] = d;

    assert(numelyx_core_matrix_create_block_grid(blocks, 2, 2, &grid) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(grid) == 3);
    assert(numelyx_core_matrix_columns(grid) == 3);
    assert(matrix_value_equals(grid, 0, 0, 1.0));
    assert(matrix_value_equals(grid, 0, 1, 2.0));
    assert(matrix_value_equals(grid, 0, 2, 3.0));
    assert(matrix_value_equals(grid, 1, 0, 4.0));
    assert(matrix_value_equals(grid, 1, 1, 5.0));
    assert(matrix_value_equals(grid, 1, 2, 8.0));
    assert(matrix_value_equals(grid, 2, 0, 6.0));
    assert(matrix_value_equals(grid, 2, 1, 7.0));
    assert(matrix_value_equals(grid, 2, 2, 9.0));

    numelyx_core_matrix_destroy(grid);
    numelyx_core_matrix_destroy(d);
    numelyx_core_matrix_destroy(c);
    numelyx_core_matrix_destroy(b);
    numelyx_core_matrix_destroy(a);
}

static void test_validation_and_overflow(void)
{
    const double values[] = {1, 2};
    const double one_value[] = {3};
    const double two_values[] = {4, 5};
    numelyx_core_matrix *a = NULL;
    numelyx_core_matrix *b = NULL;
    numelyx_core_matrix *wrong_width = NULL;
    numelyx_core_matrix *wrong_height = NULL;
    numelyx_core_matrix *grid = NULL;
    numelyx_core_matrix *valid_blocks[] = {NULL, NULL};
    numelyx_core_matrix *bad_height_blocks[] = {NULL, NULL};
    numelyx_core_matrix *bad_width_blocks[] = {NULL, NULL, NULL, NULL};
    numelyx_core_matrix *missing_blocks[] = {NULL, NULL};

    assert(numelyx_core_matrix_create_dense(1, 2, values, &a) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 1, one_value, &b) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 2, two_values, &wrong_width) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(2, 1, two_values, &wrong_height) ==
        NUMERUS_MATRIX_SUCCESS);

    valid_blocks[0] = a;
    valid_blocks[1] = b;
    bad_height_blocks[0] = a;
    bad_height_blocks[1] = wrong_height;
    bad_width_blocks[0] = a;
    bad_width_blocks[1] = b;
    bad_width_blocks[2] = a;
    bad_width_blocks[3] = wrong_width;
    missing_blocks[0] = a;
    missing_blocks[1] = NULL;

    assert(numelyx_core_matrix_create_block_grid(
        valid_blocks, 0, 2, &grid
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(grid == NULL);
    assert(numelyx_core_matrix_create_block_grid(
        bad_height_blocks, 1, 2, &grid
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(grid == NULL);
    assert(numelyx_core_matrix_create_block_grid(
        bad_width_blocks, 2, 2, &grid
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(grid == NULL);
    assert(numelyx_core_matrix_create_block_grid(
        missing_blocks, 1, 2, &grid
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(grid == NULL);
    assert(numelyx_core_matrix_create_block_grid(
        valid_blocks, SIZE_MAX, 2, &grid
    ) == NUMERUS_MATRIX_OVERFLOW);
    assert(grid == NULL);
    assert(numelyx_core_matrix_create_block_grid(valid_blocks, 1, 2, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numelyx_core_matrix_destroy(wrong_height);
    numelyx_core_matrix_destroy(wrong_width);
    numelyx_core_matrix_destroy(b);
    numelyx_core_matrix_destroy(a);
}

static void test_parent_error_propagation(void)
{
    const double values[] = {1, 2, 3, 4};
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *failing = NULL;
    numelyx_core_matrix *blocks[] = {NULL, NULL};
    numelyx_core_matrix *grid = NULL;
    double value = 9876.5;

    assert(numelyx_core_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    blocks[0] = root;
    blocks[1] = failing;

    assert(numelyx_core_matrix_create_block_grid(blocks, 1, 2, &grid) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(grid, 1, 3, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 9876.5);

    numelyx_core_matrix_destroy(grid);
    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(root);
}

int main(void)
{
    test_assemble_heterogeneous_block_grid();
    test_validation_and_overflow();
    test_parent_error_propagation();
    puts("Matrix block-grid tests passed.");
    return 0;
}
