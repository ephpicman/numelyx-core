/**
 * @file matrix_diagonal_test.c
 * @brief Native tests for Matrix diagonal extraction, construction, and diagonal-related operations.
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

static void test_extract_main_and_offset_diagonals(void)
{
    const double values[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    numerus_matrix *root = NULL;
    numerus_matrix *main_diagonal = NULL;
    numerus_matrix *upper = NULL;
    numerus_matrix *lower = NULL;
    numerus_matrix *empty = NULL;

    assert(numerus_matrix_create_dense(3, 4, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_extract(
        root, 0, &main_diagonal
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(main_diagonal) == 1);
    assert(numerus_matrix_columns(main_diagonal) == 3);
    assert(matrix_value_equals(main_diagonal, 0, 0, 1.0));
    assert(matrix_value_equals(main_diagonal, 0, 1, 6.0));
    assert(matrix_value_equals(main_diagonal, 0, 2, 11.0));

    assert(numerus_matrix_create_diagonal_extract(
        root, 1, &upper
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_columns(upper) == 3);
    assert(matrix_value_equals(upper, 0, 0, 2.0));
    assert(matrix_value_equals(upper, 0, 1, 7.0));
    assert(matrix_value_equals(upper, 0, 2, 12.0));

    assert(numerus_matrix_create_diagonal_extract(
        root, -1, &lower
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_columns(lower) == 2);
    assert(matrix_value_equals(lower, 0, 0, 5.0));
    assert(matrix_value_equals(lower, 0, 1, 10.0));

    assert(numerus_matrix_create_diagonal_extract(
        root, 4, &empty
    ) == NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(empty == NULL);
    assert(numerus_matrix_create_diagonal_extract(
        root, -3, &empty
    ) == NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(empty == NULL);

    numerus_matrix_destroy(lower);
    numerus_matrix_destroy(upper);
    numerus_matrix_destroy(main_diagonal);
    numerus_matrix_destroy(root);
}

static void test_construct_diagonal_from_vector(void)
{
    const double values[] = {5, 6, 7};
    numerus_matrix *vector = NULL;
    numerus_matrix *upper = NULL;
    numerus_matrix *lower = NULL;
    numerus_matrix *main_diagonal = NULL;
    numerus_matrix *column_vector = NULL;
    numerus_matrix *column_diagonal = NULL;

    assert(numerus_matrix_create_dense(1, 3, values, &vector) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_from_vector(
        vector, 1, &upper
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(upper) == 4);
    assert(numerus_matrix_columns(upper) == 4);
    assert(matrix_value_equals(upper, 0, 1, 5.0));
    assert(matrix_value_equals(upper, 1, 2, 6.0));
    assert(matrix_value_equals(upper, 2, 3, 7.0));
    assert(matrix_value_equals(upper, 0, 0, 0.0));
    assert(matrix_value_equals(upper, 3, 3, 0.0));

    assert(numerus_matrix_create_diagonal_from_vector(
        vector, -1, &lower
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(lower, 1, 0, 5.0));
    assert(matrix_value_equals(lower, 2, 1, 6.0));
    assert(matrix_value_equals(lower, 3, 2, 7.0));
    assert(matrix_value_equals(lower, 0, 0, 0.0));

    assert(numerus_matrix_create_diagonal_from_vector(
        vector, 0, &main_diagonal
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(main_diagonal) == 3);
    assert(matrix_value_equals(main_diagonal, 0, 0, 5.0));
    assert(matrix_value_equals(main_diagonal, 1, 1, 6.0));
    assert(matrix_value_equals(main_diagonal, 2, 2, 7.0));
    assert(matrix_value_equals(main_diagonal, 0, 1, 0.0));

    assert(numerus_matrix_create_dense(3, 1, values, &column_vector) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_from_vector(
        column_vector, 0, &column_diagonal
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(column_diagonal, 0, 0, 5.0));
    assert(matrix_value_equals(column_diagonal, 1, 1, 6.0));
    assert(matrix_value_equals(column_diagonal, 2, 2, 7.0));

    numerus_matrix_destroy(column_diagonal);
    numerus_matrix_destroy(column_vector);
    numerus_matrix_destroy(main_diagonal);
    numerus_matrix_destroy(lower);
    numerus_matrix_destroy(upper);
    numerus_matrix_destroy(vector);
}

static void test_validation_overflow_and_read_failure(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *non_vector = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *diagonal = NULL;
    numerus_matrix *huge_vector = NULL;
    numerus_matrix *small_vector = NULL;
    numerus_matrix *overflow_diagonal = NULL;
    double value = 4321.5;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_from_vector(
        root, 0, &diagonal
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(diagonal == NULL);
    assert(numerus_matrix_create_diagonal_extract(
        NULL, 0, &diagonal
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(diagonal == NULL);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_extract(
        failing, 0, &diagonal
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(diagonal, 0, 1, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 4321.5);

    assert(numerus_matrix_create_from_parent(
        root, 1, SIZE_MAX, &huge_vector
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_from_vector(
        huge_vector, 1, &non_vector
    ) == NUMERUS_MATRIX_OVERFLOW);
    assert(non_vector == NULL);
    assert(numerus_matrix_create_dense(1, 4, values, &small_vector) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_diagonal_from_vector(
        small_vector, PTRDIFF_MAX, &overflow_diagonal
    ) == NUMERUS_MATRIX_OVERFLOW);
    assert(overflow_diagonal == NULL);

    numerus_matrix_destroy(small_vector);
    numerus_matrix_destroy(huge_vector);
    numerus_matrix_destroy(diagonal);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_extract_main_and_offset_diagonals();
    test_construct_diagonal_from_vector();
    test_validation_overflow_and_read_failure();
    puts("Matrix diagonal tests passed.");
    return 0;
}
