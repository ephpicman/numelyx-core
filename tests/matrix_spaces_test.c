/**
 * @file matrix_spaces_test.c
 * @brief Native tests for row-space and column-space bases and their span properties.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <math.h>
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

static void assert_basis_columns_are_source_columns(
    const numerus_matrix *matrix,
    const numerus_matrix *basis,
    size_t dimension
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(matrix);
    size_t basis_column;

    assert(basis != NULL);
    assert(numerus_matrix_rows(basis) == rows);
    assert(numerus_matrix_columns(basis) == dimension);

    for (basis_column = 0; basis_column < dimension; basis_column++) {
        size_t source_column;
        bool found = false;

        for (source_column = 0; source_column < columns && !found; source_column++) {
            size_t row;
            bool matches = true;

            for (row = 0; row < rows; row++) {
                double source_value;
                double basis_value;
                assert(numerus_matrix_get(
                    matrix, row, source_column, &source_value
                ) == NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(
                    basis, row, basis_column, &basis_value
                ) == NUMERUS_MATRIX_SUCCESS);
                if (source_value != basis_value) {
                    matches = false;
                    break;
                }
            }
            found = matches;
        }
        assert(found);
    }
}

static void assert_basis_rows_are_source_rows(
    const numerus_matrix *matrix,
    const numerus_matrix *basis,
    size_t dimension
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(matrix);
    size_t basis_row;

    assert(basis != NULL);
    assert(numerus_matrix_rows(basis) == dimension);
    assert(numerus_matrix_columns(basis) == columns);

    for (basis_row = 0; basis_row < dimension; basis_row++) {
        size_t source_row;
        bool found = false;

        for (source_row = 0; source_row < rows && !found; source_row++) {
            size_t column;
            bool matches = true;

            for (column = 0; column < columns; column++) {
                double source_value;
                double basis_value;
                assert(numerus_matrix_get(
                    matrix, source_row, column, &source_value
                ) == NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(
                    basis, basis_row, column, &basis_value
                ) == NUMERUS_MATRIX_SUCCESS);
                if (source_value != basis_value) {
                    matches = false;
                    break;
                }
            }
            found = matches;
        }
        assert(found);
    }
}

static void test_rank_deficient_bases(void)
{
    const double values[] = {
        1, 0, 1,
        0, 1, 1,
        1, 1, 2
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix *columns = NULL;
    numerus_matrix *rows = NULL;
    size_t column_dimension = 0;
    size_t row_dimension = 0;

    assert(numerus_matrix_create_dense(3, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_column_space_basis(
        matrix, &columns, &column_dimension
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(column_dimension == 2);
    assert_basis_columns_are_source_columns(matrix, columns, column_dimension);

    assert(numerus_matrix_row_space_basis(
        matrix, &rows, &row_dimension
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(row_dimension == 2);
    assert_basis_rows_are_source_rows(matrix, rows, row_dimension);

    numerus_matrix_destroy(rows);
    numerus_matrix_destroy(columns);
    numerus_matrix_destroy(matrix);
}

static void test_full_rank_and_zero_matrix(void)
{
    const double full_rank_values[] = {1, 2, 3, 5, 7, 11};
    const double zero_values[] = {0, 0, 0, 0, 0, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *basis = (void *) 1;
    size_t dimension = 99;

    assert(numerus_matrix_create_dense(3, 2, full_rank_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_column_space_basis(
        matrix, &basis, &dimension
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(dimension == 2);
    assert_basis_columns_are_source_columns(matrix, basis, dimension);
    numerus_matrix_destroy(basis);

    assert(numerus_matrix_row_space_basis(
        matrix, &basis, &dimension
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(dimension == 2);
    assert_basis_rows_are_source_rows(matrix, basis, dimension);
    numerus_matrix_destroy(basis);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 3, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_column_space_basis(
        matrix, &basis, &dimension
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(basis == NULL && dimension == 0);
    assert(numerus_matrix_row_space_basis(
        matrix, &basis, &dimension
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(basis == NULL && dimension == 0);
    numerus_matrix_destroy(matrix);
}

static void test_failure_contracts(void)
{
    const double values[] = {1, 2, 3, 4};
    const double nonfinite_values[] = {1, NAN, 3, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *basis = (void *) 1;
    size_t dimension = 77;

    assert(numerus_matrix_column_space_basis(
        NULL, &basis, &dimension
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(basis == NULL && dimension == 77);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_column_space_basis(
        matrix, &basis, &dimension
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(basis == NULL && dimension == 77);
    assert(numerus_matrix_row_space_basis(
        matrix, &basis, &dimension
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(basis == NULL && dimension == 77);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_column_space_basis(
        failing, &basis, &dimension
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(basis == NULL && dimension == 77);
    assert(numerus_matrix_row_space_basis(
        failing, &basis, &dimension
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(basis == NULL && dimension == 77);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_rank_deficient_bases();
    test_full_rank_and_zero_matrix();
    test_failure_contracts();
    puts("Matrix space basis tests passed.");
    return 0;
}
