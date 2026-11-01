/**
 * @file matrix_null_space_test.c
 * @brief Native tests for null-space basis construction and residual properties.
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

static void assert_null_space_residual(
    const numerus_matrix *matrix,
    const numerus_matrix *basis,
    size_t nullity
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(matrix);
    size_t basis_column;

    assert(basis != NULL);
    assert(numerus_matrix_rows(basis) == columns);
    assert(numerus_matrix_columns(basis) == nullity);

    for (basis_column = 0; basis_column < nullity; basis_column++) {
        size_t row;

        for (row = 0; row < rows; row++) {
            size_t column;
            double residual = 0.0;
            double scale = 0.0;

            for (column = 0; column < columns; column++) {
                double a_value;
                double x_value;
                assert(numerus_matrix_get(
                    matrix, row, column, &a_value
                ) == NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(
                    basis, column, basis_column, &x_value
                ) == NUMERUS_MATRIX_SUCCESS);
                residual += a_value * x_value;
                scale += fabs(a_value * x_value);
            }

            assert(fabs(residual) <= 1e-8 * (1.0 + scale));
        }
    }
}

static void test_rank_deficient_rectangular_matrix(void)
{
    const double values[] = {
        1, 2, 3,
        2, 4, 6
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix *basis = NULL;
    size_t nullity = 99;

    assert(numerus_matrix_create_dense(2, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_null_space(matrix, &basis, &nullity) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(nullity == 2);
    assert_null_space_residual(matrix, basis, nullity);
    numerus_matrix_destroy(basis);
    numerus_matrix_destroy(matrix);
}

static void test_trivial_and_full_null_spaces(void)
{
    const double full_rank_values[] = {
        1, 0,
        0, 2,
        1, 1
    };
    const double zero_values[] = {0, 0, 0, 0, 0, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *basis = (void *) 1;
    size_t nullity = 99;

    assert(numerus_matrix_create_dense(3, 2, full_rank_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_null_space(matrix, &basis, &nullity) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(basis == NULL);
    assert(nullity == 0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 3, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_null_space(matrix, &basis, &nullity) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(nullity == 3);
    assert_null_space_residual(matrix, basis, nullity);
    numerus_matrix_destroy(basis);
    numerus_matrix_destroy(matrix);
}

static void test_numerical_rank_threshold(void)
{
    const double near_singular_values[] = {1, 0, 0, 1e-12};
    numerus_matrix *matrix = NULL;
    numerus_matrix *basis = NULL;
    size_t nullity = 0;

    assert(numerus_matrix_create_dense(
        2, 2, near_singular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_null_space(matrix, &basis, &nullity) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(nullity == 1);
    assert_null_space_residual(matrix, basis, nullity);
    numerus_matrix_destroy(basis);
    numerus_matrix_destroy(matrix);
}

static void test_failure_contracts(void)
{
    const double values[] = {1, 2, 3, 4};
    const double nonfinite_values[] = {1, NAN, 3, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *basis = (void *) 1;
    size_t nullity = 123;

    assert(numerus_matrix_null_space(NULL, &basis, &nullity) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(basis == NULL);
    assert(nullity == 123);
    assert(numerus_matrix_null_space(NULL, NULL, &nullity) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_null_space(matrix, &basis, &nullity) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(basis == NULL);
    assert(nullity == 123);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_null_space(failing, &basis, &nullity) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(basis == NULL);
    assert(nullity == 123);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_rank_deficient_rectangular_matrix();
    test_trivial_and_full_null_spaces();
    test_numerical_rank_threshold();
    test_failure_contracts();
    puts("Matrix null-space tests passed.");
    return 0;
}
