/**
 * @file matrix_cholesky_test.c
 * @brief Native tests for Cholesky factorization, reconstruction accuracy, and invalid matrices.
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

static void assert_cholesky_reconstructs(
    const numerus_matrix *matrix,
    const numerus_matrix *lower,
    double tolerance
)
{
    size_t size = numerus_matrix_rows(matrix);
    size_t row;
    size_t column;

    assert(numerus_matrix_rows(lower) == size);
    assert(numerus_matrix_columns(lower) == size);

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            size_t k;
            double product = 0.0;
            double expected;

            for (k = 0; k < size; k++) {
                double left;
                double right;
                assert(numerus_matrix_get(lower, row, k, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(lower, column, k, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                product += left * right;
            }

            assert(numerus_matrix_get(
                matrix, row, column, &expected
            ) == NUMERUS_MATRIX_SUCCESS);
            assert(fabs(product - expected) <=
                tolerance * (1.0 + fabs(expected)));
        }
    }

    for (row = 0; row < size; row++) {
        for (column = row + 1; column < size; column++) {
            double value;
            assert(numerus_matrix_get(lower, row, column, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(value == 0.0);
        }
    }
}

static void test_known_positive_definite_matrix(void)
{
    const double values[] = {
        4, 12, -16,
        12, 37, -43,
        -16, -43, 98
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix *lower = NULL;
    double value;

    assert(numerus_matrix_create_dense(3, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(lower, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 2.0) < 1e-12);
    assert(numerus_matrix_get(lower, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 6.0) < 1e-12);
    assert(numerus_matrix_get(lower, 2, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 8.0) < 1e-12);
    assert(numerus_matrix_get(lower, 2, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 5.0) < 1e-12);
    assert(numerus_matrix_get(lower, 2, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 3.0) < 1e-12);
    assert_cholesky_reconstructs(matrix, lower, 1e-10);

    numerus_matrix_destroy(lower);
    numerus_matrix_destroy(matrix);
}

static void test_scale_and_definiteness(void)
{
    const double small_values[] = {1e-100, 0, 0, 2e-100};
    const double semidefinite_values[] = {1, 1, 1, 1};
    const double indefinite_values[] = {1, 2, 2, 1};
    const double near_singular_values[] = {1, 0, 0, 1e-12};
    numerus_matrix *matrix = NULL;
    numerus_matrix *lower = (void *) 1;

    assert(numerus_matrix_create_dense(2, 2, small_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_cholesky_reconstructs(matrix, lower, 1e-9);
    numerus_matrix_destroy(lower);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, semidefinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE);
    assert(lower == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, indefinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE);
    assert(lower == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, near_singular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE);
    assert(lower == NULL);
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_read_failures(void)
{
    const double nonsymmetric_values[] = {1, 0, 1, 1};
    const double nonfinite_values[] = {1, NAN, NAN, 1};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *matrix = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *lower = (void *) 1;

    assert(numerus_matrix_cholesky(NULL, &lower) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(lower == NULL);
    assert(numerus_matrix_cholesky(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(lower == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, nonsymmetric_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_NOT_SYMMETRIC);
    assert(lower == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(lower == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(failing, &lower) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(lower == NULL);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_known_positive_definite_matrix();
    test_scale_and_definiteness();
    test_validation_and_read_failures();
    puts("Matrix Cholesky tests passed.");
    return 0;
}
