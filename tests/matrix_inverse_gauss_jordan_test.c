/**
 * @file matrix_inverse_gauss_jordan_test.c
 * @brief Native tests for Gauss–Jordan inversion, pivoting, and singular-input behavior.
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

static void assert_inverse(const numerus_matrix *matrix, const numerus_matrix *inverse)
{
    size_t size = numerus_matrix_rows(matrix);
    for (size_t row = 0; row < size; row++) {
        for (size_t column = 0; column < size; column++) {
            double sum = 0.0;
            for (size_t k = 0; k < size; k++) {
                double a, b;
                assert(numerus_matrix_get(matrix, row, k, &a) == NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(inverse, k, column, &b) == NUMERUS_MATRIX_SUCCESS);
                sum += a * b;
            }
            assert(fabs(sum - (row == column ? 1.0 : 0.0)) < 1e-9);
        }
    }
}

static void test_inverse_and_pivoting(void)
{
    const double values[] = {4, 7, 2, 6};
    const double pivot_values[] = {0, 1, 1, 0};
    numerus_matrix *matrix = NULL, *inverse = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse_gauss_jordan(matrix, &inverse) == NUMERUS_MATRIX_SUCCESS);
    assert_inverse(matrix, inverse);
    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, pivot_values, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse_gauss_jordan(matrix, &inverse) == NUMERUS_MATRIX_SUCCESS);
    assert_inverse(matrix, inverse);
    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);
}

static void test_failures_and_scale(void)
{
    const double singular_values[] = {1, 2, 2, 4};
    const double tiny_values[] = {1e-100, 0, 0, 2e-100};
    const double nonfinite_values[] = {1, NAN, 0, 1};
    numerus_matrix *matrix = NULL, *inverse = (void *) 1;

    assert(numerus_matrix_inverse_gauss_jordan(NULL, &inverse) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(inverse == NULL);
    assert(numerus_matrix_create_dense(2, 2, singular_values, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse_gauss_jordan(matrix, &inverse) == NUMERUS_MATRIX_SINGULAR);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, tiny_values, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse_gauss_jordan(matrix, &inverse) == NUMERUS_MATRIX_SUCCESS);
    assert_inverse(matrix, inverse);
    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, nonfinite_values, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse_gauss_jordan(matrix, &inverse) == NUMERUS_MATRIX_NON_FINITE);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_inverse_and_pivoting();
    test_failures_and_scale();
    puts("Matrix Gauss-Jordan inverse tests passed.");
    return 0;
}
