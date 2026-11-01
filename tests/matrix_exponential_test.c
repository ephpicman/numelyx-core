/**
 * @file matrix_exponential_test.c
 * @brief Native tests for matrix exponential accuracy, convergence, and invalid inputs.
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

static void assert_value(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double expected,
    double tolerance
)
{
    double actual;
    assert(numerus_matrix_get(matrix, row, column, &actual) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(actual - expected) <= tolerance);
}

static void test_diagonal_and_zero(void)
{
    const double diagonal[] = {log(2.0), 0.0, 0.0, log(3.0)};
    const double zero[] = {0.0, 0.0, 0.0, 0.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *result = NULL;

    assert(numerus_matrix_create_dense(2, 2, diagonal, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 2.0, 1e-11);
    assert_value(result, 0, 1, 0.0, 1e-11);
    assert_value(result, 1, 0, 0.0, 1e-11);
    assert_value(result, 1, 1, 3.0, 1e-11);
    numerus_matrix_destroy(result);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, zero, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 1.0, 1e-12);
    assert_value(result, 0, 1, 0.0, 1e-12);
    assert_value(result, 1, 0, 0.0, 1e-12);
    assert_value(result, 1, 1, 1.0, 1e-12);
    numerus_matrix_destroy(result);
    numerus_matrix_destroy(matrix);
}

static void test_nilpotent_and_rotation(void)
{
    const double nilpotent[] = {0.0, 1.0, 0.0, 0.0};
    const double rotation[] = {0.0, 1.57079632679489661923, -1.57079632679489661923, 0.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *result = NULL;

    assert(numerus_matrix_create_dense(2, 2, nilpotent, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 1.0, 1e-12);
    assert_value(result, 0, 1, 1.0, 1e-12);
    assert_value(result, 1, 0, 0.0, 1e-12);
    assert_value(result, 1, 1, 1.0, 1e-12);
    numerus_matrix_destroy(result);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, rotation, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 0.0, 1e-10);
    assert_value(result, 0, 1, 1.0, 1e-10);
    assert_value(result, 1, 0, -1.0, 1e-10);
    assert_value(result, 1, 1, 0.0, 1e-10);
    numerus_matrix_destroy(result);
    numerus_matrix_destroy(matrix);
}

static void test_scaling_and_failures(void)
{
    const double scaled[] = {10.0, 0.0, 0.0, -10.0};
    const double nonfinite[] = {NAN, 0.0, 0.0, 1.0};
    const double overflow[] = {1000.0, 0.0, 0.0, 0.0};
    const double rectangular[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *result = (void *) 1;
    double value;

    assert(numerus_matrix_create_dense(2, 2, scaled, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, exp(10.0), 1e-7);
    assert_value(result, 1, 1, exp(-10.0), 1e-12);
    numerus_matrix_destroy(result);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, nonfinite, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_NON_FINITE);
    assert(result == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, overflow, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_NON_FINITE);
    assert(result == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 3, rectangular, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_exponential(matrix, &result) == NUMERUS_MATRIX_NOT_SQUARE);
    assert(result == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_exponential(NULL, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_dense(1, 1, (double[]){0.0}, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(matrix, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == 0.0);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_diagonal_and_zero();
    test_nilpotent_and_rotation();
    test_scaling_and_failures();
    puts("Matrix exponential tests passed.");
    return 0;
}
