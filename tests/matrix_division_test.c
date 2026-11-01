/**
 * @file matrix_division_test.c
 * @brief Native tests for elementwise Matrix division and zero-divisor IEEE-754 semantics.
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

static void test_ieee754_elementwise_division(void)
{
    const double numerator_values[] = {1.0, -1.0, 0.0, INFINITY, NAN, 8.0};
    const double denominator_values[] = {0.0, 0.0, -0.0, INFINITY, 2.0, 2.0};
    numerus_matrix *numerator = NULL;
    numerus_matrix *denominator = NULL;
    numerus_matrix *quotient = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_dense(2, 3, numerator_values, &numerator) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 3, denominator_values, &denominator) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_divide(numerator, denominator, &quotient) ==
        NUMERUS_MATRIX_SUCCESS);

    /* Division by zero is a numeric result, not a status error. */
    assert(numerus_matrix_get(quotient, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isinf(value) && value > 0.0);
    assert(numerus_matrix_get(quotient, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isinf(value) && value < 0.0);
    assert(numerus_matrix_get(quotient, 0, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    assert(numerus_matrix_get(quotient, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value)); /* infinity / infinity */
    assert(numerus_matrix_get(quotient, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value)); /* NaN propagates */
    assert(numerus_matrix_get(quotient, 1, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 4.0);

    numerus_matrix_destroy(quotient);
    numerus_matrix_destroy(denominator);
    numerus_matrix_destroy(numerator);
}

static void test_division_shape_validation(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    numerus_matrix *square = NULL;
    numerus_matrix *row = NULL;
    numerus_matrix *quotient = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &square) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 4, values, &row) ==
        NUMERUS_MATRIX_SUCCESS);

    quotient = square;
    assert(numerus_matrix_create_divide(square, row, &quotient) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(quotient == NULL);
    assert(numerus_matrix_create_divide(NULL, square, &quotient) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(quotient == NULL);
    assert(numerus_matrix_create_divide(square, square, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(row);
    numerus_matrix_destroy(square);
}

int main(void)
{
    test_ieee754_elementwise_division();
    test_division_shape_validation();
    puts("Matrix division tests passed.");
    return 0;
}
