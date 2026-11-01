/**
 * @file matrix_scalar_division_test.c
 * @brief Native tests for scalar division views and zero-divisor rejection.
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

static void test_scalar_division_view(void)
{
    const double values[] = {2.0, -3.0, 0.0, 4.0};
    numerus_matrix *root = NULL;
    numerus_matrix *quotient = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_divide_scalar(root, 2.0, &quotient) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(matrix_value_equals(quotient, 0, 0, 1.0));
    assert(matrix_value_equals(quotient, 0, 1, -1.5));
    assert(matrix_value_equals(quotient, 1, 0, 0.0));
    assert(matrix_value_equals(quotient, 1, 1, 2.0));
    assert(matrix_value_equals(root, 0, 0, 2.0));
    assert(matrix_value_equals(root, 0, 1, -3.0));

    numerus_matrix_destroy(quotient);
    numerus_matrix_destroy(root);
}

static void test_zero_divisor_and_invalid_arguments(void)
{
    const double values[] = {1.0, 2.0};
    numerus_matrix *root = NULL;
    numerus_matrix *quotient = NULL;

    assert(numerus_matrix_create_dense(1, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    quotient = root;
    assert(numerus_matrix_create_divide_scalar(root, 0.0, &quotient) ==
        NUMERUS_MATRIX_DIVISION_BY_ZERO);
    assert(quotient == NULL);

    quotient = root;
    assert(numerus_matrix_create_divide_scalar(root, -0.0, &quotient) ==
        NUMERUS_MATRIX_DIVISION_BY_ZERO);
    assert(quotient == NULL);

    assert(numerus_matrix_create_divide_scalar(NULL, 2.0, &quotient) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(quotient == NULL);
    assert(numerus_matrix_create_divide_scalar(root, 2.0, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(root);
}

static void test_non_finite_values(void)
{
    const double values[] = {INFINITY, NAN, 1.0};
    numerus_matrix *root = NULL;
    numerus_matrix *quotient = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_dense(1, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_divide_scalar(root, 2.0, &quotient) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(quotient, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isinf(value) && value > 0.0);
    assert(numerus_matrix_get(quotient, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    assert(matrix_value_equals(quotient, 0, 2, 0.5));

    numerus_matrix_destroy(quotient);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_scalar_division_view();
    test_zero_divisor_and_invalid_arguments();
    test_non_finite_values();
    puts("Matrix scalar division tests passed.");
    return 0;
}
