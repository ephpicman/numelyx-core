/**
 * @file matrix_hadamard_test.c
 * @brief Native tests for Hadamard elementwise products and dimension validation.
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

static void test_hadamard_product(void)
{
    const double left_values[] = {1.0, -2.0, 0.0, 4.0};
    const double right_values[] = {10.0, 3.0, INFINITY, -0.5};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *product = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_dense(2, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_hadamard_product(left, right, &product) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(matrix_value_equals(product, 0, 0, 10.0));
    assert(matrix_value_equals(product, 0, 1, -6.0));
    assert(numerus_matrix_get(product, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value)); /* IEEE-754: zero multiplied by infinity. */
    assert(matrix_value_equals(product, 1, 1, -2.0));

    /* The operation is lazy and does not modify either parent. */
    assert(matrix_value_equals(left, 0, 1, -2.0));
    assert(matrix_value_equals(right, 1, 1, -0.5));

    numerus_matrix_destroy(product);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}

static void test_hadamard_validation(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    numerus_matrix *square = NULL;
    numerus_matrix *row = NULL;
    numerus_matrix *product = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &square) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 4, values, &row) ==
        NUMERUS_MATRIX_SUCCESS);

    product = square;
    assert(numerus_matrix_create_hadamard_product(square, row, &product) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(product == NULL);
    assert(numerus_matrix_create_hadamard_product(NULL, square, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);
    assert(numerus_matrix_create_hadamard_product(square, square, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(row);
    numerus_matrix_destroy(square);
}

int main(void)
{
    test_hadamard_product();
    test_hadamard_validation();
    puts("Matrix Hadamard tests passed.");
    return 0;
}
