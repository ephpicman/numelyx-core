/**
 * @file matrix_polynomial_test.c
 * @brief Native tests for Matrix polynomial evaluation and coefficient ordering.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"
#include <assert.h>
#include <stdio.h>

static numerus_matrix_status reject_coordinates(size_t row, size_t column,
    size_t *parent_row, size_t *parent_column, const void *context)
{
    (void) row; (void) column; (void) parent_row; (void) parent_column;
    (void) context;
    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static void assert_value(const numerus_matrix *matrix, size_t row,
    size_t column, double expected)
{
    double actual = 0.0;
    assert(numerus_matrix_get(matrix, row, column, &actual) == NUMERUS_MATRIX_SUCCESS);
    assert(actual == expected);
}

int main(void)
{
    const double a[] = {1, 2, 0, 3};
    const double coefficients[] = {2, 3, 4};
    const double constant[] = {5};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    const double linear[] = {1, 1};
    numerus_matrix *base = NULL, *result = NULL, *rectangular = NULL;
    numerus_matrix *failing = NULL;

    assert(numerus_matrix_create_dense(2, 2, a, &base) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_polynomial(base, coefficients, 3, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(result) == 2 && numerus_matrix_columns(result) == 2);
    assert_value(result, 0, 0, 9);
    assert_value(result, 0, 1, 38);
    assert_value(result, 1, 0, 0);
    assert_value(result, 1, 1, 47);
    numerus_matrix_destroy(result);
    result = NULL;

    assert(numerus_matrix_polynomial(base, constant, 1, &result) == NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 5);
    assert_value(result, 0, 1, 0);
    assert_value(result, 1, 0, 0);
    assert_value(result, 1, 1, 5);
    numerus_matrix_destroy(result);
    result = NULL;

    assert(numerus_matrix_polynomial(base, coefficients, 0, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_dense(2, 3, rectangular_values, &rectangular) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_polynomial(rectangular, coefficients, 3, &result) == NUMERUS_MATRIX_NOT_SQUARE);
    assert(result == NULL);

    assert(numerus_matrix_create_from_parent_with_transforms(
        base, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_polynomial(failing, linear, 2, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_polynomial(NULL, coefficients, 3, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_polynomial(base, coefficients, 3, NULL) == NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(rectangular);
    numerus_matrix_destroy(base);
    puts("Matrix polynomial tests passed.");
    return 0;
}
