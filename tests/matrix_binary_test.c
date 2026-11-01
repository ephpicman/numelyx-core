/**
 * @file matrix_binary_test.c
 * @brief Native regression tests for lazy binary Matrix arithmetic and dimension/error handling.
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

static numerus_matrix_status reject_coordinates(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) row;
    (void) column;
    (void) parent_row;
    (void) parent_column;
    (void) context;

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static void test_addition_and_subtraction(void)
{
    const double left_values[] = {1, 2, 3, 4};
    const double right_values[] = {10, 20, 30, 40};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *sum = NULL;
    numerus_matrix *difference = NULL;
    numerus_matrix *combined = NULL;

    assert(numerus_matrix_create_dense(2, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_add(left, right, &sum) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(sum, 0, 0, 11.0));
    assert(matrix_value_equals(sum, 0, 1, 22.0));
    assert(matrix_value_equals(sum, 1, 0, 33.0));
    assert(matrix_value_equals(sum, 1, 1, 44.0));

    assert(numerus_matrix_create_subtract(left, right, &difference) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(difference, 0, 0, -9.0));
    assert(matrix_value_equals(difference, 0, 1, -18.0));
    assert(matrix_value_equals(difference, 1, 0, -27.0));
    assert(matrix_value_equals(difference, 1, 1, -36.0));

    /* Two-parent nodes compose lazily and do not mutate either parent. */
    assert(numerus_matrix_create_add(sum, difference, &combined) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(combined, 0, 0, 2.0));
    assert(matrix_value_equals(combined, 0, 1, 4.0));
    assert(matrix_value_equals(combined, 1, 0, 6.0));
    assert(matrix_value_equals(combined, 1, 1, 8.0));
    assert(matrix_value_equals(left, 1, 1, 4.0));
    assert(matrix_value_equals(right, 1, 1, 40.0));

    numerus_matrix_destroy(combined);
    numerus_matrix_destroy(difference);
    numerus_matrix_destroy(sum);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}

static void test_validation_and_output_preservation(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *different_shape = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *result = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 4, values, &different_shape) ==
        NUMERUS_MATRIX_SUCCESS);

    result = root;
    assert(numerus_matrix_create_add(root, different_shape, &result) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(result == NULL);

    result = root;
    assert(numerus_matrix_create_subtract(root, NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_add(NULL, root, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_add(root, root, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_add(failing, root, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(result, 0, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 123.0);
    numerus_matrix_destroy(result);

    assert(numerus_matrix_create_subtract(root, failing, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    value = -456.0;
    assert(numerus_matrix_get(result, 0, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == -456.0);

    numerus_matrix_destroy(result);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(different_shape);
    numerus_matrix_destroy(root);
}

static void test_non_finite_arithmetic(void)
{
    const double left_values[] = {INFINITY, NAN};
    const double right_values[] = {-INFINITY, 1.0};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *sum = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_dense(1, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_add(left, right, &sum) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(sum, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    assert(numerus_matrix_get(sum, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));

    numerus_matrix_destroy(sum);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}

int main(void)
{
    test_addition_and_subtraction();
    test_validation_and_output_preservation();
    test_non_finite_arithmetic();
    puts("Matrix binary operation tests passed.");
    return 0;
}
