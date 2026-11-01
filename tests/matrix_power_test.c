/**
 * @file matrix_power_test.c
 * @brief Native tests for non-negative and signed integer Matrix powers.
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

static void assert_2x2(
    const numerus_matrix *matrix,
    double a,
    double b,
    double c,
    double d
)
{
    assert(numerus_matrix_rows(matrix) == 2);
    assert(numerus_matrix_columns(matrix) == 2);
    assert(matrix_value_equals(matrix, 0, 0, a));
    assert(matrix_value_equals(matrix, 0, 1, b));
    assert(matrix_value_equals(matrix, 1, 0, c));
    assert(matrix_value_equals(matrix, 1, 1, d));
}

static void test_power_exponents(void)
{
    const double values[] = {1, 1, 1, 0};
    numerus_matrix *base = NULL;
    numerus_matrix *power = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &base) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_power(base, 0, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(power) == NUMERUS_STORAGE_IDENTITY);
    assert_2x2(power, 1, 0, 0, 1);
    numerus_matrix_destroy(power);

    assert(numerus_matrix_power(base, 1, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(power) == NUMERUS_STORAGE_DENSE);
    assert_2x2(power, 1, 1, 1, 0);
    numerus_matrix_destroy(power);

    assert(numerus_matrix_power(base, 2, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_2x2(power, 2, 1, 1, 1);
    numerus_matrix_destroy(power);

    assert(numerus_matrix_power(base, 3, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_2x2(power, 3, 2, 2, 1);
    numerus_matrix_destroy(power);

    assert(numerus_matrix_power(base, 5, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_2x2(power, 8, 5, 5, 3);
    numerus_matrix_destroy(power);

    /* Power results do not mutate their base. */
    assert_2x2(base, 1, 1, 1, 0);
    numerus_matrix_destroy(base);
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

static void test_invalid_shapes_and_read_failures(void)
{
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    const double square_values[] = {1, 0, 0, 1};
    numerus_matrix *rectangular = NULL;
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *power = NULL;

    assert(numerus_matrix_create_dense(2, 3, rectangular_values, &rectangular) ==
        NUMERUS_MATRIX_SUCCESS);
    power = rectangular;
    assert(numerus_matrix_power(rectangular, 2, &power) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(power == NULL);
    assert(numerus_matrix_power(NULL, 2, &power) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(power == NULL);
    assert(numerus_matrix_power(rectangular, 1, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(2, 2, square_values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_power(failing, 1, &power) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(power == NULL);
    assert(numerus_matrix_power(failing, 2, &power) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(power == NULL);

    /* A^0 depends only on the square shape, not on reading A's values. */
    assert(numerus_matrix_power(failing, 0, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_2x2(power, 1, 0, 0, 1);

    numerus_matrix_destroy(power);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
    numerus_matrix_destroy(rectangular);
}

static void test_signed_power(void)
{
    const double values[] = {4, 7, 2, 6};
    const double singular_values[] = {1, 2, 2, 4};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *base = NULL;
    numerus_matrix *power = NULL;
    numerus_matrix *product = NULL;
    numerus_matrix *inverse_once = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 2, values, &base) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_power_signed(base, -1, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(power) == 2);
    assert(numerus_matrix_columns(power) == 2);
    assert(numerus_matrix_get(power, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 0.6) < 1e-12);
    assert(numerus_matrix_get(power, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 0.7) < 1e-12);
    inverse_once = power;
    power = NULL;
    assert(numerus_matrix_multiply(base, inverse_once, &product) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(product, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-12);
    assert(numerus_matrix_get(product, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value) < 1e-12);
    assert(numerus_matrix_get(product, 1, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value) < 1e-12);
    assert(numerus_matrix_get(product, 1, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-12);
    numerus_matrix_destroy(product);
    product = NULL;

    assert(numerus_matrix_power_signed(base, -2, &power) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(base, power, &product) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(product, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 0.6) < 1e-12);
    assert(numerus_matrix_get(product, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 0.7) < 1e-12);
    numerus_matrix_destroy(product);
    numerus_matrix_destroy(power);
    numerus_matrix_destroy(inverse_once);
    numerus_matrix_destroy(base);

    assert(numerus_matrix_create_dense(
        2, 2, singular_values, &base
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_power_signed(base, -1, &power) ==
        NUMERUS_MATRIX_SINGULAR);
    assert(power == NULL);
    numerus_matrix_destroy(base);

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &base
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_power_signed(base, -1, &power) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(power == NULL);
    numerus_matrix_destroy(base);

    assert(numerus_matrix_power_signed(NULL, -1, &power) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(power == NULL);
    assert(numerus_matrix_power_signed(NULL, 0, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
}

int main(void)
{
    test_power_exponents();
    test_signed_power();
    test_invalid_shapes_and_read_failures();
    puts("Matrix power tests passed.");
    return 0;
}
