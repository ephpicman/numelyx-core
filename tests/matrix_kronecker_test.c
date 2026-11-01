/**
 * @file matrix_kronecker_test.c
 * @brief Native tests for Kronecker products/sums and checked result dimensions.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
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

static void test_kronecker_layout(void)
{
    const double left_values[] = {1, 2, 3, 4};
    const double right_values[] = {0, 5, 6, 7};
    const double expected[] = {
        0, 5, 0, 10,
        6, 7, 12, 14,
        0, 15, 0, 20,
        18, 21, 24, 28
    };
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *product = NULL;

    assert(numerus_matrix_create_dense(2, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_kronecker_product(left, right, &product) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(product) == 4);
    assert(numerus_matrix_columns(product) == 4);
    assert(numerus_matrix_storage_kind(product) == NUMERUS_STORAGE_DENSE);
    for (size_t row = 0; row < 4; row++) {
        for (size_t column = 0; column < 4; column++) {
            assert(matrix_value_equals(
                product, row, column, expected[row * 4 + column]
            ));
        }
    }

    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
    assert(matrix_value_equals(product, 3, 3, 28.0));
    numerus_matrix_destroy(product);
}

static void test_overflow_and_read_failures(void)
{
    const double root_values[] = {1, 2, 3, 4};
    const double right_values[] = {1, 2};
    numerus_matrix *root = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *huge_left = NULL;
    numerus_matrix *product = NULL;

    assert(numerus_matrix_create_dense(2, 2, root_values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent(
        root, SIZE_MAX, 1, &huge_left
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_kronecker_product(huge_left, right, &product) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(product == NULL);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_kronecker_product(failing, root, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);
    assert(numerus_matrix_kronecker_product(root, failing, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);
    assert(numerus_matrix_kronecker_product(NULL, root, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);
    assert(numerus_matrix_kronecker_product(root, root, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(huge_left);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(root);
}

static void test_non_finite_product(void)
{
    const double zero[] = {0.0};
    const double infinity[] = {INFINITY};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *product = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_dense(1, 1, zero, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 1, infinity, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_kronecker_product(left, right, &product) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(product, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));

    numerus_matrix_destroy(product);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}


static void test_kronecker_sum(void)
{
    const double left_values[] = {1, 2, 3, 4};
    const double right_values[] = {5, 6, 7, 8};
    const double expected[] = {
        6, 6, 2, 0,
        7, 9, 0, 2,
        3, 0, 9, 6,
        0, 3, 7, 12
    };
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *rectangular = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *sum = NULL;

    assert(numerus_matrix_create_dense(2, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_kronecker_sum(left, right, &sum) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(sum) == 4);
    assert(numerus_matrix_columns(sum) == 4);
    assert(numerus_matrix_storage_kind(sum) == NUMERUS_STORAGE_DENSE);
    for (size_t row = 0; row < 4; row++) {
        for (size_t column = 0; column < 4; column++) {
            assert(matrix_value_equals(
                sum, row, column, expected[row * 4 + column]
            ));
        }
    }

    numerus_matrix_destroy(sum);
    sum = NULL;
    assert(numerus_matrix_create_dense(2, 3, rectangular_values, &rectangular) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        left, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_kronecker_sum(failing, right, &sum) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(sum == NULL);
    assert(numerus_matrix_kronecker_sum(rectangular, right, &sum) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(sum == NULL);
    assert(numerus_matrix_kronecker_sum(left, NULL, &sum) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(sum == NULL);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(rectangular);
    numerus_matrix_destroy(sum);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}

int main(void)
{
    test_kronecker_layout();
    test_kronecker_sum();
    test_overflow_and_read_failures();
    test_non_finite_product();
    puts("Matrix Kronecker product tests passed.");
    return 0;
}
