/**
 * @file matrix_multiply_test.c
 * @brief Native tests for Matrix multiplication, dimensions, and numerical results.
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

static void test_rectangular_product(void)
{
    const double left_values[] = {1, 2, 3, 4, 5, 6};
    const double right_values[] = {7, 8, 9, 10, 11, 12};
    const double expected[] = {58, 64, 139, 154};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *product = NULL;

    assert(numerus_matrix_create_dense(2, 3, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(left, right, &product) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(product) == 2);
    assert(numerus_matrix_columns(product) == 2);
    assert(numerus_matrix_storage_kind(product) == NUMERUS_STORAGE_DENSE);
    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 2; column++) {
            assert(matrix_value_equals(
                product, row, column, expected[row * 2 + column]
            ));
        }
    }

    /* The product is independent of both parents. */
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
    assert(matrix_value_equals(product, 1, 1, 154.0));
    numerus_matrix_destroy(product);
}

static void test_identity_and_zero_products(void)
{
    const double values[] = {1, -2, 3, 4, 5, 6};
    numerus_matrix *matrix = NULL;
    numerus_matrix *identity = NULL;
    numerus_matrix *zero = NULL;
    numerus_matrix *product = NULL;

    assert(numerus_matrix_create_dense(2, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_identity(3, &identity) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(matrix, identity, &product) ==
        NUMERUS_MATRIX_SUCCESS);
    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 3; column++) {
            assert(matrix_value_equals(
                product, row, column, values[row * 3 + column]
            ));
        }
    }
    numerus_matrix_destroy(product);

    assert(numerus_matrix_create_zero(3, 2, &zero) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(matrix, zero, &product) ==
        NUMERUS_MATRIX_SUCCESS);
    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 2; column++) {
            assert(matrix_value_equals(product, row, column, 0.0));
        }
    }

    numerus_matrix_destroy(product);
    numerus_matrix_destroy(zero);
    numerus_matrix_destroy(identity);
    numerus_matrix_destroy(matrix);
}

static void test_shape_overflow_and_read_failures(void)
{
    const double values[] = {1, 2, 3, 4};
    const double wrong_shape_values[] = {1, 2, 3, 4, 5, 6};
    const double right_values[] = {1, 2};
    numerus_matrix *root = NULL;
    numerus_matrix *right_root = NULL;
    numerus_matrix *wrong_shape = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *huge_left = NULL;
    numerus_matrix *product = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 2, right_values, &right_root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 2, wrong_shape_values, &wrong_shape) ==
        NUMERUS_MATRIX_SUCCESS);

    product = root;
    assert(numerus_matrix_multiply(root, wrong_shape, &product) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(product == NULL);
    assert(numerus_matrix_multiply(NULL, root, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);
    assert(numerus_matrix_multiply(root, root, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(failing, root, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);
    assert(numerus_matrix_multiply(root, failing, &product) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(product == NULL);

    assert(numerus_matrix_create_from_parent(
        root, SIZE_MAX, 1, &huge_left
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(huge_left, right_root, &product) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(product == NULL);

    numerus_matrix_destroy(huge_left);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(wrong_shape);
    numerus_matrix_destroy(right_root);
    numerus_matrix_destroy(root);
}

static void test_non_finite_product(void)
{
    const double left_values[] = {0.0, INFINITY};
    const double right_values[] = {INFINITY, 1.0};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *product = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_dense(1, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(left, right, &product) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(product, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));

    numerus_matrix_destroy(product);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}

static void test_safe_shortcuts(void)
{
    const double a[] = {1, 2, 3, 4};
    const double b[] = {INFINITY, 1, 2, 3};
    const double signed_zero[] = {-0.0, 1, 2, 3};
    numerus_matrix *matrix = NULL, *identity = NULL, *zero = NULL;
    numerus_matrix *nonfinite = NULL, *signed_matrix = NULL, *result = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 2, a, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_identity(2, &identity) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_zero(2, 2, &zero) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(matrix, identity, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(result) == NUMERUS_STORAGE_DENSE);
    numerus_matrix_destroy(result); result = NULL;
    assert(numerus_matrix_multiply(identity, matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(result, 1, 1, 4.0));
    numerus_matrix_destroy(result); result = NULL;
    assert(numerus_matrix_multiply(matrix, zero, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(result) == NUMERUS_STORAGE_DENSE);
    numerus_matrix_destroy(result); result = NULL;
    assert(numerus_matrix_multiply(zero, matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(result) == NUMERUS_STORAGE_DENSE);
    assert(matrix_value_equals(result, 0, 1, 0.0));
    numerus_matrix_destroy(result); result = NULL;

    assert(numerus_matrix_create_dense(2, 2, b, &nonfinite) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(nonfinite, identity, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(result, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    numerus_matrix_destroy(result); result = NULL;
    assert(numerus_matrix_multiply(nonfinite, zero, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(result, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    numerus_matrix_destroy(result); result = NULL;

    assert(numerus_matrix_create_dense(2, 2, signed_zero, &signed_matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(signed_matrix, identity, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(result, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(!signbit(value));

    numerus_matrix_destroy(result);
    numerus_matrix_destroy(signed_matrix);
    numerus_matrix_destroy(nonfinite);
    numerus_matrix_destroy(zero);
    numerus_matrix_destroy(identity);
    numerus_matrix_destroy(matrix);
}

static void test_sparse_right_product(void)
{
    const double left_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const double non_finite_values[] = {INFINITY, 2, 3, 4, 5, 6, 7, 8};
    const numerus_storage_sparse_entry entries[] = {
        {0, 2}, {5, 3}, {10, 4}, {15, 5}
    };
    numerus_matrix *left = NULL, *sparse = NULL, *non_finite = NULL;
    numerus_matrix *product = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 4, left_values, &left) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_sparse(4, 4, 0.0, entries, 4, &sparse) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(left, sparse, &product) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(product) == NUMERUS_STORAGE_DENSE);
    assert(matrix_value_equals(product, 0, 0, 2));
    assert(matrix_value_equals(product, 0, 1, 6));
    assert(matrix_value_equals(product, 0, 2, 12));
    assert(matrix_value_equals(product, 0, 3, 20));
    assert(matrix_value_equals(product, 1, 0, 10));
    assert(matrix_value_equals(product, 1, 1, 18));
    assert(matrix_value_equals(product, 1, 2, 28));
    assert(matrix_value_equals(product, 1, 3, 40));
    numerus_matrix_destroy(product); product = NULL;

    assert(numerus_matrix_create_dense(2, 4, non_finite_values, &non_finite) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(non_finite, sparse, &product) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(product, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value)); /* Sparse skipping must not erase 0 × infinity. */

    numerus_matrix_destroy(product);
    numerus_matrix_destroy(non_finite);
    numerus_matrix_destroy(sparse);
    numerus_matrix_destroy(left);
}

int main(void)
{
    test_rectangular_product();
    test_identity_and_zero_products();
    test_shape_overflow_and_read_failures();
    test_non_finite_product();
    test_safe_shortcuts();
    test_sparse_right_product();
    puts("Matrix multiplication tests passed.");
    return 0;
}
