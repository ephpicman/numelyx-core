/**
 * @file matrix_finite_test.c
 * @brief Native tests for Matrix finite-value predicates and NaN/infinity detection.
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

static numerus_matrix_status fail_on_row_one(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) column;
    (void) context;

    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_finite_matrix(void)
{
    const double values[] = {1.0, -2.5, 0.0, 4.0};
    numerus_matrix *matrix = NULL;
    bool result = false;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_has_nan(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    assert(numerus_matrix_has_infinity(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    assert(numerus_matrix_is_finite(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result);

    numerus_matrix_destroy(matrix);
}

static void test_nan_and_infinity(void)
{
    const double nan_values[] = {1.0, NAN, INFINITY};
    const double infinity_values[] = {-INFINITY, 2.0, 3.0};
    numerus_matrix *matrix = NULL;
    bool result = false;

    assert(numerus_matrix_create_dense(1, 3, nan_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_has_nan(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result);
    assert(numerus_matrix_has_infinity(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result);
    assert(numerus_matrix_is_finite(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(1, 3, infinity_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_has_nan(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    assert(numerus_matrix_has_infinity(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(result);
    assert(numerus_matrix_is_finite(matrix, &result) == NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_read_failure(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    bool result = true;

    assert(numerus_matrix_has_nan(NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numerus_matrix_has_infinity(NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numerus_matrix_is_finite(NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numerus_matrix_is_finite(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    result = true;
    assert(numerus_matrix_has_nan(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numerus_matrix_has_infinity(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numerus_matrix_is_finite(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_finite_matrix();
    test_nan_and_infinity();
    test_validation_and_read_failure();
    puts("Matrix finite-value predicate tests passed.");
    return 0;
}
