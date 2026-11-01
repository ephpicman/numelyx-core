/**
 * @file matrix_norm_test.c
 * @brief Native tests for Matrix norms and norm edge cases.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <float.h>
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

static void test_norm_values(void)
{
    const double values[] = {3, 4, 0, -12};
    numerus_matrix *matrix = NULL;
    double norm = -1.0;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_norm_frobenius(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(norm == 13.0);
    assert(numerus_matrix_norm_one(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(norm == 16.0);
    assert(numerus_matrix_norm_infinity(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(norm == 12.0);

    numerus_matrix_destroy(matrix);
}

static void test_frobenius_stability_and_nonfinite_values(void)
{
    const double large_values[] = {DBL_MAX / 2.0, DBL_MAX / 2.0};
    const double small_values[] = {1e-300, 1e-300};
    const double nan_values[] = {INFINITY, NAN};
    numerus_matrix *matrix = NULL;
    double norm = 0.0;

    assert(numerus_matrix_create_dense(1, 2, large_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_norm_frobenius(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isfinite(norm));
    assert(norm > DBL_MAX / 2.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(1, 2, small_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_norm_frobenius(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(norm > 1e-300);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(1, 2, nan_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_norm_frobenius(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(norm));
    assert(numerus_matrix_norm_one(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(norm));
    assert(numerus_matrix_norm_infinity(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(norm));
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_read_failure(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    double norm = 765.25;

    assert(numerus_matrix_norm_frobenius(NULL, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 765.25);
    assert(numerus_matrix_norm_one(NULL, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 765.25);
    assert(numerus_matrix_norm_infinity(NULL, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 765.25);
    assert(numerus_matrix_norm_frobenius(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_norm_frobenius(failing, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 765.25);
    assert(numerus_matrix_norm_one(failing, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 765.25);
    assert(numerus_matrix_norm_infinity(failing, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 765.25);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_norm_values();
    test_frobenius_stability_and_nonfinite_values();
    test_validation_and_read_failure();
    puts("Matrix norm tests passed.");
    return 0;
}
