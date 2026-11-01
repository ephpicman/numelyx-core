/**
 * @file matrix_spectral_test.c
 * @brief Native tests for spectral Matrix routines and eigenvalue-related properties.
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
    if (row == 1) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_spectral_norm(void)
{
    const double diagonal_values[] = {3, 0, 0, 4};
    const double rank_one_values[] = {1, 2, 2, 4};
    numerus_matrix *matrix = NULL;
    double norm = -1.0;

    assert(numerus_matrix_create_dense(2, 2, diagonal_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_spectral_norm(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(norm - 4.0) < 1e-10);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, rank_one_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_spectral_norm(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(norm - 5.0) < 1e-9);
    numerus_matrix_destroy(matrix);
}

static void test_symmetric_spectral_radius(void)
{
    const double values[] = {1, 2, 2, 1};
    numerus_matrix *matrix = NULL;
    double radius = -1.0;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_symmetric_spectral_radius(matrix, &radius) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(radius - 3.0) < 1e-10);
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_failure_outputs(void)
{
    const double nonsymmetric_values[] = {1, 0, 1, 1};
    const double nonfinite_values[] = {1, NAN, 0, 1};
    const double regular_values[] = {1, 0, 0, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *failing = NULL;
    double norm = 42.0;
    double radius = 43.0;

    assert(numerus_matrix_spectral_norm(NULL, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 42.0);
    assert(numerus_matrix_symmetric_spectral_radius(NULL, &radius) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(radius == 43.0);

    assert(numerus_matrix_create_dense(
        2, 2, nonsymmetric_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_symmetric_spectral_radius(matrix, &radius) ==
        NUMERUS_MATRIX_NOT_SYMMETRIC);
    assert(radius == 43.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_spectral_norm(matrix, &norm) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(norm == 42.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, regular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_spectral_norm(failing, &norm) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(norm == 42.0);
    assert(numerus_matrix_symmetric_spectral_radius(failing, &radius) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(radius == 43.0);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_spectral_norm();
    test_symmetric_spectral_radius();
    test_validation_and_failure_outputs();
    puts("Matrix spectral tests passed.");
    return 0;
}
