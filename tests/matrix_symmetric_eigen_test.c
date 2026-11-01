/**
 * @file matrix_symmetric_eigen_test.c
 * @brief Native tests for symmetric eigendecomposition and eigenpair accuracy.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static numelyx_core_matrix_status fail_on_row_one(
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

static void assert_eigendecomposition(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *eigenvalues,
    const numelyx_core_matrix *eigenvectors,
    double tolerance
)
{
    size_t size = numelyx_core_matrix_rows(matrix);
    size_t row;
    size_t column;
    double previous = INFINITY;

    assert(numelyx_core_matrix_rows(eigenvalues) == size);
    assert(numelyx_core_matrix_columns(eigenvalues) == 1);
    assert(numelyx_core_matrix_rows(eigenvectors) == size);
    assert(numelyx_core_matrix_columns(eigenvectors) == size);

    for (column = 0; column < size; column++) {
        double value;
        assert(numelyx_core_matrix_get(eigenvalues, column, 0, &value) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(value <= previous + tolerance);
        previous = value;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            size_t k;
            double reconstructed = 0.0;
            double expected;

            for (k = 0; k < size; k++) {
                double left;
                double eigenvalue;
                double right;
                assert(numelyx_core_matrix_get(
                    eigenvectors, row, k, &left
                ) == NUMERUS_MATRIX_SUCCESS);
                assert(numelyx_core_matrix_get(
                    eigenvalues, k, 0, &eigenvalue
                ) == NUMERUS_MATRIX_SUCCESS);
                assert(numelyx_core_matrix_get(
                    eigenvectors, column, k, &right
                ) == NUMERUS_MATRIX_SUCCESS);
                reconstructed += left * eigenvalue * right;
            }

            assert(numelyx_core_matrix_get(
                matrix, row, column, &expected
            ) == NUMERUS_MATRIX_SUCCESS);
            assert(fabs(reconstructed - expected) <=
                tolerance * (1.0 + fabs(expected)));
        }
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            size_t k;
            double dot = 0.0;
            for (k = 0; k < size; k++) {
                double left;
                double right;
                assert(numelyx_core_matrix_get(
                    eigenvectors, k, row, &left
                ) == NUMERUS_MATRIX_SUCCESS);
                assert(numelyx_core_matrix_get(
                    eigenvectors, k, column, &right
                ) == NUMERUS_MATRIX_SUCCESS);
                dot += left * right;
            }
            assert(fabs(dot - (row == column ? 1.0 : 0.0)) < tolerance);
        }
    }
}

static void test_symmetric_positive_and_indefinite(void)
{
    const double positive_values[] = {2, 1, 1, 2};
    const double indefinite_values[] = {1, 2, 2, 1};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *eigenvalues = NULL;
    numelyx_core_matrix *eigenvectors = NULL;
    double value;

    assert(numelyx_core_matrix_create_dense(2, 2, positive_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(eigenvalues, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 3.0) < 1e-10);
    assert(numelyx_core_matrix_get(eigenvalues, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-10);
    assert_eigendecomposition(matrix, eigenvalues, eigenvectors, 1e-9);
    numelyx_core_matrix_destroy(eigenvectors);
    numelyx_core_matrix_destroy(eigenvalues);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, indefinite_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(eigenvalues, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 3.0) < 1e-10);
    assert(numelyx_core_matrix_get(eigenvalues, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 1.0) < 1e-10);
    assert_eigendecomposition(matrix, eigenvalues, eigenvectors, 1e-9);
    numelyx_core_matrix_destroy(eigenvectors);
    numelyx_core_matrix_destroy(eigenvalues);
    numelyx_core_matrix_destroy(matrix);
}

static void test_repeated_and_zero_eigenvalues(void)
{
    const double repeated_values[] = {2, 0, 0, 2};
    const double zero_values[] = {0, 0, 0, 0};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *eigenvalues = NULL;
    numelyx_core_matrix *eigenvectors = NULL;

    assert(numelyx_core_matrix_create_dense(2, 2, repeated_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_SUCCESS);
    assert_eigendecomposition(matrix, eigenvalues, eigenvectors, 1e-9);
    numelyx_core_matrix_destroy(eigenvectors);
    numelyx_core_matrix_destroy(eigenvalues);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_SUCCESS);
    assert_eigendecomposition(matrix, eigenvalues, eigenvectors, 1e-9);
    numelyx_core_matrix_destroy(eigenvectors);
    numelyx_core_matrix_destroy(eigenvalues);
    numelyx_core_matrix_destroy(matrix);
}

static void test_validation_and_read_failures(void)
{
    const double nonsymmetric_values[] = {1, 0, 1, 1};
    const double nonfinite_values[] = {1, NAN, NAN, 1};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *failing = NULL;
    numelyx_core_matrix *eigenvalues = (void *) 1;
    numelyx_core_matrix *eigenvectors = (void *) 1;

    assert(numelyx_core_matrix_symmetric_eigen(
        NULL, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(eigenvalues == NULL && eigenvectors == NULL);

    assert(numelyx_core_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_NOT_SQUARE);
    assert(eigenvalues == NULL && eigenvectors == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, nonsymmetric_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_NOT_SYMMETRIC);
    assert(eigenvalues == NULL && eigenvectors == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(eigenvalues == NULL && eigenvectors == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_eigen(
        failing, &eigenvalues, &eigenvectors
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(eigenvalues == NULL && eigenvectors == NULL);
    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(matrix);
}

int main(void)
{
    test_symmetric_positive_and_indefinite();
    test_repeated_and_zero_eigenvalues();
    test_validation_and_read_failures();
    puts("Matrix symmetric eigen tests passed.");
    return 0;
}
