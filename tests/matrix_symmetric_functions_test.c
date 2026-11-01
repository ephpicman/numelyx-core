/**
 * @file matrix_symmetric_functions_test.c
 * @brief Native tests for Matrix functions implemented through symmetric eigendecomposition.
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

static void assert_value(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    double expected,
    double tolerance
)
{
    double actual;
    assert(numelyx_core_matrix_get(matrix, row, column, &actual) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(actual - expected) <= tolerance);
}

static void test_square_root_and_logarithm(void)
{
    const double psd[] = {4.0, 0.0, 0.0, 0.0};
    const double spd[] = {exp(2.0), 0.0, 0.0, exp(-1.0)};
    const double negative[] = {1.0, 0.0, 0.0, -1.0};
    const double singular[] = {1.0, 0.0, 0.0, 0.0};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *result = NULL;

    assert(numelyx_core_matrix_create_dense(2, 2, psd, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_square_root(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 2.0, 1e-12);
    assert_value(result, 1, 1, 0.0, 1e-12);
    numelyx_core_matrix_destroy(result);
    assert(numelyx_core_matrix_symmetric_logarithm(matrix, &result) ==
        NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE);
    assert(result == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, spd, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_logarithm(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(result, 0, 0, 2.0, 1e-10);
    assert_value(result, 1, 1, -1.0, 1e-10);
    numelyx_core_matrix_destroy(result);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, negative, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_square_root(matrix, &result) ==
        NUMERUS_MATRIX_NOT_POSITIVE_SEMIDEFINITE);
    assert(result == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, singular, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_logarithm(matrix, &result) ==
        NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE);
    assert(result == NULL);
    numelyx_core_matrix_destroy(matrix);
}

static void test_sine_and_cosine(void)
{
    const double diagonal[] = {1.57079632679489661923, 0.0, 0.0, 3.14159265358979323846};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *sine = NULL;
    numelyx_core_matrix *cosine = NULL;

    assert(numelyx_core_matrix_create_dense(2, 2, diagonal, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_sine(matrix, &sine) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_cosine(matrix, &cosine) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(sine, 0, 0, 1.0, 1e-12);
    assert_value(sine, 1, 1, 0.0, 1e-12);
    assert_value(cosine, 0, 0, 0.0, 1e-12);
    assert_value(cosine, 1, 1, -1.0, 1e-12);

    numelyx_core_matrix_destroy(sine);
    numelyx_core_matrix_destroy(cosine);
    numelyx_core_matrix_destroy(matrix);
}

static void test_symmetric_domain_and_reconstruction(void)
{
    const double symmetric[] = {2.0, 1.0, 1.0, 2.0};
    const double nonsymmetric[] = {1.0, 2.0, 0.0, 1.0};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *square = NULL;
    numelyx_core_matrix *bad = NULL;

    assert(numelyx_core_matrix_create_dense(2, 2, symmetric, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_square_root(matrix, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_multiply(root, root, &square) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(square, 0, 0, 2.0, 1e-9);
    assert_value(square, 0, 1, 1.0, 1e-9);
    assert_value(square, 1, 0, 1.0, 1e-9);
    assert_value(square, 1, 1, 2.0, 1e-9);
    numelyx_core_matrix_destroy(square);
    numelyx_core_matrix_destroy(root);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, nonsymmetric, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_symmetric_sine(matrix, &bad) ==
        NUMERUS_MATRIX_NOT_SYMMETRIC);
    assert(bad == NULL);
    numelyx_core_matrix_destroy(matrix);
}

int main(void)
{
    test_square_root_and_logarithm();
    test_sine_and_cosine();
    test_symmetric_domain_and_reconstruction();
    puts("Symmetric matrix function tests passed.");
    return 0;
}
