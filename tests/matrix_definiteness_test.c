/**
 * @file matrix_definiteness_test.c
 * @brief Native tests for positive-definite and positive-semidefinite Matrix classification.
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

static void check(const double *values, size_t rows, size_t columns, int pd, int psd)
{
    numerus_matrix *matrix = NULL;
    int actual_pd = -1, actual_psd = -1;
    assert(numerus_matrix_create_dense(rows, columns, values, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_definiteness(matrix, &actual_pd, &actual_psd) == NUMERUS_MATRIX_SUCCESS);
    assert(actual_pd == pd);
    assert(actual_psd == psd);
    numerus_matrix_destroy(matrix);
}

static void test_classes(void)
{
    const double pd[] = {2.0, 0.0, 0.0, 1.0};
    const double tiny_pd[] = {2e-200, 0.0, 0.0, 1e-200};
    const double psd[] = {1.0, 1.0, 1.0, 1.0};
    const double zero[] = {0.0, 0.0, 0.0, 0.0};
    const double indefinite[] = {1.0, 2.0, 2.0, 1.0};
    const double near_zero_negative[] = {1.0, 0.0, 0.0, -1e-16};
    check(pd, 2, 2, 1, 1);
    check(tiny_pd, 2, 2, 1, 1);
    check(psd, 2, 2, 0, 1);
    check(zero, 2, 2, 0, 1);
    check(indefinite, 2, 2, 0, 0);
    check(near_zero_negative, 2, 2, 0, 1);
}

static void test_errors_preserve_outputs(void)
{
    const double nonsymmetric[] = {1.0, 1.0, 0.0, 1.0};
    const double rectangular[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double nonfinite[] = {1.0, NAN, NAN, 1.0};
    numerus_matrix *matrix = NULL;
    int pd = 7, psd = 8;

    assert(numerus_matrix_classify_definiteness(NULL, &pd, &psd) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(pd == 7 && psd == 8);
    assert(numerus_matrix_create_dense(2, 3, rectangular, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_definiteness(matrix, &pd, &psd) == NUMERUS_MATRIX_NOT_SQUARE);
    assert(pd == 7 && psd == 8);
    numerus_matrix_destroy(matrix);
    assert(numerus_matrix_create_dense(2, 2, nonsymmetric, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_definiteness(matrix, &pd, &psd) == NUMERUS_MATRIX_NOT_SYMMETRIC);
    assert(pd == 7 && psd == 8);
    numerus_matrix_destroy(matrix);
    assert(numerus_matrix_create_dense(2, 2, nonfinite, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_definiteness(matrix, &pd, &psd) == NUMERUS_MATRIX_NON_FINITE);
    assert(pd == 7 && psd == 8);
    numerus_matrix_destroy(matrix);
    assert(numerus_matrix_classify_definiteness(NULL, &pd, &pd) == NUMERUS_MATRIX_INVALID_ARGUMENT);
}

int main(void)
{
    test_classes();
    test_errors_preserve_outputs();
    puts("Matrix definiteness classification tests passed.");
    return 0;
}
