/**
 * @file matrix_condition_test.c
 * @brief Native tests for Matrix condition estimates and numerically ill-conditioned inputs.
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

static void test_identity_and_conditioned_matrix(void)
{
    const double identity_values[] = {1, 0, 0, 1};
    const double diagonal_values[] = {1, 0, 0, 1e-8};
    numerus_matrix *matrix = NULL;
    double condition = -1.0;

    assert(numerus_matrix_create_dense(2, 2, identity_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_condition_estimate_one(matrix, &condition) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(condition - 1.0) < 1e-12);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, diagonal_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_condition_estimate_one(matrix, &condition) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(condition / 1e8 - 1.0) < 1e-9);
    numerus_matrix_destroy(matrix);
}

static void test_singular_and_invalid_inputs(void)
{
    const double singular_values[] = {1, 2, 2, 4};
    const double nonfinite_values[] = {1, NAN, 0, 1};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *matrix = NULL;
    double condition = 765.25;

    assert(numerus_matrix_condition_estimate_one(NULL, &condition) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(condition == 765.25);

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_condition_estimate_one(matrix, &condition) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(condition == 765.25);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_condition_estimate_one(matrix, &condition) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isinf(condition));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, nonfinite_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_condition_estimate_one(matrix, &condition) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(isinf(condition));
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_identity_and_conditioned_matrix();
    test_singular_and_invalid_inputs();
    puts("Matrix condition estimate tests passed.");
    return 0;
}
