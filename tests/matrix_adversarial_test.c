/**
 * @file matrix_adversarial_test.c
 * @brief Adversarial Matrix tests for extreme magnitudes, non-finite values, and numerical failure paths.
 *
 * @details These native tests define regression coverage for the named Matrix
 * or statistical contract. Assertions are executable specifications: when
 * behavior changes intentionally, update the assertions and the corresponding
 * API documentation together. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static numerus_matrix_status fail_value_read(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    (void) row;
    (void) column;
    (void) parent_value;
    (void) result;
    (void) context;
    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static void test_subnormal_round_trip(void)
{
    const double smallest_positive = nextafter(0.0, 1.0);
    const double values[] = {smallest_positive, -smallest_positive};
    numerus_matrix *matrix = NULL;
    double actual = 0.0;

    assert(smallest_positive > 0.0);
    assert(numerus_matrix_create_dense(1, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(matrix, 0, 0, &actual) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(actual == smallest_positive);
    assert(numerus_matrix_get(matrix, 0, 1, &actual) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(actual == -smallest_positive);

    numerus_matrix_destroy(matrix);
}

static void test_large_frobenius_norm(void)
{
    const double value = DBL_MAX / 4.0;
    const double values[] = {value, value};
    numerus_matrix *matrix = NULL;
    double norm = -1.0;
    double expected = hypot(value, value);

    assert(numerus_matrix_create_dense(1, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_norm_frobenius(matrix, &norm) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isfinite(norm));
    assert(fabs(norm - expected) / expected < 1e-15);

    numerus_matrix_destroy(matrix);
}

static void test_callback_failure_preserves_scalar_output(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    double sum = 987.0;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, NULL, fail_value_read, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_sum(failing, &sum) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(sum == 987.0);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

static void test_nonfinite_policy_is_operation_specific(void)
{
    const double values[] = {INFINITY, 1.0};
    numerus_matrix *matrix = NULL;
    double sum = 0.0;
    double variance = 123.0;

    assert(numerus_matrix_create_dense(1, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_sum(matrix, &sum) == NUMERUS_MATRIX_SUCCESS);
    assert(isinf(sum));
    assert(numerus_matrix_variance(matrix, false, &variance) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(variance == 123.0);

    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_subnormal_round_trip();
    test_large_frobenius_norm();
    test_callback_failure_preserves_scalar_output();
    test_nonfinite_policy_is_operation_specific();
    puts("Matrix adversarial numerical tests passed.");
    return 0;
}
