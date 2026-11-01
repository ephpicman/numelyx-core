/**
 * @file matrix_log_determinant_test.c
 * @brief Native tests for log-determinant values, determinant sign, and singular cases.
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

static void assert_logdet(
    const double *values,
    size_t size,
    int expected_sign,
    double expected_log_abs,
    double tolerance
)
{
    numerus_matrix *matrix = NULL;
    int sign = 0;
    double log_abs = NAN;

    assert(numerus_matrix_create_dense(size, size, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_log_determinant(
        matrix, &sign, &log_abs
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(sign == expected_sign);
    assert(fabs(log_abs - expected_log_abs) <= tolerance);
    numerus_matrix_destroy(matrix);
}

static void test_identity_and_diagonal(void)
{
    const double identity[] = {1, 0, 0, 1};
    const double positive_diagonal[] = {2, 0, 0, 3};
    const double negative_diagonal[] = {-2, 0, 0, 3};

    assert_logdet(identity, 2, 1, 0.0, 1e-15);
    assert_logdet(positive_diagonal, 2, 1, log(6.0), 1e-14);
    assert_logdet(negative_diagonal, 2, -1, log(6.0), 1e-14);
}

static void test_spd_and_row_permutation(void)
{
    const double spd[] = {4, 2, 2, 3};
    const double permutation[] = {0, 1, 1, 0};

    /* det([[4, 2], [2, 3]]) = 8. */
    assert_logdet(spd, 2, 1, log(8.0), 1e-14);
    assert_logdet(permutation, 2, -1, 0.0, 1e-14);
}

static void test_ill_conditioned_and_extreme_scales(void)
{
    const double ill_conditioned[] = {1, 0, 0, 1e-12};
    const double extreme_scale[] = {1e200, 0, 0, 1e-200};

    assert_logdet(ill_conditioned, 2, 1, log(1e-12), 1e-12);
    /* The product of the diagonal entries is 1, but must not be formed. */
    assert_logdet(extreme_scale, 2, 1, 0.0, 1e-12);
}

static void test_singular_and_invalid_inputs_preserve_outputs(void)
{
    const double singular[] = {1, 0, 0, 0};
    const double nonfinite[] = {1, 0, 0, NAN};
    const double non_square[] = {1, 0, 0, 1, 2, 3};
    numerus_matrix *matrix = NULL;
    int sign = 17;
    double log_abs = 23.0;

    assert(numerus_matrix_create_dense(2, 2, singular, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_log_determinant(
        matrix, &sign, &log_abs
    ) == NUMERUS_MATRIX_SINGULAR);
    assert(sign == 17);
    assert(log_abs == 23.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, nonfinite, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_log_determinant(
        matrix, &sign, &log_abs
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(sign == 17);
    assert(log_abs == 23.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 3, non_square, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_log_determinant(
        matrix, &sign, &log_abs
    ) == NUMERUS_MATRIX_NOT_SQUARE);
    assert(sign == 17);
    assert(log_abs == 23.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_log_determinant(
        NULL, &sign, &log_abs
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(sign == 17);
    assert(log_abs == 23.0);
    assert(numerus_matrix_log_determinant(
        NULL, NULL, &log_abs
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(log_abs == 23.0);
}


static void test_finite_input_with_nonfinite_factorization(void)
{
    const double values[] = {
        1e308, 1e308,
        1e308, -1e308
    };
    numerus_matrix *matrix = NULL;
    int sign = 17;
    double log_abs = 23.0;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_log_determinant(
        matrix, &sign, &log_abs
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(sign == 17);
    assert(log_abs == 23.0);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_identity_and_diagonal();
    test_spd_and_row_permutation();
    test_ill_conditioned_and_extreme_scales();
    test_singular_and_invalid_inputs_preserve_outputs();
    test_finite_input_with_nonfinite_factorization();
    puts("Matrix log-determinant tests passed.");
    return 0;
}
