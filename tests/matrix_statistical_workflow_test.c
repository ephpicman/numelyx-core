/**
 * @file matrix_statistical_workflow_test.c
 * @brief Native integration tests for statistical linear-model workflows built on Matrix primitives.
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

static numelyx_core_matrix *make_dense(
    size_t rows,
    size_t columns,
    const double *values
)
{
    numelyx_core_matrix *matrix = NULL;
    assert(numelyx_core_matrix_create_dense(rows, columns, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    return matrix;
}

static void test_ols_overdetermined_reference(void)
{
    const double design_values[] = {1, 0, 0, 1, 1, 1};
    const double observation_values[] = {1, 2, 4};
    numelyx_core_matrix *design = make_dense(3, 2, design_values);
    numelyx_core_matrix *observations = make_dense(3, 1, observation_values);
    numelyx_core_matrix *coefficients = NULL;
    double first;
    double second;

    assert(numelyx_core_matrix_least_squares(
        design, observations, &coefficients
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 0, 0, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 1, 0, &second) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(first - 4.0 / 3.0) < 1e-12);
    assert(fabs(second - 7.0 / 3.0) < 1e-12);

    numelyx_core_matrix_destroy(coefficients);
    numelyx_core_matrix_destroy(observations);
    numelyx_core_matrix_destroy(design);
}

static void test_ols_ill_scaled_reference(void)
{
    const double design_values[] = {
        1e-2, 0,
        0, 1e2,
        1e-2, 1e2
    };
    const double observation_values[] = {2e-2, 3e2, 3.0002e2};
    numelyx_core_matrix *design = make_dense(3, 2, design_values);
    numelyx_core_matrix *observations = make_dense(3, 1, observation_values);
    numelyx_core_matrix *coefficients = NULL;
    double first;
    double second;

    assert(numelyx_core_matrix_least_squares(
        design, observations, &coefficients
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 0, 0, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 1, 0, &second) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(first - 2.0) < 1e-8);
    assert(fabs(second - 3.0) < 1e-8);

    numelyx_core_matrix_destroy(coefficients);
    numelyx_core_matrix_destroy(observations);
    numelyx_core_matrix_destroy(design);
}

static void test_rank_deficient_ols_fails_without_output(void)
{
    const double design_values[] = {1, 2, 2, 4, 3, 6};
    const double observation_values[] = {1, 2, 3};
    numelyx_core_matrix *design = make_dense(3, 2, design_values);
    numelyx_core_matrix *observations = make_dense(3, 1, observation_values);
    numelyx_core_matrix *coefficients = (void *) 1;

    assert(numelyx_core_matrix_least_squares(
        design, observations, &coefficients
    ) == NUMERUS_MATRIX_RANK_DEFICIENT);
    assert(coefficients == NULL);

    numelyx_core_matrix_destroy(observations);
    numelyx_core_matrix_destroy(design);
}

static void test_wls_zero_and_unequal_weights(void)
{
    const double design_values[] = {1, 0, 0, 1, 1, 1};
    const double observation_values[] = {1, 2, 100};
    const double zero_weight_values[] = {1, 1, 0};
    const double one_column_values[] = {1, 1, 1};
    const double weighted_observations_values[] = {1, 2, 4};
    const double unequal_weights[] = {1, 0.25, 1.0 / 9.0};
    numelyx_core_matrix *design = make_dense(3, 2, design_values);
    numelyx_core_matrix *observations = make_dense(3, 1, observation_values);
    numelyx_core_matrix *coefficients = NULL;
    numelyx_core_matrix *one_column = make_dense(3, 1, one_column_values);
    numelyx_core_matrix *weighted_observations = make_dense(
        3, 1, weighted_observations_values
    );
    numelyx_core_matrix *weighted_coefficients = NULL;
    double first;
    double second;
    double weighted_value;

    assert(numelyx_core_matrix_weighted_least_squares(
        design, observations, zero_weight_values, 3, &coefficients
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 0, 0, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 1, 0, &second) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(first - 1.0) < 1e-12);
    assert(fabs(second - 2.0) < 1e-12);

    assert(numelyx_core_matrix_weighted_least_squares(
        one_column, weighted_observations, unequal_weights, 3,
        &weighted_coefficients
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(
        weighted_coefficients, 0, 0, &weighted_value
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(weighted_value - 10.0 / 7.0) < 1e-12);

    numelyx_core_matrix_destroy(weighted_coefficients);
    numelyx_core_matrix_destroy(coefficients);
    numelyx_core_matrix_destroy(weighted_observations);
    numelyx_core_matrix_destroy(one_column);
    numelyx_core_matrix_destroy(observations);
    numelyx_core_matrix_destroy(design);
}

static void test_gls_whitening_and_covariance_logdet(void)
{
    const double covariance_values[] = {
        1, 0, 0,
        0, 4, 0,
        0, 0, 9
    };
    const double design_values[] = {1, 1, 1};
    const double observation_values[] = {1, 2, 4};
    numelyx_core_matrix *covariance = make_dense(3, 3, covariance_values);
    numelyx_core_matrix *design = make_dense(3, 1, design_values);
    numelyx_core_matrix *observations = make_dense(3, 1, observation_values);
    numelyx_core_matrix *lower = NULL;
    numelyx_core_matrix *whitened_design = NULL;
    numelyx_core_matrix *whitened_observations = NULL;
    numelyx_core_matrix *coefficients = NULL;
    int determinant_sign = 0;
    double log_abs_determinant = NAN;
    double coefficient;

    assert(numelyx_core_matrix_cholesky(covariance, &lower) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_solve_triangular(
        lower, design, true, false, &whitened_design
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_solve_triangular(
        lower, observations, true, false, &whitened_observations
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_least_squares(
        whitened_design, whitened_observations, &coefficients
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(coefficients, 0, 0, &coefficient) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(coefficient - 10.0 / 7.0) < 1e-12);

    assert(numelyx_core_matrix_log_determinant(
        covariance, &determinant_sign, &log_abs_determinant
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant_sign == 1);
    assert(fabs(log_abs_determinant - log(36.0)) < 1e-12);

    numelyx_core_matrix_destroy(coefficients);
    numelyx_core_matrix_destroy(whitened_observations);
    numelyx_core_matrix_destroy(whitened_design);
    numelyx_core_matrix_destroy(lower);
    numelyx_core_matrix_destroy(observations);
    numelyx_core_matrix_destroy(design);
    numelyx_core_matrix_destroy(covariance);
}

static void test_non_spd_covariance_rejected(void)
{
    const double covariance_values[] = {
        1, 0, 0,
        0, 0, 0,
        0, 0, 1
    };
    numelyx_core_matrix *covariance = make_dense(3, 3, covariance_values);
    numelyx_core_matrix *lower = (void *) 1;

    assert(numelyx_core_matrix_cholesky(covariance, &lower) ==
        NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE);
    assert(lower == NULL);
    numelyx_core_matrix_destroy(covariance);
}

int main(void)
{
    test_ols_overdetermined_reference();
    test_ols_ill_scaled_reference();
    test_rank_deficient_ols_fails_without_output();
    test_wls_zero_and_unequal_weights();
    test_gls_whitening_and_covariance_logdet();
    test_non_spd_covariance_rejected();
    puts("Statistical linear-model workflow tests passed.");
    return 0;
}
