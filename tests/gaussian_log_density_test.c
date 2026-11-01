/**
 * @file gaussian_log_density_test.c
 * @brief Native regression tests for Gaussian log-density values, invalid inputs, and numerical edge cases.
 *
 * @details These native tests define regression coverage for the named Matrix
 * or statistical contract. Assertions are executable specifications: when
 * behavior changes intentionally, update the assertions and the corresponding
 * API documentation together. This file is test support, not runtime code.
 */

#include "../numerus_probability.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static bool fail_next_allocation = false;
static size_t live_probability_allocations = 0;

void *numerus_probability_test_alloc(size_t size)
{
    void *pointer;

    if (fail_next_allocation) {
        fail_next_allocation = false;
        return NULL;
    }

    pointer = malloc(size);
    if (pointer != NULL) {
        live_probability_allocations++;
    }
    return pointer;
}

void numerus_probability_test_free(void *pointer)
{
    if (pointer != NULL) {
        assert(live_probability_allocations > 0);
        live_probability_allocations--;
        free(pointer);
    }
}

static numerus_matrix *make_dense(
    size_t rows,
    size_t columns,
    const double *values
)
{
    numerus_matrix *matrix = NULL;
    assert(numerus_matrix_create_dense(rows, columns, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    return matrix;
}

static void assert_density(
    const double *observation_values,
    const double *mean_values,
    const double *covariance_values,
    size_t dimension,
    double expected
)
{
    numerus_matrix *observation = make_dense(dimension, 1, observation_values);
    numerus_matrix *mean = make_dense(dimension, 1, mean_values);
    numerus_matrix *covariance = make_dense(
        dimension, dimension, covariance_values
    );
    double log_density = 123.0;

    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_SUCCESS);
    assert(fabs(log_density - expected) < 1e-12);

    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
    numerus_matrix_destroy(observation);
}

static void test_univariate_standard_normal(void)
{
    const double covariance_values[] = {1.0};
    const double mean_values[] = {0.0};
    const double at_mean[] = {0.0};
    const double one_sigma[] = {1.0};
    const double log_two_pi = 1.8378770664093454835606594728112352797;

    assert_density(at_mean, mean_values, covariance_values, 1,
        -0.5 * log_two_pi);
    assert_density(one_sigma, mean_values, covariance_values, 1,
        -0.5 * (log_two_pi + 1.0));
}

static void test_multivariate_diagonal_and_correlated_covariance(void)
{
    const double diagonal_covariance[] = {
        4, 0,
        0, 9
    };
    const double correlated_covariance[] = {
        4, 2,
        2, 3
    };
    const double mean[] = {0, 0};
    const double diagonal_observation[] = {2, 3};
    const double correlated_observation[] = {2, 1};
    const double log_two_pi = 1.8378770664093454835606594728112352797;

    assert_density(
        diagonal_observation, mean, diagonal_covariance, 2,
        -0.5 * (2.0 * log_two_pi + log(36.0) + 2.0)
    );
    assert_density(
        correlated_observation, mean, correlated_covariance, 2,
        -0.5 * (2.0 * log_two_pi + log(8.0) + 1.0)
    );
}

static void test_invalid_inputs_preserve_output(void)
{
    const double observation_values[] = {1, 2};
    const double mean_values[] = {0, 0};
    const double covariance_values[] = {
        1, 0,
        0, 0
    };
    const double nonfinite_observation_values[] = {NAN, 2};
    const double nonfinite_covariance_values[] = {
        1, 0,
        0, INFINITY
    };
    const double wrong_mean_values[] = {0};
    const double nonsquare_covariance_values[] = {1, 0, 0, 1, 0, 0};
    numerus_matrix *observation = make_dense(2, 1, observation_values);
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_matrix *nonfinite_observation = make_dense(
        2, 1, nonfinite_observation_values
    );
    numerus_matrix *nonfinite_covariance = make_dense(
        2, 2, nonfinite_covariance_values
    );
    numerus_matrix *wrong_mean = make_dense(1, 1, wrong_mean_values);
    numerus_matrix *nonsquare_covariance = make_dense(
        2, 3, nonsquare_covariance_values
    );
    double log_density = 123.0;

    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_NOT_POSITIVE_DEFINITE);
    assert(log_density == 123.0);

    assert(numerus_multivariate_gaussian_log_density(
        nonfinite_observation, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_NON_FINITE_INPUT);
    assert(log_density == 123.0);

    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, nonfinite_covariance, &log_density
    ) == NUMERUS_PROBABILITY_NON_FINITE_INPUT);
    assert(log_density == 123.0);

    assert(numerus_multivariate_gaussian_log_density(
        observation, wrong_mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_DIMENSION_MISMATCH);
    assert(log_density == 123.0);

    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, nonsquare_covariance, &log_density
    ) == NUMERUS_PROBABILITY_DIMENSION_MISMATCH);
    assert(log_density == 123.0);

    assert(numerus_multivariate_gaussian_log_density(
        NULL, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_INVALID_ARGUMENT);
    assert(log_density == 123.0);
    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, covariance, NULL
    ) == NUMERUS_PROBABILITY_INVALID_ARGUMENT);

    numerus_matrix_destroy(nonsquare_covariance);
    numerus_matrix_destroy(wrong_mean);
    numerus_matrix_destroy(nonfinite_covariance);
    numerus_matrix_destroy(nonfinite_observation);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
    numerus_matrix_destroy(observation);
}

static void test_overflow_and_allocation_failure(void)
{
    const double extreme_observation_values[] = {DBL_MAX};
    const double extreme_mean_values[] = {-DBL_MAX};
    const double covariance_values[] = {1.0};
    numerus_matrix *observation = make_dense(
        1, 1, extreme_observation_values
    );
    numerus_matrix *mean = make_dense(1, 1, extreme_mean_values);
    numerus_matrix *covariance = make_dense(1, 1, covariance_values);
    double log_density = 123.0;
    size_t before = live_probability_allocations;

    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_NUMERICAL_FAILURE);
    assert(log_density == 123.0);

    fail_next_allocation = true;
    assert(numerus_multivariate_gaussian_log_density(
        observation, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_OUT_OF_MEMORY);
    assert(log_density == 123.0);
    assert(live_probability_allocations == before);

    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
    numerus_matrix_destroy(observation);
    assert(live_probability_allocations == 0);
}

static numerus_matrix_status fail_on_observation_row_one(
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
    *parent_column = 0;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_matrix_read_failure_preserves_output_and_allocations(void)
{
    const double observation_values[] = {1.0, 2.0};
    const double mean_values[] = {0.0, 0.0};
    const double covariance_values[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    numerus_matrix *root_observation = make_dense(2, 1, observation_values);
    numerus_matrix *failing_observation = NULL;
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    double log_density = 321.0;
    size_t allocations_before = live_probability_allocations;

    assert(numerus_matrix_create_from_parent_with_transforms(
        root_observation, 2, 1, fail_on_observation_row_one,
        NULL, NULL, &failing_observation
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_multivariate_gaussian_log_density(
        failing_observation, mean, covariance, &log_density
    ) == NUMERUS_PROBABILITY_INVALID_ARGUMENT);
    assert(log_density == 321.0);
    assert(live_probability_allocations == allocations_before);

    numerus_matrix_destroy(failing_observation);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
    numerus_matrix_destroy(root_observation);
    assert(live_probability_allocations == 0);
}

int main(void)
{
    test_univariate_standard_normal();
    test_multivariate_diagonal_and_correlated_covariance();
    test_invalid_inputs_preserve_output();
    test_overflow_and_allocation_failure();
    test_matrix_read_failure_preserves_output_and_allocations();
    puts("Gaussian log-density tests passed.");
    return 0;
}
