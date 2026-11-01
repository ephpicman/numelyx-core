/**
 * @file gaussian_sampling_test.c
 * @brief Native regression tests for Gaussian random sampling, reproducibility, and output validity.
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

static long allocations_before_failure = -1;
static size_t live_probability_allocations = 0;

void *numerus_probability_test_alloc(size_t size)
{
    void *pointer;

    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        return NULL;
    }
    if (allocations_before_failure > 0) {
        allocations_before_failure--;
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

static void test_fixed_state_reference_and_raw_consumption(void)
{
    const double mean_values[] = {10.0, -2.0};
    const double covariance_values[] = {
        4, 0,
        0, 9
    };
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_matrix *sample = NULL;
    numerus_rng *rng = NULL;
    numerus_rng *clone = NULL;
    double first;
    double second;
    uint32_t ignored;
    uint32_t source_next;
    uint32_t clone_next;
    size_t index;

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);
    assert(numerus_multivariate_gaussian_sample(
        mean, covariance, rng, &sample
    ) == NUMERUS_PROBABILITY_SUCCESS);
    assert(sample != NULL);

    assert(numerus_matrix_get(sample, 0, 0, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(sample, 1, 0, &second) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(first - 9.72337267075938) < 1e-12);
    assert(fabs(second - (-2.012554711056754)) < 1e-12);

    /* Two normal variates consume exactly eight raw outputs. */
    for (index = 0; index < 8; index++) {
        assert(numerus_rng_next_u32(clone, &ignored) == NUMERUS_RNG_SUCCESS);
    }
    assert(numerus_rng_next_u32(rng, &source_next) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &clone_next) == NUMERUS_RNG_SUCCESS);
    assert(source_next == clone_next);

    numerus_matrix_destroy(sample);
    numerus_rng_destroy(clone);
    numerus_rng_destroy(rng);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
}

static void test_repeatability_and_invalid_inputs(void)
{
    const double mean_values[] = {1.0, -2.0};
    const double covariance_values[] = {
        4, 1,
        1, 3
    };
    const double nonfinite_mean_values[] = {NAN, 0.0};
    const double non_spd_values[] = {
        1, 2,
        2, 1
    };
    const double wrong_mean_values[] = {0.0};
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_matrix *nonfinite_mean = make_dense(
        2, 1, nonfinite_mean_values
    );
    numerus_matrix *non_spd = make_dense(2, 2, non_spd_values);
    numerus_matrix *wrong_mean = make_dense(1, 1, wrong_mean_values);
    numerus_matrix *first_sample = NULL;
    numerus_matrix *second_sample = NULL;
    numerus_rng *first_rng = NULL;
    numerus_rng *second_rng = NULL;
    double first_value;
    double second_value;

    assert(numerus_rng_create(100, 9, &first_rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_create(100, 9, &second_rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_multivariate_gaussian_sample(
        mean, covariance, first_rng, &first_sample
    ) == NUMERUS_PROBABILITY_SUCCESS);
    assert(numerus_multivariate_gaussian_sample(
        mean, covariance, second_rng, &second_sample
    ) == NUMERUS_PROBABILITY_SUCCESS);
    assert(numerus_matrix_get(first_sample, 0, 0, &first_value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(second_sample, 0, 0, &second_value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(first_value == second_value);

    numerus_matrix_destroy(first_sample);
    numerus_matrix_destroy(second_sample);
    first_sample = (void *) 1;

    assert(numerus_multivariate_gaussian_sample(
        mean, non_spd, first_rng, &first_sample
    ) == NUMERUS_PROBABILITY_NOT_POSITIVE_DEFINITE);
    assert(first_sample == NULL);

    assert(numerus_multivariate_gaussian_sample(
        nonfinite_mean, covariance, first_rng, &first_sample
    ) == NUMERUS_PROBABILITY_NON_FINITE_INPUT);
    assert(first_sample == NULL);

    assert(numerus_multivariate_gaussian_sample(
        wrong_mean, covariance, first_rng, &first_sample
    ) == NUMERUS_PROBABILITY_DIMENSION_MISMATCH);
    assert(first_sample == NULL);

    assert(numerus_multivariate_gaussian_sample(
        mean, covariance, NULL, &first_sample
    ) == NUMERUS_PROBABILITY_INVALID_ARGUMENT);
    assert(first_sample == NULL);
    assert(numerus_multivariate_gaussian_sample(
        mean, covariance, first_rng, NULL
    ) == NUMERUS_PROBABILITY_INVALID_ARGUMENT);

    numerus_rng_destroy(second_rng);
    numerus_rng_destroy(first_rng);
    numerus_matrix_destroy(wrong_mean);
    numerus_matrix_destroy(non_spd);
    numerus_matrix_destroy(nonfinite_mean);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
}

static void test_empirical_mean_and_covariance(void)
{
    const double mean_values[] = {1.0, -2.0};
    const double covariance_values[] = {
        4, 1,
        1, 3
    };
    const size_t sample_count = 3000;
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_rng *rng = NULL;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x_squared = 0.0;
    double sum_y_squared = 0.0;
    double sum_xy = 0.0;
    size_t index;

    assert(numerus_rng_create(2026, 17, &rng) == NUMERUS_RNG_SUCCESS);

    for (index = 0; index < sample_count; index++) {
        numerus_matrix *sample = NULL;
        double x;
        double y;

        assert(numerus_multivariate_gaussian_sample(
            mean, covariance, rng, &sample
        ) == NUMERUS_PROBABILITY_SUCCESS);
        assert(numerus_matrix_get(sample, 0, 0, &x) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_get(sample, 1, 0, &y) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(isfinite(x) && isfinite(y));

        sum_x += x;
        sum_y += y;
        sum_x_squared += x * x;
        sum_y_squared += y * y;
        sum_xy += x * y;
        numerus_matrix_destroy(sample);
    }

    {
        double average_x = sum_x / (double) sample_count;
        double average_y = sum_y / (double) sample_count;
        double variance_x = (
            sum_x_squared - sum_x * sum_x / (double) sample_count
        ) / (double) (sample_count - 1);
        double variance_y = (
            sum_y_squared - sum_y * sum_y / (double) sample_count
        ) / (double) (sample_count - 1);
        double covariance_xy = (
            sum_xy - sum_x * sum_y / (double) sample_count
        ) / (double) (sample_count - 1);

        assert(fabs(average_x - 1.0) < 0.15);
        assert(fabs(average_y + 2.0) < 0.15);
        assert(variance_x > 3.5 && variance_x < 4.5);
        assert(variance_y > 2.6 && variance_y < 3.4);
        assert(covariance_xy > 0.7 && covariance_xy < 1.3);
    }

    numerus_rng_destroy(rng);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
}


static void test_allocation_failures_do_not_advance_rng(void)
{
    const double mean_values[] = {0.0, 0.0};
    const double covariance_values[] = {
        1, 0,
        0, 1
    };
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_rng *rng = NULL;
    numerus_rng *clone = NULL;
    uint32_t first;
    uint32_t second;
    long fail_after;

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);

    /* Both probability-owned buffers must be failure-safe independently. */
    for (fail_after = 0; fail_after < 2; fail_after++) {
        numerus_matrix *sample = (void *) 1;

        allocations_before_failure = fail_after;
        assert(numerus_multivariate_gaussian_sample(
            mean, covariance, rng, &sample
        ) == NUMERUS_PROBABILITY_OUT_OF_MEMORY);
        assert(sample == NULL);
        assert(allocations_before_failure == -1);
        assert(live_probability_allocations == 0);
    }

    assert(numerus_rng_next_u32(rng, &first) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &second) == NUMERUS_RNG_SUCCESS);
    assert(first == second);

    numerus_rng_destroy(clone);
    numerus_rng_destroy(rng);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
    assert(live_probability_allocations == 0);
}

static numerus_matrix_status fail_on_mean_row_one(
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

static void test_matrix_read_failure_preserves_rng_and_output(void)
{
    const double mean_values[] = {1.0, 2.0};
    const double covariance_values[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    numerus_matrix *root_mean = make_dense(2, 1, mean_values);
    numerus_matrix *failing_mean = NULL;
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_matrix *sample = (numerus_matrix *) 1;
    numerus_rng *rng = NULL;
    numerus_rng *clone = NULL;
    uint32_t actual_next;
    uint32_t expected_next;

    assert(numerus_matrix_create_from_parent_with_transforms(
        root_mean, 2, 1, fail_on_mean_row_one, NULL, NULL, &failing_mean
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_rng_create(73, 11, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);

    assert(numerus_multivariate_gaussian_sample(
        failing_mean, covariance, rng, &sample
    ) == NUMERUS_PROBABILITY_INVALID_ARGUMENT);
    assert(sample == NULL);

    assert(numerus_rng_next_u32(rng, &actual_next) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &expected_next) == NUMERUS_RNG_SUCCESS);
    assert(actual_next == expected_next);

    numerus_rng_destroy(clone);
    numerus_rng_destroy(rng);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(failing_mean);
    numerus_matrix_destroy(root_mean);
    assert(live_probability_allocations == 0);
}

int main(void)
{
    test_fixed_state_reference_and_raw_consumption();
    test_repeatability_and_invalid_inputs();
    test_empirical_mean_and_covariance();
    test_allocation_failures_do_not_advance_rng();
    test_matrix_read_failure_preserves_rng_and_output();
    puts("Gaussian sampling tests passed.");
    return 0;
}
