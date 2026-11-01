/**
 * @file statistical_foundation.c
 * @brief Runnable example exercising the statistical foundation API.
 *
 * @details This standalone program demonstrates how to call the statistical foundation's native C APIs. It is illustrative executable code, not part of the extension runtime or a compatibility forwarding header.
 */

#include "../src/matrix/numerus_matrix.h"
#include "../src/core/numerus_numeric.h"
#include "../src/statistics/numerus_optimizer.h"
#include "../src/statistics/numerus_probability.h"
#include "../src/statistics/numerus_rng.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static bool quadratic_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    const double *target = context;
    double dx;
    double dy;

    if (count != 2 || parameters == NULL || target == NULL || value == NULL) {
        return false;
    }

    dx = parameters[0] - target[0];
    dy = parameters[1] - target[1];
    *value = dx * dx + 2.0 * dy * dy;
    return true;
}

static bool quadratic_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    const double *target = context;

    if (count != 2 || parameters == NULL || target == NULL || gradient == NULL) {
        return false;
    }

    gradient[0] = 2.0 * (parameters[0] - target[0]);
    gradient[1] = 4.0 * (parameters[1] - target[1]);
    return true;
}

static int fail(const char *operation, int status)
{
    fprintf(stderr, "%s failed (status %d)\n", operation, status);
    return 1;
}

int main(void)
{
    const double target[] = {1.5, -0.75};
    const double observation_values[] = {1.0, -1.0};
    const double mean_values[] = {0.0, 0.0};
    const double covariance_values[] = {
        2.0, 0.5,
        0.5, 1.0
    };
    double initial_parameters[] = {-3.0, 4.0};
    double sigmoid;
    double log_density;
    double parameter_0;
    double parameter_1;
    double sample_x;
    double sample_y;
    numerus_optimizer_options options;
    numerus_optimizer_result *optimization = NULL;
    numerus_optimizer_termination termination;
    numerus_matrix *observation = NULL;
    numerus_matrix *mean = NULL;
    numerus_matrix *covariance = NULL;
    numerus_matrix *sample = NULL;
    numerus_rng *rng = NULL;
    size_t *indices = NULL;
    int result = 1;

    if (numerus_numeric_sigmoid(0.5, &sigmoid) != NUMERUS_NUMERIC_SUCCESS) {
        return fail("sigmoid", NUMERUS_NUMERIC_NUMERICAL_FAILURE);
    }

    if (numerus_matrix_create_dense(
            2, 1, observation_values, &observation
        ) != NUMERUS_MATRIX_SUCCESS ||
        numerus_matrix_create_dense(2, 1, mean_values, &mean) !=
            NUMERUS_MATRIX_SUCCESS ||
        numerus_matrix_create_dense(2, 2, covariance_values, &covariance) !=
            NUMERUS_MATRIX_SUCCESS) {
        (void) fail("Matrix construction", NUMERUS_MATRIX_OUT_OF_MEMORY);
        goto cleanup;
    }

    if (numerus_multivariate_gaussian_log_density(
            observation, mean, covariance, &log_density
        ) != NUMERUS_PROBABILITY_SUCCESS) {
        (void) fail("Gaussian log density", NUMERUS_PROBABILITY_NUMERICAL_FAILURE);
        goto cleanup;
    }

    if (numerus_rng_create(UINT64_C(20261010), UINT64_C(17), &rng) !=
        NUMERUS_RNG_SUCCESS) {
        (void) fail("RNG construction", NUMERUS_RNG_OUT_OF_MEMORY);
        goto cleanup;
    }

    if (numerus_multivariate_gaussian_sample(
            mean, covariance, rng, &sample
        ) != NUMERUS_PROBABILITY_SUCCESS) {
        (void) fail("Gaussian sampling", NUMERUS_PROBABILITY_NUMERICAL_FAILURE);
        goto cleanup;
    }

    if (numerus_rng_sample_indices(
            rng, 100, 8, false, &indices
        ) != NUMERUS_RNG_SUCCESS) {
        (void) fail("Index sampling", NUMERUS_RNG_NUMERICAL_FAILURE);
        goto cleanup;
    }

    if (numerus_optimizer_default_options(&options) !=
        NUMERUS_OPTIMIZER_SUCCESS ||
        numerus_optimizer_minimize(
            quadratic_objective,
            quadratic_gradient,
            target,
            initial_parameters,
            2,
            &options,
            &optimization
        ) != NUMERUS_OPTIMIZER_SUCCESS) {
        (void) fail("Optimization", NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN);
        goto cleanup;
    }

    if (numerus_optimizer_result_get_parameter(
            optimization, 0, &parameter_0
        ) != NUMERUS_OPTIMIZER_SUCCESS ||
        numerus_optimizer_result_get_parameter(
            optimization, 1, &parameter_1
        ) != NUMERUS_OPTIMIZER_SUCCESS ||
        numerus_optimizer_result_get_termination(
            optimization, &termination
        ) != NUMERUS_OPTIMIZER_SUCCESS ||
        termination != NUMERUS_OPTIMIZER_CONVERGED_GRADIENT ||
        fabs(parameter_0 - target[0]) > 1e-5 ||
        fabs(parameter_1 - target[1]) > 1e-5) {
        (void) fail("Optimization result", NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN);
        goto cleanup;
    }

    if (numerus_matrix_get(sample, 0, 0, &sample_x) !=
            NUMERUS_MATRIX_SUCCESS ||
        numerus_matrix_get(sample, 1, 0, &sample_y) !=
            NUMERUS_MATRIX_SUCCESS) {
        (void) fail("Sample access", NUMERUS_MATRIX_INVALID_ARGUMENT);
        goto cleanup;
    }

    printf("sigmoid(0.5) = %.8f\\n", sigmoid);
    printf("Gaussian log density = %.8f\\n", log_density);
    printf("sample = [%.8f, %.8f]\\n", sample_x, sample_y);
    printf("sampled 8 unique indices from [0, 100)\n");
    printf("optimizer minimizer = [%.6f, %.6f]\n", parameter_0, parameter_1);
    result = 0;

cleanup:
    numerus_optimizer_result_destroy(optimization);
    numerus_rng_free_indices(indices);
    numerus_rng_destroy(rng);
    numerus_matrix_destroy(sample);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
    numerus_matrix_destroy(observation);
    return result;
}
