/**
 * @file numerus_probability.c
 * @brief Probability distributions and random variate generation.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#include "numerus_probability.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#if defined(NUMERUS_PROBABILITY_TEST_ALLOCATOR)
void *numerus_probability_test_alloc(size_t size);
void numerus_probability_test_free(void *pointer);
# define numerus_probability_alloc(size) numerus_probability_test_alloc(size)
# define numerus_probability_free(pointer) numerus_probability_test_free(pointer)
#elif !defined(NUMERUS_PROBABILITY_USE_LIBC_ALLOC)
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_probability_alloc(size) emalloc(size)
# define numerus_probability_free(pointer) efree(pointer)
#else
# define numerus_probability_alloc(size) malloc(size)
# define numerus_probability_free(pointer) free(pointer)
#endif

static numerus_probability_status numerus_probability_map_matrix_status(
    numerus_matrix_status status
)
{
    switch (status) {
        case NUMERUS_MATRIX_SUCCESS:
            return NUMERUS_PROBABILITY_SUCCESS;
        case NUMERUS_MATRIX_INVALID_ARGUMENT:
            return NUMERUS_PROBABILITY_INVALID_ARGUMENT;
        case NUMERUS_MATRIX_DIMENSION_MISMATCH:
        case NUMERUS_MATRIX_NOT_SQUARE:
            return NUMERUS_PROBABILITY_DIMENSION_MISMATCH;
        case NUMERUS_MATRIX_NON_FINITE:
            return NUMERUS_PROBABILITY_NON_FINITE_INPUT;
        case NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE:
        case NUMERUS_MATRIX_SINGULAR:
            return NUMERUS_PROBABILITY_NOT_POSITIVE_DEFINITE;
        case NUMERUS_MATRIX_OVERFLOW:
            return NUMERUS_PROBABILITY_SIZE_OVERFLOW;
        case NUMERUS_MATRIX_OUT_OF_MEMORY:
            return NUMERUS_PROBABILITY_OUT_OF_MEMORY;
        default:
            return NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
    }
}

numerus_probability_status numerus_multivariate_gaussian_log_density(
    const numerus_matrix *observation,
    const numerus_matrix *mean,
    const numerus_matrix *covariance,
    double *log_density
)
{
    const double log_two_pi = 1.8378770664093454835606594728112352797;
    size_t dimension;
    size_t residual_bytes;
    size_t index;
    double *residual_values = NULL;
    numerus_matrix *residual = NULL;
    numerus_matrix *lower = NULL;
    numerus_matrix *whitened_residual = NULL;
    int determinant_sign = 0;
    double log_abs_determinant = 0.0;
    double quadratic_form = 0.0;
    double result;
    numerus_matrix_status matrix_status;
    numerus_probability_status status;

    if (observation == NULL || mean == NULL || covariance == NULL ||
        log_density == NULL) {
        return NUMERUS_PROBABILITY_INVALID_ARGUMENT;
    }

    dimension = numerus_matrix_rows(observation);
    if (dimension == 0 ||
        numerus_matrix_columns(observation) != 1 ||
        numerus_matrix_rows(mean) != dimension ||
        numerus_matrix_columns(mean) != 1 ||
        numerus_matrix_rows(covariance) != dimension ||
        numerus_matrix_columns(covariance) != dimension) {
        return NUMERUS_PROBABILITY_DIMENSION_MISMATCH;
    }

    if (!numerus_size_multiply(
            dimension, sizeof(*residual_values), &residual_bytes)) {
        return NUMERUS_PROBABILITY_SIZE_OVERFLOW;
    }

    residual_values = numerus_probability_alloc(residual_bytes);
    if (residual_values == NULL) {
        return NUMERUS_PROBABILITY_OUT_OF_MEMORY;
    }

    for (index = 0; index < dimension; index++) {
        double observed_value;
        double mean_value;

        matrix_status = numerus_matrix_get(
            observation, index, 0, &observed_value
        );
        if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
            status = numerus_probability_map_matrix_status(matrix_status);
            goto cleanup;
        }
        matrix_status = numerus_matrix_get(mean, index, 0, &mean_value);
        if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
            status = numerus_probability_map_matrix_status(matrix_status);
            goto cleanup;
        }
        if (!isfinite(observed_value) || !isfinite(mean_value)) {
            status = NUMERUS_PROBABILITY_NON_FINITE_INPUT;
            goto cleanup;
        }

        residual_values[index] = observed_value - mean_value;
        if (!isfinite(residual_values[index])) {
            status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
            goto cleanup;
        }
    }

    matrix_status = numerus_matrix_cholesky(covariance, &lower);
    if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
        status = numerus_probability_map_matrix_status(matrix_status);
        goto cleanup;
    }

    matrix_status = (numerus_matrix_status) numerus_matrix_create_dense(
        dimension, 1, residual_values, &residual
    );
    if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
        status = numerus_probability_map_matrix_status(matrix_status);
        goto cleanup;
    }

    matrix_status = numerus_matrix_solve_triangular(
        lower, residual, true, false, &whitened_residual
    );
    if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
        status = numerus_probability_map_matrix_status(matrix_status);
        goto cleanup;
    }

    matrix_status = numerus_matrix_log_determinant(
        covariance, &determinant_sign, &log_abs_determinant
    );
    if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
        status = numerus_probability_map_matrix_status(matrix_status);
        goto cleanup;
    }
    if (determinant_sign != 1 || !isfinite(log_abs_determinant)) {
        status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
        goto cleanup;
    }

    for (index = 0; index < dimension; index++) {
        double value;
        double square;

        matrix_status = numerus_matrix_get(
            whitened_residual, index, 0, &value
        );
        if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
            status = numerus_probability_map_matrix_status(matrix_status);
            goto cleanup;
        }
        square = value * value;
        if (!isfinite(value) || !isfinite(square)) {
            status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
            goto cleanup;
        }
        quadratic_form += square;
        if (!isfinite(quadratic_form)) {
            status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
            goto cleanup;
        }
    }

    result = -0.5 * (
        (double) dimension * log_two_pi +
        log_abs_determinant +
        quadratic_form
    );
    if (!isfinite(result)) {
        status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
        goto cleanup;
    }

    *log_density = result;
    status = NUMERUS_PROBABILITY_SUCCESS;

cleanup:
    if (whitened_residual != NULL) {
        numerus_matrix_destroy(whitened_residual);
    }
    if (lower != NULL) {
        numerus_matrix_destroy(lower);
    }
    if (residual != NULL) {
        numerus_matrix_destroy(residual);
    }
    if (residual_values != NULL) {
        numerus_probability_free(residual_values);
    }
    return status;
}


static numerus_probability_status numerus_probability_map_rng_status(
    numerus_rng_status status
)
{
    switch (status) {
        case NUMERUS_RNG_SUCCESS:
            return NUMERUS_PROBABILITY_SUCCESS;
        case NUMERUS_RNG_INVALID_ARGUMENT:
            return NUMERUS_PROBABILITY_INVALID_ARGUMENT;
        case NUMERUS_RNG_OUT_OF_MEMORY:
            return NUMERUS_PROBABILITY_OUT_OF_MEMORY;
        case NUMERUS_RNG_SIZE_OVERFLOW:
            return NUMERUS_PROBABILITY_SIZE_OVERFLOW;
        default:
            return NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
    }
}

numerus_probability_status numerus_multivariate_gaussian_sample(
    const numerus_matrix *mean,
    const numerus_matrix *covariance,
    numerus_rng *rng,
    numerus_matrix **sample
)
{
    size_t dimension;
    size_t buffer_bytes;
    size_t row;
    size_t column;
    size_t draw_index;
    double *standard_normals = NULL;
    double *sample_values = NULL;
    numerus_matrix *lower = NULL;
    numerus_matrix *result_matrix = NULL;
    numerus_rng *working_rng = NULL;
    numerus_matrix_status matrix_status;
    numerus_rng_status rng_status;
    numerus_probability_status status;

    if (sample == NULL) {
        return NUMERUS_PROBABILITY_INVALID_ARGUMENT;
    }
    *sample = NULL;

    if (mean == NULL || covariance == NULL || rng == NULL) {
        return NUMERUS_PROBABILITY_INVALID_ARGUMENT;
    }

    dimension = numerus_matrix_rows(mean);
    if (dimension == 0 ||
        numerus_matrix_columns(mean) != 1 ||
        numerus_matrix_rows(covariance) != dimension ||
        numerus_matrix_columns(covariance) != dimension) {
        return NUMERUS_PROBABILITY_DIMENSION_MISMATCH;
    }

    for (row = 0; row < dimension; row++) {
        double value;

        matrix_status = numerus_matrix_get(mean, row, 0, &value);
        if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
            return numerus_probability_map_matrix_status(matrix_status);
        }
        if (!isfinite(value)) {
            return NUMERUS_PROBABILITY_NON_FINITE_INPUT;
        }
    }

    matrix_status = numerus_matrix_cholesky(covariance, &lower);
    if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
        return numerus_probability_map_matrix_status(matrix_status);
    }

    if (!numerus_size_multiply(
            dimension, sizeof(*standard_normals), &buffer_bytes)) {
        status = NUMERUS_PROBABILITY_SIZE_OVERFLOW;
        goto cleanup;
    }

    standard_normals = numerus_probability_alloc(buffer_bytes);
    sample_values = numerus_probability_alloc(buffer_bytes);
    if (standard_normals == NULL || sample_values == NULL) {
        status = NUMERUS_PROBABILITY_OUT_OF_MEMORY;
        goto cleanup;
    }

    /*
     * Work on a clone so any failure before the output Matrix is fully
     * allocated leaves the caller's RNG state untouched. The original state
     * is advanced by the same raw-output count only after output creation.
     */
    rng_status = numerus_rng_clone(rng, &working_rng);
    if (rng_status != NUMERUS_RNG_SUCCESS) {
        status = numerus_probability_map_rng_status(rng_status);
        goto cleanup;
    }

    for (row = 0; row < dimension; row++) {
        rng_status = numerus_rng_normal(working_rng, &standard_normals[row]);
        if (rng_status != NUMERUS_RNG_SUCCESS) {
            status = numerus_probability_map_rng_status(rng_status);
            goto cleanup;
        }
    }

    for (row = 0; row < dimension; row++) {
        double mean_value;
        double value;

        matrix_status = numerus_matrix_get(mean, row, 0, &mean_value);
        if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
            status = numerus_probability_map_matrix_status(matrix_status);
            goto cleanup;
        }
        value = mean_value;

        for (column = 0; column <= row; column++) {
            double coefficient;
            double term;

            matrix_status = numerus_matrix_get(
                lower, row, column, &coefficient
            );
            if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
                status = numerus_probability_map_matrix_status(matrix_status);
                goto cleanup;
            }

            term = coefficient * standard_normals[column];
            if (!isfinite(term)) {
                status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
                goto cleanup;
            }
            value += term;
            if (!isfinite(value)) {
                status = NUMERUS_PROBABILITY_NUMERICAL_FAILURE;
                goto cleanup;
            }
        }

        sample_values[row] = value;
    }

    matrix_status = (numerus_matrix_status) numerus_matrix_create_dense(
        dimension, 1, sample_values, &result_matrix
    );
    if (matrix_status != NUMERUS_MATRIX_SUCCESS) {
        status = numerus_probability_map_matrix_status(matrix_status);
        goto cleanup;
    }

    /* Each normal variate consumes exactly four raw outputs (no cache). */
    for (row = 0; row < dimension; row++) {
        for (draw_index = 0; draw_index < 4; draw_index++) {
            uint32_t ignored;

            rng_status = numerus_rng_next_u32(rng, &ignored);
            if (rng_status != NUMERUS_RNG_SUCCESS) {
                status = numerus_probability_map_rng_status(rng_status);
                goto cleanup;
            }
        }
    }

    *sample = result_matrix;
    result_matrix = NULL;
    status = NUMERUS_PROBABILITY_SUCCESS;

cleanup:
    if (result_matrix != NULL) {
        numerus_matrix_destroy(result_matrix);
    }
    if (working_rng != NULL) {
        numerus_rng_destroy(working_rng);
    }
    if (lower != NULL) {
        numerus_matrix_destroy(lower);
    }
    if (sample_values != NULL) {
        numerus_probability_free(sample_values);
    }
    if (standard_normals != NULL) {
        numerus_probability_free(standard_normals);
    }
    return status;
}
