/**
 * @file numelyx_core_probability.h
 * @brief Probability primitive declarations and argument/result contracts.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#ifndef NUMERUS_PROBABILITY_H
#define NUMERUS_PROBABILITY_H

#include "numelyx_core_matrix.h"
#include "numelyx_core_rng.h"

/** Status values for reusable probability primitives. */
typedef enum {
    NUMERUS_PROBABILITY_SUCCESS = 0,
    NUMERUS_PROBABILITY_INVALID_ARGUMENT,
    NUMERUS_PROBABILITY_DIMENSION_MISMATCH,
    NUMERUS_PROBABILITY_NON_FINITE_INPUT,
    NUMERUS_PROBABILITY_NOT_POSITIVE_DEFINITE,
    NUMERUS_PROBABILITY_SIZE_OVERFLOW,
    NUMERUS_PROBABILITY_OUT_OF_MEMORY,
    NUMERUS_PROBABILITY_NUMERICAL_FAILURE
} numelyx_core_probability_status;

/**
 * Compute the multivariate Gaussian log density for column-vector observation
 * and mean with a symmetric positive-definite covariance Matrix.
 *
 * Uses Cholesky factorization, a triangular solve and signed log determinant;
 * never forms the covariance inverse. Inputs are borrowed and unchanged.
 * The scalar output remains unchanged on every failure.
 */
numelyx_core_probability_status numelyx_core_multivariate_gaussian_log_density(
    const numelyx_core_matrix *observation,
    const numelyx_core_matrix *mean,
    const numelyx_core_matrix *covariance,
    double *log_density
);

/**
 * Draw one multivariate Gaussian sample from a column-vector mean and an SPD
 * covariance Matrix using explicit RNG state. The returned dense Matrix is
 * owned by the caller. All required allocations and factorization happen
 * before the RNG is advanced; on failure, *sample remains NULL.
 */
numelyx_core_probability_status numelyx_core_multivariate_gaussian_sample(
    const numelyx_core_matrix *mean,
    const numelyx_core_matrix *covariance,
    numelyx_core_rng *rng,
    numelyx_core_matrix **sample
);

#endif
