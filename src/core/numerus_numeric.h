/**
 * @file numerus_numeric.h
 * @brief Shared floating-point comparison helpers for Numerus.
 */
#ifndef NUMERUS_NUMERIC_H
#define NUMERUS_NUMERIC_H

#include <stdbool.h>
#include <stddef.h>
#include <math.h>

/**
 * Default absolute/relative tolerance used for approximate comparisons.
 *
 * This is an application-level tolerance, not the machine epsilon of double.
 */
#define NUMERUS_EPSILON 1e-9

/**
 * Compare two finite values with a combined absolute/relative tolerance.
 *
 * Exact equality is handled first so equal infinities compare equal; NaN
 * compares unequal to every value, including itself.
 */
/**
 * @brief Compare two doubles with Numerus's combined absolute/relative tolerance.
 * @param left First value.
 * @param right Second value.
 * @return true for exact equality (including equal infinities), or for finite
 *         values within NUMERUS_EPSILON * max(1, |left|, |right|).
 *
 * NaN is unequal to every value. Unequal infinities and finite/infinite pairs
 * compare false; this helper is intended for approximate numerical equality,
 * not bitwise identity.
 */
static inline bool numerus_double_equals(double left, double right)
{
    double left_magnitude;
    double right_magnitude;
    double scale;

    if (left == right) {
        return true;
    }

    if (!isfinite(left) || !isfinite(right)) {
        return false;
    }

    left_magnitude = fabs(left);
    right_magnitude = fabs(right);
    scale = left_magnitude > right_magnitude
        ? left_magnitude
        : right_magnitude;

    if (scale < 1.0) {
        scale = 1.0;
    }

    return fabs(left - right) <= NUMERUS_EPSILON * scale;
}

/** Return whether a value is within NUMERUS_EPSILON of zero. */
/**
 * @brief Test whether a finite value is within NUMERUS_EPSILON of zero.
 * @param value Value to test.
 * @return true only when value is finite and its absolute magnitude is at most
 *         NUMERUS_EPSILON.
 */
static inline bool numerus_double_is_zero(double value)
{
    return isfinite(value) && fabs(value) <= NUMERUS_EPSILON;
}

/** Return whether a value is within NUMERUS_EPSILON of one. */
/**
 * @brief Test whether a value is approximately one under numerus_double_equals().
 * @param value Value to test.
 * @return Result of comparing @p value with 1.0 using the shared tolerance.
 */
static inline bool numerus_double_is_one(double value)
{
    return numerus_double_equals(value, 1.0);
}


/** Status values for stable scalar and log-domain operations. */
typedef enum {
    NUMERUS_NUMERIC_SUCCESS = 0,
    NUMERUS_NUMERIC_INVALID_ARGUMENT,
    NUMERUS_NUMERIC_DOMAIN_ERROR,
    NUMERUS_NUMERIC_NON_FINITE,
    NUMERUS_NUMERIC_NUMERICAL_FAILURE
} numerus_numeric_status;

/** Compute a stable sigmoid for a finite input; output is unchanged on failure. */
/**
 * @brief Evaluate the logistic sigmoid without avoidable overflow.
 * @param input Finite input value.
 * @param result Receives the result on success; unchanged on failure.
 * @return A numeric status describing success or the invalid/non-finite input.
 */
numerus_numeric_status numerus_numeric_sigmoid(double input, double *result);

/** Compute log(p / (1-p)) for finite 0 < p < 1. */
/**
 * @brief Compute log(p / (1 - p)) for a probability strictly between zero and one.
 * @param probability Finite probability in the open interval (0, 1).
 * @param result Receives the result on success; unchanged on failure.
 * @return A numeric status describing success or domain/argument failure.
 */
numerus_numeric_status numerus_numeric_logit(double probability, double *result);

/** Compute log(1 + exp(x)) without avoidable overflow. */
/**
 * @brief Evaluate log(1 + exp(input)) using a stable piecewise formulation.
 * @param input Finite input value.
 * @param result Receives the result on success; unchanged on failure.
 * @return A numeric status describing success or invalid/non-finite input.
 */
numerus_numeric_status numerus_numeric_softplus(double input, double *result);

/** Compute log(sigmoid(x)) without taking log of a rounded sigmoid result. */
/**
 * @brief Compute log(sigmoid(input)) directly in a numerically stable form.
 * @param input Finite input value.
 * @param result Receives the result on success; unchanged on failure.
 * @return A numeric status describing success or invalid/non-finite input.
 */
numerus_numeric_status numerus_numeric_log_sigmoid(double input, double *result);

/**
 * Compute log(sum(exp(values[i]))) for a non-empty array of finite values.
 * Uses max-shifting and does not allocate. Output is unchanged on failure.
 */
numerus_numeric_status numerus_numeric_log_sum_exp(
    const double *values,
    size_t count,
    double *result
);


#endif
