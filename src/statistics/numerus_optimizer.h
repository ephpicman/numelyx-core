/**
 * @file numerus_optimizer.h
 * @brief Optimization request, callback, status, and result contracts.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#ifndef NUMERUS_OPTIMIZER_H
#define NUMERUS_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>

/** API-level status for generic optimization requests and result accessors. */
typedef enum {
    NUMERUS_OPTIMIZER_SUCCESS = 0,
    NUMERUS_OPTIMIZER_INVALID_ARGUMENT,
    NUMERUS_OPTIMIZER_NON_FINITE_INPUT,
    NUMERUS_OPTIMIZER_SIZE_OVERFLOW,
    NUMERUS_OPTIMIZER_OUT_OF_MEMORY,
    NUMERUS_OPTIMIZER_OUT_OF_BOUNDS,
    NUMERUS_OPTIMIZER_NO_OBJECTIVE_VALUE
} numerus_optimizer_status;

/** Numerical stopping reason, distinct from API-level status. */
typedef enum {
    NUMERUS_OPTIMIZER_CONVERGED_GRADIENT = 0,
    NUMERUS_OPTIMIZER_ITERATION_LIMIT,
    NUMERUS_OPTIMIZER_STEP_TOO_SMALL,
    NUMERUS_OPTIMIZER_LINE_SEARCH_FAILED,
    NUMERUS_OPTIMIZER_OBJECTIVE_CALLBACK_FAILED,
    NUMERUS_OPTIMIZER_GRADIENT_CALLBACK_FAILED,
    NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED,
    NUMERUS_OPTIMIZER_NON_FINITE_OBJECTIVE,
    NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT,
    NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN
} numerus_optimizer_termination;

/** Caller-supplied objective; returns true only when writing a complete value. */
typedef bool (*numerus_objective_fn)(
    const double *parameters,
    size_t parameter_count,
    const void *context,
    double *value
);

/** Optional caller-supplied gradient; NULL selects central finite differences. */
typedef bool (*numerus_gradient_fn)(
    const double *parameters,
    size_t parameter_count,
    const void *context,
    double *gradient
);

typedef struct {
    size_t max_iterations;
    size_t max_line_search_iterations;
    double gradient_tolerance;
    double minimum_step;
    double step_tolerance;
    double initial_step;
    double finite_difference_relative_step;
} numerus_optimizer_options;

/** Opaque result owning the last accepted parameter vector. */
typedef struct numerus_optimizer_result numerus_optimizer_result;

/** Fill conservative v1 defaults; returns INVALID_ARGUMENT for a NULL output. */
numerus_optimizer_status numerus_optimizer_default_options(
    numerus_optimizer_options *options
);

/**
 * Minimize an unconstrained smooth objective using BFGS and Armijo backtracking.
 *
 * options must be initialized (normally with numerus_optimizer_default_options)
 * and valid. A NULL gradient selects central finite differences. For a valid
 * request, *result receives a result even when the optimizer does not converge;
 * API/initialization failure leaves *result NULL.
 */
numerus_optimizer_status numerus_optimizer_minimize(
    numerus_objective_fn objective,
    numerus_gradient_fn gradient,
    const void *context,
    const double *initial_parameters,
    size_t parameter_count,
    const numerus_optimizer_options *options,
    numerus_optimizer_result **result
);

/** Release a result; accepts NULL. */
void numerus_optimizer_result_destroy(numerus_optimizer_result *result);

/** Return the number of parameters owned by the result, or zero for NULL. */
size_t numerus_optimizer_result_parameter_count(
    const numerus_optimizer_result *result
);

/**
 * Copy one accepted parameter into value. The index is zero-based; invalid
 * indices or NULL outputs return an API status and leave value unchanged.
 */
numerus_optimizer_status numerus_optimizer_result_get_parameter(
    const numerus_optimizer_result *result,
    size_t index,
    double *value
);

/** True when the result contains a finite objective value for an accepted point. */
bool numerus_optimizer_result_objective_value_valid(
    const numerus_optimizer_result *result
);

/** Read the last accepted objective value; fail if no value was evaluated. */
numerus_optimizer_status numerus_optimizer_result_get_objective(
    const numerus_optimizer_result *result,
    double *value
);

/** Read the numerical termination reason, distinct from API-level status. */
numerus_optimizer_status numerus_optimizer_result_get_termination(
    const numerus_optimizer_result *result,
    numerus_optimizer_termination *termination
);

/** Return accepted optimization iterations, or zero for NULL. */
size_t numerus_optimizer_result_iterations(
    const numerus_optimizer_result *result
);

/** Return objective callback evaluations, or zero for NULL. */
size_t numerus_optimizer_result_objective_evaluations(
    const numerus_optimizer_result *result
);

/** Return gradient evaluations (including finite-difference passes), or zero for NULL. */
size_t numerus_optimizer_result_gradient_evaluations(
    const numerus_optimizer_result *result
);

#endif
