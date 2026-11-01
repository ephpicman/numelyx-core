/**
 * @file numelyx_core_optimizer.h
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
} numelyx_core_optimizer_status;

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
} numelyx_core_optimizer_termination;

/** Caller-supplied objective; returns true only when writing a complete value. */
typedef bool (*numelyx_core_objective_fn)(
    const double *parameters,
    size_t parameter_count,
    const void *context,
    double *value
);

/** Optional caller-supplied gradient; NULL selects central finite differences. */
typedef bool (*numelyx_core_gradient_fn)(
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
} numelyx_core_optimizer_options;

/**
 * Opaque immutable result owning the last accepted parameter vector.
 * Concurrent getters are safe while each reader holds a retained reference.
 */
typedef struct numelyx_core_optimizer_result numelyx_core_optimizer_result;

/** Fill conservative v1 defaults; returns INVALID_ARGUMENT for a NULL output. */
numelyx_core_optimizer_status numelyx_core_optimizer_default_options(
    numelyx_core_optimizer_options *options
);

/**
 * Minimize an unconstrained smooth objective using BFGS and Armijo backtracking.
 *
 * options must be initialized (normally with numelyx_core_optimizer_default_options)
 * and valid. A NULL gradient selects central finite differences. For a valid
 * request, *result receives a result even when the optimizer does not converge;
 * API/initialization failure leaves *result NULL.
 */
numelyx_core_optimizer_status numelyx_core_optimizer_minimize(
    numelyx_core_objective_fn objective,
    numelyx_core_gradient_fn gradient,
    const void *context,
    const double *initial_parameters,
    size_t parameter_count,
    const numelyx_core_optimizer_options *options,
    numelyx_core_optimizer_result **result
);

/** Retain a result for shared read-only use; caller must already hold a valid reference. */
numelyx_core_optimizer_status numelyx_core_optimizer_result_retain(
    numelyx_core_optimizer_result *result
);

/** Release a result reference; accepts NULL. */
void numelyx_core_optimizer_result_destroy(numelyx_core_optimizer_result *result);

/** Return the number of parameters owned by the result, or zero for NULL. */
size_t numelyx_core_optimizer_result_parameter_count(
    const numelyx_core_optimizer_result *result
);

/**
 * Copy one accepted parameter into value. The index is zero-based; invalid
 * indices or NULL outputs return an API status and leave value unchanged.
 */
numelyx_core_optimizer_status numelyx_core_optimizer_result_get_parameter(
    const numelyx_core_optimizer_result *result,
    size_t index,
    double *value
);

/** True when the result contains a finite objective value for an accepted point. */
bool numelyx_core_optimizer_result_objective_value_valid(
    const numelyx_core_optimizer_result *result
);

/** Read the last accepted objective value; fail if no value was evaluated. */
numelyx_core_optimizer_status numelyx_core_optimizer_result_get_objective(
    const numelyx_core_optimizer_result *result,
    double *value
);

/** Read the numerical termination reason, distinct from API-level status. */
numelyx_core_optimizer_status numelyx_core_optimizer_result_get_termination(
    const numelyx_core_optimizer_result *result,
    numelyx_core_optimizer_termination *termination
);

/** Return accepted optimization iterations, or zero for NULL. */
size_t numelyx_core_optimizer_result_iterations(
    const numelyx_core_optimizer_result *result
);

/** Return objective callback evaluations, or zero for NULL. */
size_t numelyx_core_optimizer_result_objective_evaluations(
    const numelyx_core_optimizer_result *result
);

/** Return gradient evaluations (including finite-difference passes), or zero for NULL. */
size_t numelyx_core_optimizer_result_gradient_evaluations(
    const numelyx_core_optimizer_result *result
);

#endif
