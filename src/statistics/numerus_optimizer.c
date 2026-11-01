/**
 * @file numerus_optimizer.c
 * @brief Generic numerical optimization routines and result handling.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#include "numerus_optimizer.h"
#include "numerus_size.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(NUMERUS_OPTIMIZER_TEST_ALLOCATOR)
void *numerus_optimizer_test_alloc(size_t size);
void numerus_optimizer_test_free(void *pointer);
# define numerus_optimizer_alloc(size) numerus_optimizer_test_alloc(size)
# define numerus_optimizer_free(pointer) numerus_optimizer_test_free(pointer)
#elif !defined(NUMERUS_OPTIMIZER_USE_LIBC_ALLOC)
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_optimizer_alloc(size) emalloc(size)
# define numerus_optimizer_free(pointer) efree(pointer)
#else
# define numerus_optimizer_alloc(size) malloc(size)
# define numerus_optimizer_free(pointer) free(pointer)
#endif

#define NUMERUS_OPTIMIZER_ARMIJO_CONSTANT 1e-4
#define NUMERUS_OPTIMIZER_BACKTRACKING_RATIO 0.5

struct numerus_optimizer_result {
    size_t parameter_count;
    double *parameters;
    double objective_value;
    bool objective_value_valid;
    size_t iterations;
    size_t objective_evaluations;
    size_t gradient_evaluations;
    numerus_optimizer_termination termination;
};

typedef enum {
    OBJECTIVE_EVALUATION_OK = 0,
    OBJECTIVE_EVALUATION_CALLBACK_FAILED,
    OBJECTIVE_EVALUATION_NON_FINITE
} objective_evaluation_status;

static void numerus_optimizer_reset_identity(double *matrix, size_t size)
{
    size_t row;
    size_t column;

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            matrix[row * size + column] = row == column ? 1.0 : 0.0;
        }
    }
}

static bool numerus_optimizer_options_are_valid(
    const numerus_optimizer_options *options
)
{
    if (options == NULL ||
        options->max_iterations == 0 ||
        options->max_line_search_iterations == 0) {
        return false;
    }

    if (!isfinite(options->gradient_tolerance) ||
        !isfinite(options->minimum_step) ||
        !isfinite(options->step_tolerance) ||
        !isfinite(options->initial_step) ||
        !isfinite(options->finite_difference_relative_step)) {
        return false;
    }

    return options->gradient_tolerance > 0.0 &&
        options->minimum_step > 0.0 &&
        options->step_tolerance > 0.0 &&
        options->initial_step > 0.0 &&
        options->minimum_step <= options->initial_step &&
        options->finite_difference_relative_step > 0.0;
}

static objective_evaluation_status numerus_optimizer_evaluate_objective(
    numerus_objective_fn objective,
    const void *context,
    const double *parameters,
    size_t parameter_count,
    size_t *evaluation_count,
    double *value
)
{
    double evaluated_value;

    (*evaluation_count)++;
    if (!objective(parameters, parameter_count, context, &evaluated_value)) {
        return OBJECTIVE_EVALUATION_CALLBACK_FAILED;
    }
    if (!isfinite(evaluated_value)) {
        return OBJECTIVE_EVALUATION_NON_FINITE;
    }

    *value = evaluated_value;
    return OBJECTIVE_EVALUATION_OK;
}

static numerus_optimizer_termination numerus_optimizer_evaluate_gradient(
    numerus_objective_fn objective,
    numerus_gradient_fn gradient,
    const void *context,
    const double *parameters,
    size_t parameter_count,
    const numerus_optimizer_options *options,
    numerus_optimizer_result *result,
    double *gradient_values,
    double *temporary_parameters
)
{
    size_t index;

    result->gradient_evaluations++;

    if (gradient != NULL) {
        if (!gradient(
            parameters, parameter_count, context, gradient_values
        )) {
            return NUMERUS_OPTIMIZER_GRADIENT_CALLBACK_FAILED;
        }

        for (index = 0; index < parameter_count; index++) {
            if (!isfinite(gradient_values[index])) {
                return NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT;
            }
        }
        return NUMERUS_OPTIMIZER_CONVERGED_GRADIENT;
    }

    /* If no gradient callback is supplied, estimate each component with a
     * central difference. The step scales with max(1, |xᵢ|) so parameters
     * with different magnitudes are not all perturbed by the same absolute
     * amount. Both callback evaluations are counted as objective evaluations.
     */
    for (index = 0; index < parameter_count; index++) {
        double step = options->finite_difference_relative_step *
            fmax(1.0, fabs(parameters[index]));
        double plus_value;
        double minus_value;
        objective_evaluation_status evaluation_status;

        if (!isfinite(step) || step <= 0.0) {
            return NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED;
        }

        memcpy(
            temporary_parameters,
            parameters,
            parameter_count * sizeof(*temporary_parameters)
        );
        temporary_parameters[index] = parameters[index] + step;
        if (!isfinite(temporary_parameters[index]) ||
            temporary_parameters[index] == parameters[index]) {
            return NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED;
        }

        evaluation_status = numerus_optimizer_evaluate_objective(
            objective,
            context,
            temporary_parameters,
            parameter_count,
            &result->objective_evaluations,
            &plus_value
        );
        if (evaluation_status != OBJECTIVE_EVALUATION_OK) {
            return NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED;
        }

        temporary_parameters[index] = parameters[index] - step;
        if (!isfinite(temporary_parameters[index]) ||
            temporary_parameters[index] == parameters[index]) {
            return NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED;
        }

        evaluation_status = numerus_optimizer_evaluate_objective(
            objective,
            context,
            temporary_parameters,
            parameter_count,
            &result->objective_evaluations,
            &minus_value
        );
        if (evaluation_status != OBJECTIVE_EVALUATION_OK) {
            return NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED;
        }

        gradient_values[index] = (plus_value - minus_value) / (2.0 * step);
        if (!isfinite(gradient_values[index])) {
            return NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT;
        }
    }

    return NUMERUS_OPTIMIZER_CONVERGED_GRADIENT;
}

static double numerus_optimizer_gradient_norm(
    const double *gradient,
    size_t size
)
{
    size_t index;
    double norm = 0.0;

    for (index = 0; index < size; index++) {
        double magnitude = fabs(gradient[index]);
        if (magnitude > norm) {
            norm = magnitude;
        }
    }
    return norm;
}

static bool numerus_optimizer_compute_direction(
    const double *inverse_hessian,
    const double *gradient,
    size_t size,
    double *direction,
    double *directional_derivative
)
{
    size_t row;
    size_t column;
    double derivative = 0.0;

    for (row = 0; row < size; row++) {
        double value = 0.0;

        for (column = 0; column < size; column++) {
            value += inverse_hessian[row * size + column] * gradient[column];
            if (!isfinite(value)) {
                return false;
            }
        }

        direction[row] = -value;
        if (!isfinite(direction[row])) {
            return false;
        }
        derivative += gradient[row] * direction[row];
        if (!isfinite(derivative)) {
            return false;
        }
    }

    *directional_derivative = derivative;
    return true;
}

static bool numerus_optimizer_compute_steepest_descent(
    const double *gradient,
    size_t size,
    double *direction,
    double *directional_derivative
)
{
    size_t index;
    double derivative = 0.0;

    for (index = 0; index < size; index++) {
        direction[index] = -gradient[index];
        derivative -= gradient[index] * gradient[index];
        if (!isfinite(derivative)) {
            return false;
        }
    }

    *directional_derivative = derivative;
    return true;
}

static void numerus_optimizer_update_inverse_hessian(
    double *inverse_hessian,
    double *next_inverse_hessian,
    const double *step_vector,
    const double *gradient_delta,
    double *hessian_gradient,
    size_t size
)
{
    size_t row;
    size_t column;
    double curvature = 0.0;
    double gradient_curvature = 0.0;
    double step_norm = 0.0;
    double gradient_delta_norm = 0.0;
    double threshold;
    double coefficient;
    double reciprocal_curvature;

    for (row = 0; row < size; row++) {
        curvature += gradient_delta[row] * step_vector[row];
        step_norm = hypot(step_norm, step_vector[row]);
        gradient_delta_norm = hypot(
            gradient_delta_norm, gradient_delta[row]
        );
    }

    /* BFGS requires positive curvature sᵀy. When round-off or a non-convex
     * local step makes that condition unreliable, reset the approximation to
     * identity instead of applying an unstable rank-two update.
     */
    threshold = sqrt(DBL_EPSILON) * step_norm * gradient_delta_norm;
    if (!isfinite(curvature) || !isfinite(threshold) ||
        curvature <= threshold || curvature <= 0.0) {
        numerus_optimizer_reset_identity(inverse_hessian, size);
        return;
    }

    for (row = 0; row < size; row++) {
        double value = 0.0;

        for (column = 0; column < size; column++) {
            value += inverse_hessian[row * size + column] *
                gradient_delta[column];
            if (!isfinite(value)) {
                numerus_optimizer_reset_identity(inverse_hessian, size);
                return;
            }
        }
        hessian_gradient[row] = value;
        gradient_curvature += gradient_delta[row] * value;
        if (!isfinite(gradient_curvature)) {
            numerus_optimizer_reset_identity(inverse_hessian, size);
            return;
        }
    }

    if (gradient_curvature < 0.0) {
        numerus_optimizer_reset_identity(inverse_hessian, size);
        return;
    }

    coefficient = (1.0 + gradient_curvature / curvature) / curvature;
    reciprocal_curvature = 1.0 / curvature;
    if (!isfinite(coefficient) || !isfinite(reciprocal_curvature)) {
        numerus_optimizer_reset_identity(inverse_hessian, size);
        return;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double rank_one = coefficient *
                step_vector[row] * step_vector[column];
            double cross = reciprocal_curvature *
                (hessian_gradient[row] * step_vector[column] +
                 step_vector[row] * hessian_gradient[column]);
            double value = inverse_hessian[row * size + column] +
                rank_one - cross;

            if (!isfinite(rank_one) || !isfinite(cross) || !isfinite(value)) {
                numerus_optimizer_reset_identity(inverse_hessian, size);
                return;
            }
            next_inverse_hessian[row * size + column] = value;
        }
    }

    memcpy(
        inverse_hessian,
        next_inverse_hessian,
        size * size * sizeof(*inverse_hessian)
    );
}

static void numerus_optimizer_release_workspace(
    double *gradient,
    double *new_gradient,
    double *direction,
    double *trial_parameters,
    double *step_vector,
    double *gradient_delta,
    double *hessian_gradient,
    double *inverse_hessian,
    double *next_inverse_hessian
)
{
    if (gradient != NULL) numerus_optimizer_free(gradient);
    if (new_gradient != NULL) numerus_optimizer_free(new_gradient);
    if (direction != NULL) numerus_optimizer_free(direction);
    if (trial_parameters != NULL) numerus_optimizer_free(trial_parameters);
    if (step_vector != NULL) numerus_optimizer_free(step_vector);
    if (gradient_delta != NULL) numerus_optimizer_free(gradient_delta);
    if (hessian_gradient != NULL) numerus_optimizer_free(hessian_gradient);
    if (inverse_hessian != NULL) numerus_optimizer_free(inverse_hessian);
    if (next_inverse_hessian != NULL) numerus_optimizer_free(next_inverse_hessian);
}

numerus_optimizer_status numerus_optimizer_default_options(
    numerus_optimizer_options *options
)
{
    if (options == NULL) {
        return NUMERUS_OPTIMIZER_INVALID_ARGUMENT;
    }

    options->max_iterations = 1000;
    options->max_line_search_iterations = 40;
    options->gradient_tolerance = 1e-8;
    options->minimum_step = 1e-14;
    options->step_tolerance = 1e-12;
    options->initial_step = 1.0;
    options->finite_difference_relative_step = cbrt(DBL_EPSILON);
    return NUMERUS_OPTIMIZER_SUCCESS;
}

numerus_optimizer_status numerus_optimizer_minimize(
    numerus_objective_fn objective,
    numerus_gradient_fn gradient,
    const void *context,
    const double *initial_parameters,
    size_t parameter_count,
    const numerus_optimizer_options *options,
    numerus_optimizer_result **result
)
{
    numerus_optimizer_result *output = NULL;
    double *gradient_values = NULL;
    double *new_gradient = NULL;
    double *direction = NULL;
    double *trial_parameters = NULL;
    double *step_vector = NULL;
    double *gradient_delta = NULL;
    double *hessian_gradient = NULL;
    double *inverse_hessian = NULL;
    double *next_inverse_hessian = NULL;
    size_t vector_bytes;
    size_t matrix_elements;
    size_t matrix_bytes;
    size_t index;
    double objective_value = 0.0;
    numerus_optimizer_status api_status = NUMERUS_OPTIMIZER_SUCCESS;
    numerus_optimizer_termination termination =
        NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN;

    if (result == NULL) {
        return NUMERUS_OPTIMIZER_INVALID_ARGUMENT;
    }
    *result = NULL;

    if (objective == NULL || initial_parameters == NULL ||
        parameter_count == 0 || !numerus_optimizer_options_are_valid(options)) {
        if (options != NULL &&
            (!isfinite(options->gradient_tolerance) ||
             !isfinite(options->minimum_step) ||
             !isfinite(options->step_tolerance) ||
             !isfinite(options->initial_step) ||
             !isfinite(options->finite_difference_relative_step))) {
            return NUMERUS_OPTIMIZER_NON_FINITE_INPUT;
        }
        return NUMERUS_OPTIMIZER_INVALID_ARGUMENT;
    }

    for (index = 0; index < parameter_count; index++) {
        if (!isfinite(initial_parameters[index])) {
            return NUMERUS_OPTIMIZER_NON_FINITE_INPUT;
        }
    }

    if (!numerus_size_multiply(
            parameter_count, sizeof(*initial_parameters), &vector_bytes) ||
        !numerus_size_multiply(
            parameter_count, parameter_count, &matrix_elements) ||
        !numerus_size_multiply(
            matrix_elements, sizeof(*inverse_hessian), &matrix_bytes)) {
        return NUMERUS_OPTIMIZER_SIZE_OVERFLOW;
    }

    output = numerus_optimizer_alloc(sizeof(*output));
    if (output == NULL) {
        return NUMERUS_OPTIMIZER_OUT_OF_MEMORY;
    }
    memset(output, 0, sizeof(*output));
    output->parameter_count = parameter_count;
    output->termination = NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN;
    output->parameters = numerus_optimizer_alloc(vector_bytes);
    if (output->parameters == NULL) {
        api_status = NUMERUS_OPTIMIZER_OUT_OF_MEMORY;
        goto api_failure;
    }
    memcpy(output->parameters, initial_parameters, vector_bytes);

    gradient_values = numerus_optimizer_alloc(vector_bytes);
    new_gradient = numerus_optimizer_alloc(vector_bytes);
    direction = numerus_optimizer_alloc(vector_bytes);
    trial_parameters = numerus_optimizer_alloc(vector_bytes);
    step_vector = numerus_optimizer_alloc(vector_bytes);
    gradient_delta = numerus_optimizer_alloc(vector_bytes);
    hessian_gradient = numerus_optimizer_alloc(vector_bytes);
    inverse_hessian = numerus_optimizer_alloc(matrix_bytes);
    next_inverse_hessian = numerus_optimizer_alloc(matrix_bytes);

    if (gradient_values == NULL || new_gradient == NULL || direction == NULL ||
        trial_parameters == NULL || step_vector == NULL ||
        gradient_delta == NULL || hessian_gradient == NULL ||
        inverse_hessian == NULL || next_inverse_hessian == NULL) {
        api_status = NUMERUS_OPTIMIZER_OUT_OF_MEMORY;
        goto api_failure;
    }

    numerus_optimizer_reset_identity(inverse_hessian, parameter_count);

    {
        objective_evaluation_status evaluation_status =
            numerus_optimizer_evaluate_objective(
                objective,
                context,
                output->parameters,
                parameter_count,
                &output->objective_evaluations,
                &objective_value
            );

        if (evaluation_status == OBJECTIVE_EVALUATION_CALLBACK_FAILED) {
            termination = NUMERUS_OPTIMIZER_OBJECTIVE_CALLBACK_FAILED;
            goto publish_result;
        }
        if (evaluation_status == OBJECTIVE_EVALUATION_NON_FINITE) {
            termination = NUMERUS_OPTIMIZER_NON_FINITE_OBJECTIVE;
            goto publish_result;
        }
    }

    output->objective_value = objective_value;
    output->objective_value_valid = true;

    termination = numerus_optimizer_evaluate_gradient(
        objective, gradient, context, output->parameters, parameter_count,
        options, output, gradient_values, trial_parameters
    );
    if (termination != NUMERUS_OPTIMIZER_CONVERGED_GRADIENT) {
        goto publish_result;
    }

    for (;;) {
        double gradient_norm = numerus_optimizer_gradient_norm(
            gradient_values, parameter_count
        );
        double directional_derivative;
        double alpha;
        double accepted_objective_value = 0.0;
        bool accepted = false;
        bool step_too_small = false;
        size_t line_search_iteration;

        if (!isfinite(gradient_norm)) {
            termination = NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT;
            goto publish_result;
        }
        if (gradient_norm <= options->gradient_tolerance) {
            termination = NUMERUS_OPTIMIZER_CONVERGED_GRADIENT;
            goto publish_result;
        }
        if (output->iterations >= options->max_iterations) {
            termination = NUMERUS_OPTIMIZER_ITERATION_LIMIT;
            goto publish_result;
        }

        /* BFGS should produce a descent direction. If its approximation has
         * become numerically untrustworthy, reset it and retry with steepest
         * descent; only fail when even that direction is not a descent step.
         */
        if (!numerus_optimizer_compute_direction(
                inverse_hessian, gradient_values, parameter_count,
                direction, &directional_derivative) ||
            directional_derivative >= 0.0) {
            numerus_optimizer_reset_identity(inverse_hessian, parameter_count);
            if (!numerus_optimizer_compute_steepest_descent(
                    gradient_values, parameter_count, direction,
                    &directional_derivative) ||
                directional_derivative >= 0.0) {
                termination = NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN;
                goto publish_result;
            }
        }

        /* Armijo backtracking accepts only a finite trial that decreases the
         * objective by the sufficient-decrease bound. Failed/non-finite trial
         * evaluations shrink alpha; the current accepted point is not changed
         * until a candidate passes this test.
         */
        alpha = options->initial_step;
        for (line_search_iteration = 0;
             line_search_iteration < options->max_line_search_iterations;
             line_search_iteration++) {
            double step_norm = 0.0;
            double trial_objective;
            double armijo_bound;
            objective_evaluation_status evaluation_status;
            bool finite_trial = true;

            if (alpha < options->minimum_step) {
                step_norm = 0.0;
                for (index = 0; index < parameter_count; index++) {
                    double step = alpha * direction[index];
                    double magnitude = fabs(step);

                    if (!isfinite(step)) {
                        step_norm = INFINITY;
                        break;
                    }
                    if (magnitude > step_norm) {
                        step_norm = magnitude;
                    }
                }
                step_too_small = step_norm <= options->step_tolerance;
                break;
            }

            for (index = 0; index < parameter_count; index++) {
                double step = alpha * direction[index];
                double magnitude = fabs(step);

                if (!isfinite(step)) {
                    finite_trial = false;
                    break;
                }
                if (magnitude > step_norm) {
                    step_norm = magnitude;
                }
                trial_parameters[index] =
                    output->parameters[index] + step;
                if (!isfinite(trial_parameters[index])) {
                    finite_trial = false;
                    break;
                }
            }

            if (!finite_trial) {
                alpha *= NUMERUS_OPTIMIZER_BACKTRACKING_RATIO;
                continue;
            }
            if (step_norm <= options->step_tolerance) {
                step_too_small = true;
                break;
            }

            evaluation_status = numerus_optimizer_evaluate_objective(
                objective, context, trial_parameters, parameter_count,
                &output->objective_evaluations, &trial_objective
            );
            if (evaluation_status != OBJECTIVE_EVALUATION_OK) {
                alpha *= NUMERUS_OPTIMIZER_BACKTRACKING_RATIO;
                continue;
            }

            armijo_bound = objective_value +
                NUMERUS_OPTIMIZER_ARMIJO_CONSTANT *
                alpha * directional_derivative;
            if (isfinite(armijo_bound) && trial_objective <= armijo_bound) {
                accepted_objective_value = trial_objective;
                accepted = true;
                break;
            }

            alpha *= NUMERUS_OPTIMIZER_BACKTRACKING_RATIO;
        }

        if (!accepted) {
            termination = step_too_small ?
                NUMERUS_OPTIMIZER_STEP_TOO_SMALL :
                NUMERUS_OPTIMIZER_LINE_SEARCH_FAILED;
            goto publish_result;
        }

        for (index = 0; index < parameter_count; index++) {
            step_vector[index] =
                trial_parameters[index] - output->parameters[index];
            if (!isfinite(step_vector[index])) {
                termination = NUMERUS_OPTIMIZER_NUMERICAL_BREAKDOWN;
                goto publish_result;
            }
        }

        memcpy(output->parameters, trial_parameters, vector_bytes);
        objective_value = accepted_objective_value;
        output->objective_value = objective_value;
        output->objective_value_valid = true;
        output->iterations++;

        termination = numerus_optimizer_evaluate_gradient(
            objective, gradient, context, output->parameters, parameter_count,
            options, output, new_gradient, trial_parameters
        );
        if (termination != NUMERUS_OPTIMIZER_CONVERGED_GRADIENT) {
            goto publish_result;
        }

        for (index = 0; index < parameter_count; index++) {
            gradient_delta[index] = new_gradient[index] - gradient_values[index];
            if (!isfinite(gradient_delta[index])) {
                termination = NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT;
                goto publish_result;
            }
        }

        numerus_optimizer_update_inverse_hessian(
            inverse_hessian, next_inverse_hessian, step_vector,
            gradient_delta, hessian_gradient, parameter_count
        );
        memcpy(gradient_values, new_gradient, vector_bytes);
    }

publish_result:
    /* A numerically unsuccessful run is still a valid optimization result:
     * preserve the last accepted parameters and report why iteration stopped.
     * API/setup failures take the separate path below and publish no result.
     */
    output->termination = termination;
    numerus_optimizer_release_workspace(
        gradient_values, new_gradient, direction, trial_parameters,
        step_vector, gradient_delta, hessian_gradient, inverse_hessian,
        next_inverse_hessian
    );
    *result = output;
    return NUMERUS_OPTIMIZER_SUCCESS;

api_failure:
    numerus_optimizer_release_workspace(
        gradient_values, new_gradient, direction, trial_parameters,
        step_vector, gradient_delta, hessian_gradient, inverse_hessian,
        next_inverse_hessian
    );
    numerus_optimizer_result_destroy(output);
    return api_status;
}

void numerus_optimizer_result_destroy(numerus_optimizer_result *result)
{
    if (result != NULL) {
        if (result->parameters != NULL) {
            numerus_optimizer_free(result->parameters);
        }
        numerus_optimizer_free(result);
    }
}

size_t numerus_optimizer_result_parameter_count(
    const numerus_optimizer_result *result
)
{
    return result == NULL ? 0 : result->parameter_count;
}

numerus_optimizer_status numerus_optimizer_result_get_parameter(
    const numerus_optimizer_result *result,
    size_t index,
    double *value
)
{
    if (result == NULL || value == NULL) {
        return NUMERUS_OPTIMIZER_INVALID_ARGUMENT;
    }
    if (index >= result->parameter_count) {
        return NUMERUS_OPTIMIZER_OUT_OF_BOUNDS;
    }

    *value = result->parameters[index];
    return NUMERUS_OPTIMIZER_SUCCESS;
}

bool numerus_optimizer_result_objective_value_valid(
    const numerus_optimizer_result *result
)
{
    return result != NULL && result->objective_value_valid;
}

numerus_optimizer_status numerus_optimizer_result_get_objective(
    const numerus_optimizer_result *result,
    double *value
)
{
    if (result == NULL || value == NULL) {
        return NUMERUS_OPTIMIZER_INVALID_ARGUMENT;
    }
    if (!result->objective_value_valid) {
        return NUMERUS_OPTIMIZER_NO_OBJECTIVE_VALUE;
    }

    *value = result->objective_value;
    return NUMERUS_OPTIMIZER_SUCCESS;
}

numerus_optimizer_status numerus_optimizer_result_get_termination(
    const numerus_optimizer_result *result,
    numerus_optimizer_termination *termination
)
{
    if (result == NULL || termination == NULL) {
        return NUMERUS_OPTIMIZER_INVALID_ARGUMENT;
    }

    *termination = result->termination;
    return NUMERUS_OPTIMIZER_SUCCESS;
}

size_t numerus_optimizer_result_iterations(
    const numerus_optimizer_result *result
)
{
    return result == NULL ? 0 : result->iterations;
}

size_t numerus_optimizer_result_objective_evaluations(
    const numerus_optimizer_result *result
)
{
    return result == NULL ? 0 : result->objective_evaluations;
}

size_t numerus_optimizer_result_gradient_evaluations(
    const numerus_optimizer_result *result
)
{
    return result == NULL ? 0 : result->gradient_evaluations;
}
