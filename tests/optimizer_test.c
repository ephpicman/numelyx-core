/**
 * @file optimizer_test.c
 * @brief Native tests for optimizer convergence, stopping reasons, callback handling, and result accessors.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_optimizer.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static long fail_after_allocations = -1;
static size_t live_allocations = 0;

void *numerus_optimizer_test_alloc(size_t size)
{
    void *pointer;

    if (fail_after_allocations == 0) {
        fail_after_allocations = -1;
        return NULL;
    }
    if (fail_after_allocations > 0) {
        fail_after_allocations--;
    }

    pointer = malloc(size);
    if (pointer != NULL) {
        live_allocations++;
    }
    return pointer;
}

void numerus_optimizer_test_free(void *pointer)
{
    if (pointer != NULL) {
        assert(live_allocations > 0);
        live_allocations--;
        free(pointer);
    }
}

typedef struct {
    double target_x;
    double target_y;
    double y_weight;
} quadratic2_context;

static bool quadratic2_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    const quadratic2_context *settings = context;
    double dx;
    double dy;

    if (count != 2 || value == NULL) {
        return false;
    }

    dx = parameters[0] - settings->target_x;
    dy = parameters[1] - settings->target_y;
    *value = dx * dx + settings->y_weight * dy * dy;
    return true;
}

static bool quadratic2_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    const quadratic2_context *settings = context;

    if (count != 2 || gradient == NULL) {
        return false;
    }

    gradient[0] = 2.0 * (parameters[0] - settings->target_x);
    gradient[1] = 2.0 * settings->y_weight *
        (parameters[1] - settings->target_y);
    return true;
}

typedef struct {
    int mode;
    double target;
    double required_initial;
} scalar_context;

enum {
    SCALAR_QUADRATIC = 0,
    SCALAR_FAIL_ALWAYS,
    SCALAR_NONFINITE_OBJECTIVE,
    SCALAR_FAIL_OFF_INITIAL
};


static bool constant_finite_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    (void) parameters;
    (void) context;

    if (count != 1 || value == NULL) {
        return false;
    }
    *value = 0.0;
    return true;
}

static bool opposite_extreme_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    (void) context;

    if (count != 1 || value == NULL) {
        return false;
    }
    if (parameters[0] > 0.0) {
        *value = DBL_MAX;
    } else if (parameters[0] < 0.0) {
        *value = -DBL_MAX;
    } else {
        *value = 0.0;
    }
    return true;
}

static bool scalar_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    const scalar_context *settings = context;
    double delta;

    if (count != 1 || value == NULL) {
        return false;
    }
    if (settings->mode == SCALAR_FAIL_ALWAYS) {
        return false;
    }
    if (settings->mode == SCALAR_NONFINITE_OBJECTIVE) {
        *value = NAN;
        return true;
    }
    if (settings->mode == SCALAR_FAIL_OFF_INITIAL &&
        parameters[0] != settings->required_initial) {
        return false;
    }

    delta = parameters[0] - settings->target;
    *value = delta * delta;
    return true;
}

static bool scalar_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    const scalar_context *settings = context;

    if (count != 1 || gradient == NULL) {
        return false;
    }
    gradient[0] = 2.0 * (parameters[0] - settings->target);
    return true;
}

static bool failing_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    (void) parameters;
    (void) count;
    (void) context;
    (void) gradient;
    return false;
}

static bool nonfinite_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    (void) parameters;
    (void) context;
    if (count != 1 || gradient == NULL) {
        return false;
    }
    gradient[0] = NAN;
    return true;
}

static bool rosenbrock_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    double x;
    double y;
    double residual;

    (void) context;
    if (count != 2 || value == NULL) {
        return false;
    }

    x = parameters[0];
    y = parameters[1];
    residual = y - x * x;
    *value = (1.0 - x) * (1.0 - x) + 100.0 * residual * residual;
    return true;
}

static bool rosenbrock_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    double x;
    double y;
    double residual;

    (void) context;
    if (count != 2 || gradient == NULL) {
        return false;
    }

    x = parameters[0];
    y = parameters[1];
    residual = y - x * x;
    gradient[0] = -2.0 * (1.0 - x) - 400.0 * x * residual;
    gradient[1] = 200.0 * residual;
    return true;
}

static numerus_optimizer_result *minimize(
    numerus_objective_fn objective,
    numerus_gradient_fn gradient,
    const void *context,
    const double *initial,
    size_t count,
    numerus_optimizer_options *options
)
{
    numerus_optimizer_result *result = NULL;

    assert(numerus_optimizer_minimize(
        objective, gradient, context, initial, count, options, &result
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(result != NULL);
    return result;
}

static void test_quadratic_with_analytic_gradient_and_result_fields(void)
{
    const double initial[] = {-5.0, 8.0};
    quadratic2_context context = {3.0, -2.0, 4.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result;
    numerus_optimizer_termination termination;
    double x;
    double y;
    double objective_value;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    result = minimize(
        quadratic2_objective, quadratic2_gradient, &context,
        initial, 2, &options
    );

    assert(numerus_optimizer_result_parameter_count(result) == 2);
    assert(numerus_optimizer_result_get_parameter(result, 0, &x) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numerus_optimizer_result_get_parameter(result, 1, &y) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(fabs(x - 3.0) < 1e-6);
    assert(fabs(y + 2.0) < 1e-6);
    assert(numerus_optimizer_result_get_objective(result, &objective_value) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(objective_value < 1e-10);
    assert(numerus_optimizer_result_get_termination(result, &termination) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_CONVERGED_GRADIENT);
    assert(numerus_optimizer_result_iterations(result) > 0);
    assert(numerus_optimizer_result_objective_evaluations(result) > 0);
    assert(numerus_optimizer_result_gradient_evaluations(result) > 0);

    /* Caller input is borrowed read-only and never modified. */
    assert(initial[0] == -5.0 && initial[1] == 8.0);
    x = 123.0;
    assert(numerus_optimizer_result_get_parameter(
        result, 2, &x
    ) == NUMERUS_OPTIMIZER_OUT_OF_BOUNDS);
    assert(x == 123.0);

    numerus_optimizer_result_destroy(result);
}

static void test_finite_difference_and_ill_conditioned_quadratic(void)
{
    const double initial_scalar[] = {10.0};
    scalar_context scalar = {SCALAR_QUADRATIC, 2.0, 0.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result;
    double x;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    result = minimize(
        scalar_objective, NULL, &scalar, initial_scalar, 1, &options
    );
    assert(numerus_optimizer_result_get_parameter(result, 0, &x) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(fabs(x - 2.0) < 1e-5);
    numerus_optimizer_result_destroy(result);

    {
        const double initial[] = {10.0, 10.0};
        quadratic2_context scaled = {1.0, -2.0, 1e-3};

        result = minimize(
            quadratic2_objective, quadratic2_gradient, &scaled,
            initial, 2, &options
        );
        assert(numerus_optimizer_result_get_parameter(result, 0, &x) ==
            NUMERUS_OPTIMIZER_SUCCESS);
        assert(fabs(x - 1.0) < 1e-5);
        assert(numerus_optimizer_result_get_parameter(result, 1, &x) ==
            NUMERUS_OPTIMIZER_SUCCESS);
        assert(fabs(x + 2.0) < 1e-4);
        numerus_optimizer_result_destroy(result);
    }
}

static void test_rosenbrock_reference(void)
{
    const double initial[] = {-1.2, 1.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result;
    double x;
    double y;
    double objective_value;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.max_iterations = 3000;
    options.gradient_tolerance = 1e-7;

    result = minimize(
        rosenbrock_objective, rosenbrock_gradient, NULL,
        initial, 2, &options
    );
    assert(numerus_optimizer_result_get_parameter(result, 0, &x) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numerus_optimizer_result_get_parameter(result, 1, &y) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numerus_optimizer_result_get_objective(result, &objective_value) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(fabs(x - 1.0) < 1e-4);
    assert(fabs(y - 1.0) < 1e-4);
    assert(objective_value < 1e-8);
    numerus_optimizer_result_destroy(result);
}

static void test_iteration_limit_and_small_step_are_not_convergence(void)
{
    const double initial[] = {10.0, 10.0};
    quadratic2_context context = {1.0, 2.0, 4.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result;
    numerus_optimizer_termination termination;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.max_iterations = 1;
    result = minimize(
        quadratic2_objective, quadratic2_gradient, &context,
        initial, 2, &options
    );
    assert(numerus_optimizer_result_get_termination(result, &termination) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_ITERATION_LIMIT);
    numerus_optimizer_result_destroy(result);

    {
        const double scalar_initial[] = {10.0};
        scalar_context scalar = {SCALAR_QUADRATIC, 1.0, 0.0};

        assert(numerus_optimizer_default_options(&options) ==
            NUMERUS_OPTIMIZER_SUCCESS);
        options.step_tolerance = 1e6;
        result = minimize(
            scalar_objective, scalar_gradient, &scalar,
            scalar_initial, 1, &options
        );
        assert(numerus_optimizer_result_get_termination(
            result, &termination
        ) == NUMERUS_OPTIMIZER_SUCCESS);
        assert(termination == NUMERUS_OPTIMIZER_STEP_TOO_SMALL);
        assert(numerus_optimizer_result_iterations(result) == 0);
        numerus_optimizer_result_destroy(result);
    }
}

static void test_callback_failures_and_nonfinite_values(void)
{
    const double initial[] = {0.0};
    scalar_context context = {SCALAR_FAIL_ALWAYS, 1.0, 0.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result;
    numerus_optimizer_termination termination;
    double objective_value = 123.0;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    result = minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options
    );
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_OBJECTIVE_CALLBACK_FAILED);
    assert(!numerus_optimizer_result_objective_value_valid(result));
    assert(numerus_optimizer_result_get_objective(
        result, &objective_value
    ) == NUMERUS_OPTIMIZER_NO_OBJECTIVE_VALUE);
    assert(objective_value == 123.0);
    numerus_optimizer_result_destroy(result);

    context.mode = SCALAR_NONFINITE_OBJECTIVE;
    result = minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options
    );
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_NON_FINITE_OBJECTIVE);
    numerus_optimizer_result_destroy(result);

    context.mode = SCALAR_QUADRATIC;
    result = minimize(
        scalar_objective, failing_gradient, &context,
        initial, 1, &options
    );
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_GRADIENT_CALLBACK_FAILED);
    numerus_optimizer_result_destroy(result);

    result = minimize(
        scalar_objective, nonfinite_gradient, &context,
        initial, 1, &options
    );
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT);
    numerus_optimizer_result_destroy(result);

    context.mode = SCALAR_FAIL_OFF_INITIAL;
    context.required_initial = 0.0;
    result = minimize(
        scalar_objective, NULL, &context,
        initial, 1, &options
    );
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED);
    numerus_optimizer_result_destroy(result);

    result = minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options
    );
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_LINE_SEARCH_FAILED);
    numerus_optimizer_result_destroy(result);
}

static void test_invalid_arguments_and_allocation_failures(void)
{
    const double initial[] = {1.0};
    scalar_context context = {SCALAR_QUADRATIC, 0.0, 0.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result = (void *) 1;
    size_t before = live_allocations;

    assert(numerus_optimizer_default_options(NULL) ==
        NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.max_iterations = 0;
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(result == NULL);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    {
        const double nonfinite_initial[] = {INFINITY};
        assert(numerus_optimizer_minimize(
            scalar_objective, scalar_gradient, &context,
            nonfinite_initial, 1, &options, &result
        ) == NUMERUS_OPTIMIZER_NON_FINITE_INPUT);
        assert(result == NULL);
    }

    fail_after_allocations = 0;
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_OUT_OF_MEMORY);
    assert(result == NULL);
    assert(live_allocations == before);

    fail_after_allocations = 2;
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_OUT_OF_MEMORY);
    assert(result == NULL);
    assert(live_allocations == before);
}


static void test_result_accessor_invalid_arguments(void)
{
    const double initial[] = {1.0};
    scalar_context context = {SCALAR_QUADRATIC, 0.0, 0.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result = NULL;
    numerus_optimizer_termination termination = NUMERUS_OPTIMIZER_CONVERGED_GRADIENT;
    double value = 987.0;

    assert(numerus_optimizer_result_parameter_count(NULL) == 0);
    assert(!numerus_optimizer_result_objective_value_valid(NULL));
    assert(numerus_optimizer_result_iterations(NULL) == 0);
    assert(numerus_optimizer_result_objective_evaluations(NULL) == 0);
    assert(numerus_optimizer_result_gradient_evaluations(NULL) == 0);

    assert(numerus_optimizer_result_get_parameter(
        NULL, 0, &value
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(value == 987.0);
    assert(numerus_optimizer_result_get_objective(
        NULL, &value
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(value == 987.0);
    assert(numerus_optimizer_result_get_termination(
        NULL, &termination
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(termination == NUMERUS_OPTIMIZER_CONVERGED_GRADIENT);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(result != NULL);

    assert(numerus_optimizer_result_get_parameter(
        result, 0, NULL
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(numerus_optimizer_result_get_objective(
        result, NULL
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(numerus_optimizer_result_get_termination(
        result, NULL
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);

    numerus_optimizer_result_destroy(result);
}

static void assert_invalid_options(numerus_optimizer_options options)
{
    const double initial[] = {1.0};
    scalar_context context = {SCALAR_QUADRATIC, 0.0, 0.0};
    numerus_optimizer_result *result = (void *) 1;

    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(result == NULL);
}

static void test_all_optimizer_option_boundaries(void)
{
    numerus_optimizer_options options;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.max_line_search_iterations = 0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.gradient_tolerance = 0.0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.minimum_step = 0.0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.step_tolerance = 0.0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.initial_step = 0.0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.minimum_step = options.initial_step * 2.0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.finite_difference_relative_step = 0.0;
    assert_invalid_options(options);

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.step_tolerance = NAN;
    {
        const double initial[] = {1.0};
        scalar_context context = {SCALAR_QUADRATIC, 0.0, 0.0};
        numerus_optimizer_result *result = (void *) 1;

        assert(numerus_optimizer_minimize(
            scalar_objective, scalar_gradient, &context,
            initial, 1, &options, &result
        ) == NUMERUS_OPTIMIZER_NON_FINITE_INPUT);
        assert(result == NULL);
    }
}


static void test_finite_difference_extreme_boundaries(void)
{
    numerus_optimizer_options options;
    numerus_optimizer_result *result = NULL;
    numerus_optimizer_termination termination;
    const double positive_extreme[] = {DBL_MAX};
    const double negative_extreme[] = {-DBL_MAX};
    const double origin[] = {0.0};

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    options.finite_difference_relative_step = 0.5;

    assert(numerus_optimizer_minimize(
        constant_finite_objective, NULL, NULL,
        positive_extreme, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(result != NULL);
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED);
    numerus_optimizer_result_destroy(result);
    result = NULL;

    assert(numerus_optimizer_minimize(
        constant_finite_objective, NULL, NULL,
        negative_extreme, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(result != NULL);
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_GRADIENT_EVALUATION_FAILED);
    numerus_optimizer_result_destroy(result);
    result = NULL;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numerus_optimizer_minimize(
        opposite_extreme_objective, NULL, NULL,
        origin, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(result != NULL);
    assert(numerus_optimizer_result_get_termination(
        result, &termination
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(termination == NUMERUS_OPTIMIZER_NON_FINITE_GRADIENT);
    numerus_optimizer_result_destroy(result);
}


static void test_minimize_invalid_argument_matrix(void)
{
    const double initial[] = {1.0};
    scalar_context context = {SCALAR_QUADRATIC, 0.0, 0.0};
    numerus_optimizer_options options;
    numerus_optimizer_result *result = (void *) 1;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, &options, NULL
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);

    assert(numerus_optimizer_minimize(
        NULL, scalar_gradient, &context,
        initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(result == NULL);

    result = (void *) 1;
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        NULL, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(result == NULL);

    result = (void *) 1;
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 0, &options, &result
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(result == NULL);

    result = (void *) 1;
    assert(numerus_optimizer_minimize(
        scalar_objective, scalar_gradient, &context,
        initial, 1, NULL, &result
    ) == NUMERUS_OPTIMIZER_INVALID_ARGUMENT);
    assert(result == NULL);
}

int main(void)
{
    test_quadratic_with_analytic_gradient_and_result_fields();
    test_finite_difference_and_ill_conditioned_quadratic();
    test_rosenbrock_reference();
    test_iteration_limit_and_small_step_are_not_convergence();
    test_callback_failures_and_nonfinite_values();
    test_invalid_arguments_and_allocation_failures();
    test_result_accessor_invalid_arguments();
    test_all_optimizer_option_boundaries();
    test_finite_difference_extreme_boundaries();
    test_minimize_invalid_argument_matrix();
    assert(live_allocations == 0);
    puts("Generic optimizer tests passed.");
    return 0;
}
