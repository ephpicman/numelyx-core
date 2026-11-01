#include "../numelyx_core_optimizer.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <threads.h>

typedef struct {
    numelyx_core_optimizer_result *result;
    int failed;
} result_worker_args;

static int result_worker(void *argument)
{
    result_worker_args *args = (result_worker_args *) argument;
    size_t iteration;

    for (iteration = 0; iteration < 10000; iteration++) {
        double parameter = 0.0;
        double objective = 0.0;
        if (numelyx_core_optimizer_result_get_parameter(
                args->result, 0, &parameter
            ) != NUMERUS_OPTIMIZER_SUCCESS ||
            numelyx_core_optimizer_result_get_objective(
                args->result, &objective
            ) != NUMERUS_OPTIMIZER_SUCCESS ||
            !isfinite(parameter) || !isfinite(objective) ||
            fabs(parameter - 3.0) > 1e-4 || objective > 1e-8) {
            args->failed = 1;
            break;
        }
    }

    numelyx_core_optimizer_result_destroy(args->result);
    return 0;
}

static bool objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    double delta;
    (void) context;
    if (count != 1) return false;
    delta = parameters[0] - 3.0;
    *value = delta * delta;
    return true;
}

static bool gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *values
)
{
    (void) context;
    if (count != 1) return false;
    values[0] = 2.0 * (parameters[0] - 3.0);
    return true;
}

int main(void)
{
    enum { THREAD_COUNT = 8 };
    const double initial[] = {-5.0};
    numelyx_core_optimizer_options options;
    numelyx_core_optimizer_result *result = NULL;
    result_worker_args arguments[THREAD_COUNT];
    thrd_t threads[THREAD_COUNT];
    size_t index;

    assert(numelyx_core_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);
    assert(numelyx_core_optimizer_minimize(
        objective, gradient, NULL, initial, 1, &options, &result
    ) == NUMERUS_OPTIMIZER_SUCCESS);
    assert(result != NULL);

    for (index = 0; index < THREAD_COUNT; index++) {
        arguments[index].result = result;
        arguments[index].failed = 0;
        assert(numelyx_core_optimizer_result_retain(result) ==
            NUMERUS_OPTIMIZER_SUCCESS);
        assert(thrd_create(&threads[index], result_worker, &arguments[index]) ==
            thrd_success);
    }
    for (index = 0; index < THREAD_COUNT; index++) {
        int thread_result = -1;
        assert(thrd_join(threads[index], &thread_result) == thrd_success);
        assert(thread_result == 0);
        assert(arguments[index].failed == 0);
    }

    numelyx_core_optimizer_result_destroy(result);
    return 0;
}
