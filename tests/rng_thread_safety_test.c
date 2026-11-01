#include "../numelyx_core_rng.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <threads.h>

typedef struct {
    numelyx_core_rng *rng;
    int failed;
} rng_worker_args;

static int rng_worker(void *argument)
{
    rng_worker_args *args = (rng_worker_args *) argument;
    size_t iteration;

    for (iteration = 0; iteration < 1000; iteration++) {
        uint32_t raw;
        double uniform;
        double normal;
        size_t *indices = NULL;
        size_t index;

        if (numelyx_core_rng_next_u32(args->rng, &raw) != NUMERUS_RNG_SUCCESS ||
            numelyx_core_rng_uniform(args->rng, &uniform) != NUMERUS_RNG_SUCCESS ||
            numelyx_core_rng_normal(args->rng, &normal) != NUMERUS_RNG_SUCCESS ||
            !isfinite(uniform) || uniform < 0.0 || uniform >= 1.0 ||
            !isfinite(normal) ||
            numelyx_core_rng_sample_indices(
                args->rng, 17, 5, false, &indices
            ) != NUMERUS_RNG_SUCCESS ||
            indices == NULL) {
            args->failed = 1;
            numelyx_core_rng_free_indices(indices);
            break;
        }

        for (index = 0; index < 5; index++) {
            if (indices[index] >= 17) args->failed = 1;
        }
        numelyx_core_rng_free_indices(indices);
        if (args->failed) break;
    }

    numelyx_core_rng_destroy(args->rng);
    return 0;
}

int main(void)
{
    enum { THREAD_COUNT = 8 };
    numelyx_core_rng *rng = NULL;
    rng_worker_args arguments[THREAD_COUNT];
    thrd_t threads[THREAD_COUNT];
    size_t index;

    assert(numelyx_core_rng_create(12345, 6789, &rng) == NUMERUS_RNG_SUCCESS);

    for (index = 0; index < THREAD_COUNT; index++) {
        arguments[index].rng = rng;
        arguments[index].failed = 0;
        assert(numelyx_core_rng_retain(rng) == NUMERUS_RNG_SUCCESS);
        assert(thrd_create(&threads[index], rng_worker, &arguments[index]) ==
            thrd_success);
    }

    for (index = 0; index < THREAD_COUNT; index++) {
        int result = -1;
        assert(thrd_join(threads[index], &result) == thrd_success);
        assert(result == 0);
        assert(arguments[index].failed == 0);
    }

    numelyx_core_rng_destroy(rng);
    return 0;
}
