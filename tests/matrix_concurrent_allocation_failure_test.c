/**
 * @file matrix_concurrent_allocation_failure_test.c
 * @brief Concurrent cleanup coverage for partially allocated Block Grid views.
 *
 * This test requires GNU ld --wrap and is built only on supported GNU/Linux
 * configurations. Each worker injects its own allocation failure so all
 * workers exercise the same partial-construction path concurrently.
 */
#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include "test_threads.h"

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void __real_free(void *pointer);

static _Thread_local long allocations_before_failure = -1;

static int should_fail_allocation(void)
{
    if (allocations_before_failure < 0) return 0;
    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        return 1;
    }
    allocations_before_failure--;
    return 0;
}

void *__wrap_malloc(size_t size)
{
    return should_fail_allocation() ? NULL : __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size)
{
    return should_fail_allocation() ? NULL : __real_calloc(count, size);
}

void __wrap_free(void *pointer)
{
    __real_free(pointer);
}

typedef struct {
    numelyx_core_matrix *const *blocks;
    atomic_size_t *ready;
    atomic_bool *start;
    int failed;
} worker_args;

static int worker(void *argument)
{
    worker_args *args = (worker_args *) argument;
    size_t iteration;

    atomic_fetch_add_explicit(args->ready, 1, memory_order_release);
    while (!atomic_load_explicit(args->start, memory_order_acquire)) {
        /* Begin allocation-failure scenarios together. */
    }

    for (iteration = 0; iteration < 100; iteration++) {
        numelyx_core_matrix *grid = NULL;
        int status;

        /*
         * Block Grid allocates its object, block pointer array, row offsets,
         * and column offsets before retaining any parent. Failing allocation
         * #3 verifies cleanup of an initialized mutex and a partially built
         * allocation set while other workers do the same.
         */
        allocations_before_failure = 2;
        status = numelyx_core_matrix_create_block_grid(
            args->blocks, 1, 2, &grid
        );
        if (status != NUMERUS_MATRIX_OUT_OF_MEMORY || grid != NULL) {
            args->failed = 1;
            numelyx_core_matrix_destroy(grid);
            allocations_before_failure = -1;
            break;
        }

        allocations_before_failure = -1;
        status = numelyx_core_matrix_create_block_grid(
            args->blocks, 1, 2, &grid
        );
        if (status != NUMERUS_MATRIX_SUCCESS || grid == NULL) {
            args->failed = 1;
            numelyx_core_matrix_destroy(grid);
            break;
        }
        numelyx_core_matrix_destroy(grid);
    }

    return 0;
}

int main(void)
{
    enum { THREAD_COUNT = 8 };
    const double left_value[] = {2.0};
    const double right_value[] = {3.0};
    numelyx_core_matrix *left = NULL;
    numelyx_core_matrix *right = NULL;
    numelyx_core_matrix *blocks[2];
    worker_args arguments[THREAD_COUNT];
    test_thread_t threads[THREAD_COUNT];
    atomic_size_t ready;
    atomic_bool start;
    size_t index;

    atomic_init(&ready, 0);
    atomic_init(&start, false);

    assert(numelyx_core_matrix_create_dense(1, 1, left_value, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 1, right_value, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    blocks[0] = left;
    blocks[1] = right;

    for (index = 0; index < THREAD_COUNT; index++) {
        arguments[index].blocks = blocks;
        arguments[index].ready = &ready;
        arguments[index].start = &start;
        arguments[index].failed = 0;
        assert(test_thread_create(&threads[index], worker, &arguments[index]));
    }

    while (atomic_load_explicit(&ready, memory_order_acquire) != THREAD_COUNT) {
        /* Keep both source matrices alive until every worker has finished. */
    }
    atomic_store_explicit(&start, true, memory_order_release);

    for (index = 0; index < THREAD_COUNT; index++) {
        int result = -1;
        assert(test_thread_join(&threads[index], &result));
        assert(result == 0);
        assert(arguments[index].failed == 0);
    }

    numelyx_core_matrix_destroy(right);
    numelyx_core_matrix_destroy(left);
    return 0;
}
