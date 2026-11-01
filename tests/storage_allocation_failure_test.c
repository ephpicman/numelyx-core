/**
 * @file storage_allocation_failure_test.c
 * @brief Fault-injection tests for Storage constructors and owned-buffer cleanup.
 */

#include "../src/storage/numerus_storage.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void __real_free(void *pointer);

static long allocations_before_failure = -1;

static int should_fail_allocation(void)
{
    if (allocations_before_failure < 0) {
        return 0;
    }
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

static void test_dense_storage_allocation_failures(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    long fail_after;
    int failure_observed = 0;
    int success_observed = 0;

    for (fail_after = 0; fail_after < 8; fail_after++) {
        numerus_storage *storage = (numerus_storage *) 1;
        int status;

        allocations_before_failure = fail_after;
        status = numerus_storage_create_dense(2, 2, values, &storage);

        if (status == NUMERUS_STORAGE_SUCCESS) {
            double value;

            assert(storage != NULL);
            assert(numerus_storage_get(storage, 1, 1, &value) ==
                NUMERUS_STORAGE_SUCCESS);
            assert(value == 4.0);
            numerus_storage_destroy(storage);
            success_observed = 1;
            break;
        }

        assert(status == NUMERUS_STORAGE_OUT_OF_MEMORY);
        assert(storage == NULL);
        failure_observed = 1;
    }

    allocations_before_failure = -1;
    assert(failure_observed);
    assert(success_observed);
}

static void test_sparse_storage_allocation_failures(void)
{
    const numerus_storage_sparse_entry entries[] = {
        {3, 30.0},
        {0, 10.0}
    };
    long fail_after;
    int failure_observed = 0;
    int success_observed = 0;

    for (fail_after = 0; fail_after < 8; fail_after++) {
        numerus_storage *storage = (numerus_storage *) 1;
        int status;

        allocations_before_failure = fail_after;
        status = numerus_storage_create_sparse(
            2, 2, -1.0, entries, 2, &storage
        );

        if (status == NUMERUS_STORAGE_SUCCESS) {
            double value;

            assert(storage != NULL);
            assert(numerus_storage_get(storage, 0, 0, &value) ==
                NUMERUS_STORAGE_SUCCESS);
            assert(value == 10.0);
            assert(numerus_storage_get(storage, 1, 1, &value) ==
                NUMERUS_STORAGE_SUCCESS);
            assert(value == 30.0);
            assert(numerus_storage_get(storage, 0, 1, &value) ==
                NUMERUS_STORAGE_SUCCESS);
            assert(value == -1.0);
            numerus_storage_destroy(storage);
            success_observed = 1;
            break;
        }

        assert(status == NUMERUS_STORAGE_OUT_OF_MEMORY);
        assert(storage == NULL);
        failure_observed = 1;
    }

    allocations_before_failure = -1;
    assert(failure_observed);
    assert(success_observed);
}

int main(void)
{
    test_dense_storage_allocation_failures();
    test_sparse_storage_allocation_failures();
    allocations_before_failure = -1;

    puts("Storage allocation-failure tests passed.");
    return 0;
}
