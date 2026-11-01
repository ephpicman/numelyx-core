/**
 * @file rng_sampling_test.c
 * @brief Native tests for sampling transforms, distribution support, and deterministic seeded output.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_rng.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static bool fail_next_allocation = false;
static size_t live_allocations = 0;

void *numelyx_core_rng_test_alloc(size_t size)
{
    void *pointer;

    if (fail_next_allocation) {
        fail_next_allocation = false;
        return NULL;
    }

    pointer = malloc(size);
    if (pointer != NULL) {
        live_allocations++;
    }
    return pointer;
}

void numelyx_core_rng_test_free(void *pointer)
{
    if (pointer != NULL) {
        assert(live_allocations > 0);
        live_allocations--;
        free(pointer);
    }
}

static void test_with_replacement_reference(void)
{
    static const size_t expected[] = {5, 9, 6, 5, 8, 5, 1, 5};
    numelyx_core_rng *rng = NULL;
    size_t *indices = NULL;
    size_t index;

    assert(numelyx_core_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_sample_indices(
        rng, 10, sizeof(expected) / sizeof(expected[0]), true, &indices
    ) == NUMERUS_RNG_SUCCESS);
    assert(indices != NULL);

    for (index = 0; index < sizeof(expected) / sizeof(expected[0]); index++) {
        assert(indices[index] == expected[index]);
        assert(indices[index] < 10);
    }

    numelyx_core_rng_free_indices(indices);
    numelyx_core_rng_destroy(rng);
    assert(live_allocations == 0);
}

static void test_without_replacement_reference_and_uniqueness(void)
{
    static const size_t expected[] = {5, 7, 3};
    numelyx_core_rng *rng = NULL;
    size_t *indices = NULL;
    size_t first;
    size_t second;
    size_t third;

    assert(numelyx_core_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_sample_indices(rng, 10, 3, false, &indices) ==
        NUMERUS_RNG_SUCCESS);
    assert(indices != NULL);
    assert(indices[0] == expected[0]);
    assert(indices[1] == expected[1]);
    assert(indices[2] == expected[2]);

    first = indices[0];
    second = indices[1];
    third = indices[2];
    assert(first < 10 && second < 10 && third < 10);
    assert(first != second && first != third && second != third);

    numelyx_core_rng_free_indices(indices);
    numelyx_core_rng_destroy(rng);
    assert(live_allocations == 0);
}

static void test_full_and_empty_samples(void)
{
    numelyx_core_rng *rng = NULL;
    size_t *indices = NULL;
    size_t index;
    uint32_t first_raw;

    assert(numelyx_core_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_sample_indices(
        rng, 10, 0, false, &indices
    ) == NUMERUS_RNG_SUCCESS);
    assert(indices == NULL);
    assert(numelyx_core_rng_next_u32(rng, &first_raw) == NUMERUS_RNG_SUCCESS);
    assert(first_raw == UINT32_C(0xa15c02b7));
    numelyx_core_rng_destroy(rng);

    assert(numelyx_core_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_sample_indices(
        rng, 10, 10, false, &indices
    ) == NUMERUS_RNG_SUCCESS);
    assert(indices != NULL);
    for (index = 0; index < 10; index++) {
        size_t other;
        assert(indices[index] < 10);
        for (other = index + 1; other < 10; other++) {
            assert(indices[index] != indices[other]);
        }
    }
    numelyx_core_rng_free_indices(indices);
    numelyx_core_rng_destroy(rng);
    assert(live_allocations == 0);
}

static void test_invalid_sizes_and_overflow_preserve_state(void)
{
    numelyx_core_rng *rng = NULL;
    numelyx_core_rng *clone = NULL;
    size_t *indices = (void *) 1;
    uint32_t first;
    uint32_t second;

    assert(numelyx_core_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);

    assert(numelyx_core_rng_sample_indices(
        rng, 0, 1, true, &indices
    ) == NUMERUS_RNG_INVALID_ARGUMENT);
    assert(indices == NULL);
    assert(numelyx_core_rng_sample_indices(
        rng, 3, 4, false, &indices
    ) == NUMERUS_RNG_INVALID_ARGUMENT);
    assert(indices == NULL);
    assert(numelyx_core_rng_sample_indices(
        rng, 1, SIZE_MAX, true, &indices
    ) == NUMERUS_RNG_SIZE_OVERFLOW);
    assert(indices == NULL);

    assert(numelyx_core_rng_next_u32(rng, &first) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_next_u32(clone, &second) == NUMERUS_RNG_SUCCESS);
    assert(first == second);
    assert(first == UINT32_C(0xa15c02b7));

    assert(numelyx_core_rng_sample_indices(
        NULL, 10, 1, true, &indices
    ) == NUMERUS_RNG_INVALID_ARGUMENT);
    assert(indices == NULL);
    assert(numelyx_core_rng_sample_indices(
        rng, 10, 1, true, NULL
    ) == NUMERUS_RNG_INVALID_ARGUMENT);

    numelyx_core_rng_destroy(clone);
    numelyx_core_rng_destroy(rng);
    assert(live_allocations == 0);
}

static void test_allocation_failure_does_not_advance_or_leak(void)
{
    numelyx_core_rng *rng = NULL;
    numelyx_core_rng *clone = NULL;
    size_t *indices = (void *) 1;
    size_t before;
    uint32_t first;
    uint32_t second;

    assert(numelyx_core_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);
    before = live_allocations;

    fail_next_allocation = true;
    assert(numelyx_core_rng_sample_indices(
        rng, 10, 3, false, &indices
    ) == NUMERUS_RNG_OUT_OF_MEMORY);
    assert(indices == NULL);
    assert(live_allocations == before);

    assert(numelyx_core_rng_next_u32(rng, &first) == NUMERUS_RNG_SUCCESS);
    assert(numelyx_core_rng_next_u32(clone, &second) == NUMERUS_RNG_SUCCESS);
    assert(first == second);

    numelyx_core_rng_destroy(clone);
    numelyx_core_rng_destroy(rng);
    assert(live_allocations == 0);
}

int main(void)
{
    test_with_replacement_reference();
    test_without_replacement_reference_and_uniqueness();
    test_full_and_empty_samples();
    test_invalid_sizes_and_overflow_preserve_state();
    test_allocation_failure_does_not_advance_or_leak();
    puts("RNG sampling tests passed.");
    return 0;
}
