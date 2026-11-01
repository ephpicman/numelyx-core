/**
 * @file rng_variates_test.c
 * @brief Native tests for RNG primitive draws, bounds, and deterministic state behavior.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_rng.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_uniform_reference_and_consumption(void)
{
    numerus_rng *rng = NULL;
    numerus_rng *clone = NULL;
    uint32_t expected_next;
    uint32_t source_next;
    uint32_t clone_next;
    double value = -1.0;
    double expected = 5677329748551934.0 / 9007199254740992.0;

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_uniform(rng, &value) == NUMERUS_RNG_SUCCESS);
    assert(value == expected);
    assert(value >= 0.0 && value < 1.0);

    assert(numerus_rng_next_u32(clone, &expected_next) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &expected_next) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(rng, &source_next) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &clone_next) == NUMERUS_RNG_SUCCESS);
    assert(source_next == clone_next);

    numerus_rng_destroy(clone);
    numerus_rng_destroy(rng);
}

static void test_normal_reference_and_consumption(void)
{
    numerus_rng *rng = NULL;
    numerus_rng *clone = NULL;
    uint32_t ignored;
    uint32_t source_next;
    uint32_t clone_next;
    double value = 0.0;

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_clone(rng, &clone) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_normal(rng, &value) == NUMERUS_RNG_SUCCESS);
    assert(isfinite(value));
    assert(fabs(value - (-0.13831366462030992)) < 1e-14);

    /* Box-Muller without caching consumes exactly four raw PCG32 outputs. */
    assert(numerus_rng_next_u32(clone, &ignored) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &ignored) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &ignored) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &ignored) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(rng, &source_next) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_next_u32(clone, &clone_next) == NUMERUS_RNG_SUCCESS);
    assert(source_next == clone_next);

    numerus_rng_destroy(clone);
    numerus_rng_destroy(rng);
}

static void test_uniform_range_and_normal_sanity(void)
{
    numerus_rng *rng = NULL;
    size_t index;
    double mean = 0.0;
    double sum_squared_deviations = 0.0;
    const size_t sample_count = 10000;

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    for (index = 0; index < sample_count; index++) {
        double value;
        assert(numerus_rng_uniform(rng, &value) == NUMERUS_RNG_SUCCESS);
        assert(isfinite(value));
        assert(value >= 0.0 && value < 1.0);
    }
    numerus_rng_destroy(rng);

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    for (index = 0; index < sample_count; index++) {
        double value;
        double delta;
        double updated_delta;

        assert(numerus_rng_normal(rng, &value) == NUMERUS_RNG_SUCCESS);
        assert(isfinite(value));
        delta = value - mean;
        mean += delta / (double) (index + 1);
        updated_delta = value - mean;
        sum_squared_deviations += delta * updated_delta;
    }

    {
        double variance = sum_squared_deviations / (double) (sample_count - 1);
        assert(fabs(mean) < 0.05);
        assert(variance > 0.9 && variance < 1.1);
    }

    numerus_rng_destroy(rng);
}

static void test_invalid_arguments_do_not_advance_state(void)
{
    numerus_rng *rng = NULL;
    uint32_t first_raw;
    double value = 123.0;

    assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
    assert(numerus_rng_uniform(NULL, &value) ==
        NUMERUS_RNG_INVALID_ARGUMENT);
    assert(value == 123.0);
    assert(numerus_rng_normal(rng, NULL) == NUMERUS_RNG_INVALID_ARGUMENT);
    assert(numerus_rng_next_u32(rng, &first_raw) == NUMERUS_RNG_SUCCESS);
    assert(first_raw == UINT32_C(0xa15c02b7));
    numerus_rng_destroy(rng);
}

int main(void)
{
    test_uniform_reference_and_consumption();
    test_normal_reference_and_consumption();
    test_uniform_range_and_normal_sanity();
    test_invalid_arguments_do_not_advance_state();
    puts("RNG variate tests passed.");
    return 0;
}
