/**
 * @file numerus_rng.c
 * @brief Deterministic pseudo-random number generator state and primitive draws.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#include "numerus_rng.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
# error "Numerus RNG variates require IEEE-754 binary64 double"
#endif

#if defined(NUMERUS_RNG_TEST_ALLOCATOR)
void *numerus_rng_test_alloc(size_t size);
void numerus_rng_test_free(void *pointer);
# define numerus_rng_alloc(size) numerus_rng_test_alloc(size)
# define numerus_rng_free(pointer) numerus_rng_test_free(pointer)
#elif !defined(NUMERUS_RNG_USE_LIBC_ALLOC)
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_rng_alloc(size) emalloc(size)
# define numerus_rng_free(pointer) efree(pointer)
#else
# define numerus_rng_alloc(size) malloc(size)
# define numerus_rng_free(pointer) free(pointer)
#endif

#define NUMERUS_PCG32_MULTIPLIER UINT64_C(6364136223846793005)

struct numerus_rng {
    uint64_t state;
    uint64_t increment;
};

/*
 * PCG-XSH-RR 64/32 reference transition and output permutation.
 * Unsigned uint64_t wraparound is part of the specified algorithm.
 */
static uint32_t numerus_rng_next_u32_unchecked(numerus_rng *rng)
{
    uint64_t old_state = rng->state;
    uint32_t xorshifted;
    uint32_t rotation;

    rng->state = old_state * NUMERUS_PCG32_MULTIPLIER + rng->increment;
    xorshifted = (uint32_t) (((old_state >> 18u) ^ old_state) >> 27u);
    rotation = (uint32_t) (old_state >> 59u);

    return (xorshifted >> rotation) |
        (xorshifted << ((0u - rotation) & 31u));
}

static void numerus_rng_seed_unchecked(
    numerus_rng *rng,
    uint64_t seed,
    uint64_t stream
)
{
    rng->state = 0u;
    rng->increment = (stream << 1u) | 1u;

    (void) numerus_rng_next_u32_unchecked(rng);
    rng->state += seed;
    (void) numerus_rng_next_u32_unchecked(rng);
}

numerus_rng_status numerus_rng_create(
    uint64_t seed,
    uint64_t stream,
    numerus_rng **rng
)
{
    numerus_rng *result;

    if (rng == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    *rng = NULL;

    result = numerus_rng_alloc(sizeof(*result));
    if (result == NULL) {
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }

    numerus_rng_seed_unchecked(result, seed, stream);
    *rng = result;
    return NUMERUS_RNG_SUCCESS;
}

numerus_rng_status numerus_rng_next_u32(
    numerus_rng *rng,
    uint32_t *value
)
{
    uint32_t result;

    if (rng == NULL || value == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    result = numerus_rng_next_u32_unchecked(rng);
    *value = result;
    return NUMERUS_RNG_SUCCESS;
}

numerus_rng_status numerus_rng_clone(
    const numerus_rng *source,
    numerus_rng **clone
)
{
    numerus_rng *result;

    if (clone == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    *clone = NULL;

    if (source == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    result = numerus_rng_alloc(sizeof(*result));
    if (result == NULL) {
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }

    result->state = source->state;
    result->increment = source->increment;
    *clone = result;
    return NUMERUS_RNG_SUCCESS;
}

/*
 * Convert exactly 53 random bits to a binary64 value in [0, 1). Scaling by
 * 2^-53 is exact for the integer range represented here.
 */
static uint64_t numerus_rng_next_53_bits(numerus_rng *rng)
{
    uint64_t high = numerus_rng_next_u32_unchecked(rng);
    uint64_t low = numerus_rng_next_u32_unchecked(rng);

    return (high << 21u) | (low >> 11u);
}

numerus_rng_status numerus_rng_uniform(
    numerus_rng *rng,
    double *value
)
{
    uint64_t bits;
    double result;

    if (rng == NULL || value == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    bits = numerus_rng_next_53_bits(rng);
    result = (double) bits / 9007199254740992.0;
    if (!isfinite(result) || result < 0.0 || result >= 1.0) {
        return NUMERUS_RNG_NUMERICAL_FAILURE;
    }

    *value = result;
    return NUMERUS_RNG_SUCCESS;
}

numerus_rng_status numerus_rng_normal(
    numerus_rng *rng,
    double *value
)
{
    const double two_pi = 6.2831853071795864769252867665590057683943387987502;
    uint64_t radial_bits;
    uint64_t angular_bits;
    double u1;
    double u2;
    double result;

    if (rng == NULL || value == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    radial_bits = numerus_rng_next_53_bits(rng);
    angular_bits = numerus_rng_next_53_bits(rng);

    /*
     * Add one before scaling to obtain u1 in (0, 1]. This prevents log(0)
     * while keeping consumption fixed at four raw outputs per normal value.
     */
    u1 = (double) (radial_bits + 1u) / 9007199254740992.0;
    u2 = (double) angular_bits / 9007199254740992.0;
    result = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);

    if (!isfinite(result)) {
        return NUMERUS_RNG_NUMERICAL_FAILURE;
    }

    *value = result;
    return NUMERUS_RNG_SUCCESS;
}

void numerus_rng_destroy(numerus_rng *rng)
{
    if (rng != NULL) {
        numerus_rng_free(rng);
    }
}
