/**
 * @file numelyx_core_rng.c
 * @brief Deterministic pseudo-random number generator state and primitive draws.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#include "numelyx_core_rng.h"
#include "numelyx_core_rng_internal.h"
#include "numelyx_core_lifetime.h"
#include "numelyx_core_mutex.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
# error "Numerus RNG variates require IEEE-754 binary64 double"
#endif

#if defined(NUMERUS_RNG_TEST_ALLOCATOR)
void *numelyx_core_rng_test_alloc(size_t size);
void numelyx_core_rng_test_free(void *pointer);
# define numelyx_core_rng_alloc(size) numelyx_core_rng_test_alloc(size)
# define numelyx_core_rng_free(pointer) numelyx_core_rng_test_free(pointer)
#else
# define numelyx_core_rng_alloc(size) malloc(size)
# define numelyx_core_rng_free(pointer) free(pointer)
#endif

#define NUMERUS_PCG32_MULTIPLIER UINT64_C(6364136223846793005)

struct numelyx_core_rng {
    numelyx_core_lifetime lifetime;
    numelyx_core_mutex mutex;
    uint64_t state;
    uint64_t increment;
};

/*
 * PCG-XSH-RR 64/32 reference transition and output permutation.
 * Unsigned uint64_t wraparound is part of the specified algorithm.
 */
static void rng_destroy_final(void *object);

static uint32_t numelyx_core_rng_next_u32_unchecked(numelyx_core_rng *rng)
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

static void numelyx_core_rng_seed_unchecked(
    numelyx_core_rng *rng,
    uint64_t seed,
    uint64_t stream
)
{
    rng->state = 0u;
    rng->increment = (stream << 1u) | 1u;

    (void) numelyx_core_rng_next_u32_unchecked(rng);
    rng->state += seed;
    (void) numelyx_core_rng_next_u32_unchecked(rng);
}

numelyx_core_rng_status numelyx_core_rng_create(
    uint64_t seed,
    uint64_t stream,
    numelyx_core_rng **rng
)
{
    numelyx_core_rng *result;

    if (rng == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    *rng = NULL;

    result = numelyx_core_rng_alloc(sizeof(*result));
    if (result == NULL) {
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }
    if (!numelyx_core_mutex_init(&result->mutex)) {
        numelyx_core_rng_free(result);
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }
    if (numelyx_core_lifetime_init(
            &result->lifetime, rng_destroy_final
        ) != NUMELYX_CORE_LIFETIME_SUCCESS) {
        numelyx_core_mutex_destroy(&result->mutex);
        numelyx_core_rng_free(result);
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    numelyx_core_rng_seed_unchecked(result, seed, stream);
    *rng = result;
    return NUMERUS_RNG_SUCCESS;
}

numelyx_core_rng_status numelyx_core_rng_next_u32(
    numelyx_core_rng *rng,
    uint32_t *value
)
{
    uint32_t result;

    if (rng == NULL || value == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    if (!numelyx_core_mutex_lock(&rng->mutex)) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    result = numelyx_core_rng_next_u32_unchecked(rng);
    *value = result;
    (void) numelyx_core_mutex_unlock(&rng->mutex);
    return NUMERUS_RNG_SUCCESS;
}

numelyx_core_rng_status numelyx_core_rng_clone(
    const numelyx_core_rng *source,
    numelyx_core_rng **clone
)
{
    numelyx_core_rng *result;

    if (clone == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    *clone = NULL;

    if (source == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    if (!numelyx_core_mutex_lock((numelyx_core_mutex *) &source->mutex)) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    result = numelyx_core_rng_alloc(sizeof(*result));
    if (result == NULL) {
        (void) numelyx_core_mutex_unlock((numelyx_core_mutex *) &source->mutex);
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }
    if (!numelyx_core_mutex_init(&result->mutex)) {
        numelyx_core_rng_free(result);
        (void) numelyx_core_mutex_unlock((numelyx_core_mutex *) &source->mutex);
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }
    if (numelyx_core_lifetime_init(
            &result->lifetime, rng_destroy_final
        ) != NUMELYX_CORE_LIFETIME_SUCCESS) {
        numelyx_core_mutex_destroy(&result->mutex);
        numelyx_core_rng_free(result);
        (void) numelyx_core_mutex_unlock((numelyx_core_mutex *) &source->mutex);
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    result->state = source->state;
    result->increment = source->increment;
    *clone = result;
    (void) numelyx_core_mutex_unlock((numelyx_core_mutex *) &source->mutex);
    return NUMERUS_RNG_SUCCESS;
}

/*
 * Convert exactly 53 random bits to a binary64 value in [0, 1). Scaling by
 * 2^-53 is exact for the integer range represented here.
 */
static uint64_t numelyx_core_rng_next_53_bits(numelyx_core_rng *rng)
{
    uint64_t high = numelyx_core_rng_next_u32_unchecked(rng);
    uint64_t low = numelyx_core_rng_next_u32_unchecked(rng);

    return (high << 21u) | (low >> 11u);
}

numelyx_core_rng_status numelyx_core_rng_uniform(
    numelyx_core_rng *rng,
    double *value
)
{
    uint64_t bits;
    double result;

    if (rng == NULL || value == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }

    if (!numelyx_core_mutex_lock(&rng->mutex)) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    bits = numelyx_core_rng_next_53_bits(rng);
    result = (double) bits / 9007199254740992.0;
    if (!isfinite(result) || result < 0.0 || result >= 1.0) {
        (void) numelyx_core_mutex_unlock(&rng->mutex);
        return NUMERUS_RNG_NUMERICAL_FAILURE;
    }

    *value = result;
    (void) numelyx_core_mutex_unlock(&rng->mutex);
    return NUMERUS_RNG_SUCCESS;
}

numelyx_core_rng_status numelyx_core_rng_normal(
    numelyx_core_rng *rng,
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

    if (!numelyx_core_mutex_lock(&rng->mutex)) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    radial_bits = numelyx_core_rng_next_53_bits(rng);
    angular_bits = numelyx_core_rng_next_53_bits(rng);

    /*
     * Add one before scaling to obtain u1 in (0, 1]. This prevents log(0)
     * while keeping consumption fixed at four raw outputs per normal value.
     */
    u1 = (double) (radial_bits + 1u) / 9007199254740992.0;
    u2 = (double) angular_bits / 9007199254740992.0;
    result = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);

    if (!isfinite(result)) {
        (void) numelyx_core_mutex_unlock(&rng->mutex);
        return NUMERUS_RNG_NUMERICAL_FAILURE;
    }

    *value = result;
    (void) numelyx_core_mutex_unlock(&rng->mutex);
    return NUMERUS_RNG_SUCCESS;
}

int numelyx_core_rng_lock_internal(numelyx_core_rng *rng)
{
    return rng != NULL && numelyx_core_mutex_lock(&rng->mutex);
}

void numelyx_core_rng_unlock_internal(numelyx_core_rng *rng)
{
    if (rng != NULL) (void) numelyx_core_mutex_unlock(&rng->mutex);
}

static void rng_destroy_final(void *object)
{
    numelyx_core_rng *rng = (numelyx_core_rng *) object;
    numelyx_core_mutex_destroy(&rng->mutex);
    numelyx_core_rng_free(rng);
}

numelyx_core_rng_status numelyx_core_rng_retain(numelyx_core_rng *rng)
{
    numelyx_core_lifetime_status status;

    if (rng == NULL) return NUMERUS_RNG_INVALID_ARGUMENT;
    status = numelyx_core_lifetime_retain(&rng->lifetime);
    if (status == NUMELYX_CORE_LIFETIME_SUCCESS) return NUMERUS_RNG_SUCCESS;
    if (status == NUMELYX_CORE_LIFETIME_OVERFLOW) return NUMERUS_RNG_SIZE_OVERFLOW;
    return NUMERUS_RNG_INVALID_ARGUMENT;
}

void numelyx_core_rng_destroy(numelyx_core_rng *rng)
{
    if (rng != NULL) {
        (void) numelyx_core_lifetime_release(&rng->lifetime, rng);
    }
}
