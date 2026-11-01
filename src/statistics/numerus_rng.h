/**
 * @file numerus_rng.h
 * @brief RNG state, status values, and deterministic generation API.
 *
 * @details This file belongs to Numerus's internal C implementation. Its
 * declarations and behavior are coordinated with the focused headers in the
 * same subsystem; changes should preserve their documented ownership,
 * validation, and error-reporting contracts.
 */

#ifndef NUMERUS_RNG_H
#define NUMERUS_RNG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Status values for the internal Numerus RNG API.
 */
typedef enum {
    NUMERUS_RNG_SUCCESS = 0,
    NUMERUS_RNG_INVALID_ARGUMENT,
    NUMERUS_RNG_OUT_OF_MEMORY,
    NUMERUS_RNG_NUMERICAL_FAILURE,
    NUMERUS_RNG_SIZE_OVERFLOW
} numerus_rng_status;

/** Opaque mutable PCG-XSH-RR 64/32 state. */
typedef struct numerus_rng numerus_rng;

/**
 * Create a PCG32 state from explicit 64-bit seed and stream selector.
 *
 * Uses the reference PCG seeding procedure. The output handle is set to NULL
 * before allocation and remains NULL on failure.
 */
numerus_rng_status numerus_rng_create(
    uint64_t seed,
    uint64_t stream,
    numerus_rng **rng
);

/**
 * Advance one state transition and return the next exact 32-bit output.
 * On failure, the state and output value are unchanged.
 */
numerus_rng_status numerus_rng_next_u32(
    numerus_rng *rng,
    uint32_t *value
);

/**
 * Clone the exact state and stream increment. Advancing either object does
 * not change the other. The output handle is NULL on failure.
 */
numerus_rng_status numerus_rng_clone(
    const numerus_rng *source,
    numerus_rng **clone
);

/**
 * Return a uniform double in [0, 1) using exactly two raw PCG32 outputs.
 * On invalid arguments, the RNG state and output are unchanged.
 */
numerus_rng_status numerus_rng_uniform(
    numerus_rng *rng,
    double *value
);

/**
 * Return one standard normal variate using Box-Muller without caching.
 * Exactly four raw PCG32 outputs are consumed for a valid call. The output
 * remains unchanged on failure; the state may already have advanced if a
 * platform math failure is detected after consuming those outputs.
 */
numerus_rng_status numerus_rng_normal(
    numerus_rng *rng,
    double *value
);

/**
 * Sample indices from [0, population_size).
 *
 * With replacement, each index is drawn independently. Without replacement,
 * sample_size must not exceed population_size and returned indices are
 * randomly ordered without duplicates. population_size must be positive;
 * sample_size may be zero, in which case *indices is NULL and the RNG is not
 * advanced. The returned array is owned by the caller and must be released
 * with numerus_rng_free_indices(). On failure, *indices remains NULL and the
 * RNG state is unchanged.
 */
numerus_rng_status numerus_rng_sample_indices(
    numerus_rng *rng,
    size_t population_size,
    size_t sample_size,
    bool with_replacement,
    size_t **indices
);

/** Release an index array returned by numerus_rng_sample_indices(); accepts NULL. */
void numerus_rng_free_indices(size_t *indices);

/** Destroy one state; accepts NULL. */
void numerus_rng_destroy(numerus_rng *rng);

#endif
