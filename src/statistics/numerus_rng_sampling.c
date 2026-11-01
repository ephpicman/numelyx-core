/**
 * @file numerus_rng_sampling.c
 * @brief Sampling algorithms built on the shared deterministic RNG.
 *
 * @details This translation unit implements sampling transforms on top of the shared RNG primitives. It reuses the RNG state API so seeded sampling remains reproducible and does not maintain an independent generator state.
 */

#include "numerus_rng.h"
#include "numerus_size.h"

#include <stdlib.h>

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

static numerus_rng_status numerus_rng_bounded_index(
    numerus_rng *rng,
    size_t bound,
    size_t *value
)
{
    uint64_t wide_bound;
    uint64_t threshold;
    uint32_t high;
    uint32_t low;
    uint64_t random_value;
    numerus_rng_status status;

    if (rng == NULL || value == NULL || bound == 0) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    if (bound == 1) {
        *value = 0;
        return NUMERUS_RNG_SUCCESS;
    }

    wide_bound = (uint64_t) bound;
    threshold = (UINT64_C(0) - wide_bound) % wide_bound;

    do {
        status = numerus_rng_next_u32(rng, &high);
        if (status != NUMERUS_RNG_SUCCESS) {
            return status;
        }
        status = numerus_rng_next_u32(rng, &low);
        if (status != NUMERUS_RNG_SUCCESS) {
            return status;
        }
        random_value = ((uint64_t) high << 32u) | (uint64_t) low;
    } while (random_value < threshold);

    *value = (size_t) (random_value % wide_bound);
    return NUMERUS_RNG_SUCCESS;
}

numerus_rng_status numerus_rng_sample_indices(
    numerus_rng *rng,
    size_t population_size,
    size_t sample_size,
    bool with_replacement,
    size_t **indices
)
{
    size_t bytes;
    size_t *values;
    size_t index;
    numerus_rng_status status;

    if (indices == NULL) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    *indices = NULL;

    if (rng == NULL || population_size == 0) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    if (!with_replacement && sample_size > population_size) {
        return NUMERUS_RNG_INVALID_ARGUMENT;
    }
    if (sample_size == 0) {
        return NUMERUS_RNG_SUCCESS;
    }
    if (!numerus_size_multiply(sample_size, sizeof(*values), &bytes)) {
        return NUMERUS_RNG_SIZE_OVERFLOW;
    }

    values = numerus_rng_alloc(bytes);
    if (values == NULL) {
        return NUMERUS_RNG_OUT_OF_MEMORY;
    }

    if (with_replacement) {
        for (index = 0; index < sample_size; index++) {
            status = numerus_rng_bounded_index(
                rng, population_size, &values[index]
            );
            if (status != NUMERUS_RNG_SUCCESS) {
                numerus_rng_free(values);
                return status;
            }
        }
    } else {
        size_t selected = 0;

        /*
         * Algorithm S selects a uniform subset in O(population_size) time
         * and O(sample_size) storage. A final Fisher-Yates shuffle makes the
         * output order random rather than leaving selected indices sorted.
         */
        for (index = 0;
             index < population_size && selected < sample_size;
             index++) {
            size_t remaining_population = population_size - index;
            size_t remaining_samples = sample_size - selected;
            bool take = remaining_samples == remaining_population;

            if (!take) {
                size_t draw;

                status = numerus_rng_bounded_index(
                    rng, remaining_population, &draw
                );
                if (status != NUMERUS_RNG_SUCCESS) {
                    numerus_rng_free(values);
                    return status;
                }
                take = draw < remaining_samples;
            }

            if (take) {
                values[selected++] = index;
            }
        }

        if (selected != sample_size) {
            numerus_rng_free(values);
            return NUMERUS_RNG_NUMERICAL_FAILURE;
        }

        index = sample_size;
        while (index > 1) {
            size_t other;
            size_t temporary;

            index--;
            status = numerus_rng_bounded_index(rng, index + 1, &other);
            if (status != NUMERUS_RNG_SUCCESS) {
                numerus_rng_free(values);
                return status;
            }

            temporary = values[index];
            values[index] = values[other];
            values[other] = temporary;
        }
    }

    *indices = values;
    return NUMERUS_RNG_SUCCESS;
}

void numerus_rng_free_indices(size_t *indices)
{
    if (indices != NULL) {
        numerus_rng_free(indices);
    }
}
