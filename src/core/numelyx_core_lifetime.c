/**
 * @file numelyx_core_lifetime.c
 * @brief Atomic intrusive reference counting.
 */
#include "numelyx_core_lifetime.h"

#include <stdint.h>

numelyx_core_lifetime_status numelyx_core_lifetime_init(
    numelyx_core_lifetime *lifetime,
    void (*destroy)(void *object)
)
{
    if (lifetime == NULL || destroy == NULL) {
        return NUMELYX_CORE_LIFETIME_INVALID_ARGUMENT;
    }

    atomic_init(&lifetime->reference_count, 1);
    lifetime->destroy = destroy;
    return NUMELYX_CORE_LIFETIME_SUCCESS;
}

numelyx_core_lifetime_status numelyx_core_lifetime_retain(
    numelyx_core_lifetime *lifetime
)
{
    size_t current;

    if (lifetime == NULL || lifetime->destroy == NULL) {
        return NUMELYX_CORE_LIFETIME_INVALID_ARGUMENT;
    }

    current = atomic_load_explicit(
        &lifetime->reference_count, memory_order_relaxed
    );

    for (;;) {
        if (current == 0) {
            return NUMELYX_CORE_LIFETIME_INVALID_ARGUMENT;
        }
        if (current == SIZE_MAX) {
            return NUMELYX_CORE_LIFETIME_OVERFLOW;
        }
        if (atomic_compare_exchange_weak_explicit(
                &lifetime->reference_count,
                &current,
                current + 1,
                memory_order_relaxed,
                memory_order_relaxed)) {
            return NUMELYX_CORE_LIFETIME_SUCCESS;
        }
    }
}

numelyx_core_lifetime_status numelyx_core_lifetime_release(
    numelyx_core_lifetime *lifetime,
    void *object
)
{
    size_t current;

    if (lifetime == NULL || object == NULL || lifetime->destroy == NULL) {
        return NUMELYX_CORE_LIFETIME_INVALID_ARGUMENT;
    }

    current = atomic_load_explicit(
        &lifetime->reference_count, memory_order_relaxed
    );

    for (;;) {
        if (current == 0) {
            return NUMELYX_CORE_LIFETIME_UNDERFLOW;
        }
        if (atomic_compare_exchange_weak_explicit(
                &lifetime->reference_count,
                &current,
                current - 1,
                memory_order_release,
                memory_order_relaxed)) {
            if (current == 1) {
                atomic_thread_fence(memory_order_acquire);
                lifetime->destroy(object);
            }
            return NUMELYX_CORE_LIFETIME_SUCCESS;
        }
    }
}
