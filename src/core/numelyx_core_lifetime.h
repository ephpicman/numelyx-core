/**
 * @file numelyx_core_lifetime.h
 * @brief Thread-safe intrusive reference counting for shared Core objects.
 */
#ifndef NUMELYX_CORE_LIFETIME_H
#define NUMELYX_CORE_LIFETIME_H

#include <stdatomic.h>
#include <stddef.h>

typedef enum {
    NUMELYX_CORE_LIFETIME_SUCCESS = 0,
    NUMELYX_CORE_LIFETIME_INVALID_ARGUMENT,
    NUMELYX_CORE_LIFETIME_OVERFLOW,
    NUMELYX_CORE_LIFETIME_UNDERFLOW
} numelyx_core_lifetime_status;

typedef struct {
    atomic_size_t reference_count;
    void (*destroy)(void *object);
} numelyx_core_lifetime;

/**
 * Initialize a newly allocated object's lifetime with one owned reference.
 * The object must not be published to another thread before initialization.
 */
numelyx_core_lifetime_status numelyx_core_lifetime_init(
    numelyx_core_lifetime *lifetime,
    void (*destroy)(void *object)
);

/**
 * Acquire another reference. The caller must already hold a valid reference
 * (or otherwise externally guarantee that the object cannot be reclaimed).
 */
numelyx_core_lifetime_status numelyx_core_lifetime_retain(
    numelyx_core_lifetime *lifetime
);

/**
 * Release one owned reference. The final release invokes destroy exactly once.
 * The caller must own the reference being released.
 */
numelyx_core_lifetime_status numelyx_core_lifetime_release(
    numelyx_core_lifetime *lifetime,
    void *object
);

#endif
