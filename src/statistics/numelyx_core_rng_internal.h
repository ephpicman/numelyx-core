/**
 * @file numelyx_core_rng_internal.h
 * @brief Private synchronization helpers for composite RNG operations.
 */
#ifndef NUMELYX_CORE_RNG_INTERNAL_H
#define NUMELYX_CORE_RNG_INTERNAL_H

#include "numelyx_core_rng.h"

int numelyx_core_rng_lock_internal(numelyx_core_rng *rng);
void numelyx_core_rng_unlock_internal(numelyx_core_rng *rng);

#endif
