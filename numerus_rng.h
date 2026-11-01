/**
 * @file numerus_rng.h
 * @brief Compatibility include forwarding RNG declarations to src/statistics/numerus_rng.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_RNG_ROOT_COMPAT_H
#define NUMERUS_RNG_ROOT_COMPAT_H
#include "src/statistics/numerus_rng.h"
#endif
