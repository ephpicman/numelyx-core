/**
 * @file numerus_numeric.h
 * @brief Compatibility include forwarding numeric declarations to src/core/numerus_numeric.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_NUMERIC_ROOT_COMPAT_H
#define NUMERUS_NUMERIC_ROOT_COMPAT_H
#include "src/core/numerus_numeric.h"
#endif
