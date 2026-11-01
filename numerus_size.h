/**
 * @file numerus_size.h
 * @brief Compatibility include forwarding checked-size declarations to src/core/numerus_size.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_SIZE_ROOT_COMPAT_H
#define NUMERUS_SIZE_ROOT_COMPAT_H
#include "src/core/numerus_size.h"
#endif
