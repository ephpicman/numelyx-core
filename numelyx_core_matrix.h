/**
 * @file numelyx_core_matrix.h
 * @brief Compatibility include forwarding Matrix declarations to src/matrix/numelyx_core_matrix.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_MATRIX_ROOT_COMPAT_H
#define NUMERUS_MATRIX_ROOT_COMPAT_H
#include "src/matrix/numelyx_core_matrix.h"
#endif
