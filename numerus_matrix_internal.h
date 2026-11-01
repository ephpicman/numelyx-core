/**
 * @file numerus_matrix_internal.h
 * @brief Compatibility include forwarding internal Matrix declarations to src/matrix/numerus_matrix_internal.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_MATRIX_INTERNAL_ROOT_COMPAT_H
#define NUMERUS_MATRIX_INTERNAL_ROOT_COMPAT_H
#include "src/matrix/numerus_matrix_internal.h"
#endif
