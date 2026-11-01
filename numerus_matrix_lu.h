/**
 * @file numerus_matrix_lu.h
 * @brief Compatibility include forwarding LU declarations to src/matrix/numerus_matrix_lu.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_MATRIX_LU_ROOT_COMPAT_H
#define NUMERUS_MATRIX_LU_ROOT_COMPAT_H
#include "src/matrix/numerus_matrix_lu.h"
#endif
