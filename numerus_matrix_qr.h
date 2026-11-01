/**
 * @file numerus_matrix_qr.h
 * @brief Compatibility include forwarding QR declarations to src/matrix.
 *
 * @details This header exists for existing native consumers. New internal
 * code should include src/matrix/numerus_matrix_qr.h directly; no separate
 * declarations or implementation are defined here.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_MATRIX_QR_ROOT_COMPAT_H
#define NUMERUS_MATRIX_QR_ROOT_COMPAT_H
#include "src/matrix/numerus_matrix_qr.h"
#endif
