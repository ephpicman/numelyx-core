/**
 * @file numelyx_core_matrix_lu.h
 * @brief Private LU factorization API for Matrix algorithms and tests.
 */
#ifndef NUMERUS_MATRIX_LU_H
#define NUMERUS_MATRIX_LU_H

#include "numelyx_core_matrix.h"

typedef struct numelyx_core_matrix_lu_factorization numelyx_core_matrix_lu_factorization;

/**
 * Factor a square Matrix using LAPACK's partial-pivoted LU routine for
 * nonsingular finite inputs.
 *
 * The factorization owns a packed LU buffer and a row permutation and is
 * reference-counted with atomic lifetime management. Concurrent retain/release
 * is supported when each caller already owns a valid reference. It does not
 * retain the source Matrix. Singular matrices, non-finite inputs, and
 * dimensions outside the supported LAPACK integer range use the native
 * implementation. A zero pivot is skipped rather than treated as an API
 * failure; rank counts non-zero diagonal pivots under the exact-zero policy.
 * On failure, *factorization remains NULL.
 */
numelyx_core_matrix_status numelyx_core_matrix_lu_factorize(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix_lu_factorization **factorization
);

numelyx_core_matrix_status numelyx_core_matrix_lu_retain(
    numelyx_core_matrix_lu_factorization *factorization
);

/** Release one owned reference to a factorization. */
void numelyx_core_matrix_lu_destroy(numelyx_core_matrix_lu_factorization *factorization);

/** Return factorization dimension, or zero for NULL. */
size_t numelyx_core_matrix_lu_size(
    const numelyx_core_matrix_lu_factorization *factorization
);

/** Return the count of non-zero pivots under the exact-zero policy. */
size_t numelyx_core_matrix_lu_rank(
    const numelyx_core_matrix_lu_factorization *factorization
);

/** Return +1 or -1 for permutation parity, or zero for NULL. */
int numelyx_core_matrix_lu_permutation_sign(
    const numelyx_core_matrix_lu_factorization *factorization
);

/**
 * Return the original source row moved to the requested factor row.
 * The output is unchanged on invalid input or out-of-bounds access.
 */
numelyx_core_matrix_status numelyx_core_matrix_lu_get_permutation(
    const numelyx_core_matrix_lu_factorization *factorization,
    size_t factor_row,
    size_t *source_row
);

/** Read L(row,column), where L has a unit diagonal. */
numelyx_core_matrix_status numelyx_core_matrix_lu_get_lower(
    const numelyx_core_matrix_lu_factorization *factorization,
    size_t row,
    size_t column,
    double *value
);

/** Read U(row,column), the upper-triangular factor. */
numelyx_core_matrix_status numelyx_core_matrix_lu_get_upper(
    const numelyx_core_matrix_lu_factorization *factorization,
    size_t row,
    size_t column,
    double *value
);

#endif
