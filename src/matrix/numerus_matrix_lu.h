/**
 * @file numerus_matrix_lu.h
 * @brief Private LU factorization API for Matrix algorithms and tests.
 */
#ifndef NUMERUS_MATRIX_LU_H
#define NUMERUS_MATRIX_LU_H

#include "numerus_matrix.h"

typedef struct numerus_matrix_lu_factorization numerus_matrix_lu_factorization;

/**
 * Factor a square Matrix using Gaussian elimination with partial pivoting.
 *
 * The factorization owns a packed LU buffer and a row permutation and is
 * reference-counted. It does not retain the source Matrix. A zero pivot is
 * skipped rather than treated as an API failure; rank counts non-zero pivots
 * under this exact-zero policy. On failure, *factorization remains NULL.
 */
numerus_matrix_status numerus_matrix_lu_factorize(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization **factorization
);

numerus_matrix_status numerus_matrix_lu_retain(
    numerus_matrix_lu_factorization *factorization
);

/** Release one owned reference to a factorization. */
void numerus_matrix_lu_destroy(numerus_matrix_lu_factorization *factorization);

/** Return factorization dimension, or zero for NULL. */
size_t numerus_matrix_lu_size(
    const numerus_matrix_lu_factorization *factorization
);

/** Return the count of non-zero pivots under the exact-zero policy. */
size_t numerus_matrix_lu_rank(
    const numerus_matrix_lu_factorization *factorization
);

/** Return +1 or -1 for permutation parity, or zero for NULL. */
int numerus_matrix_lu_permutation_sign(
    const numerus_matrix_lu_factorization *factorization
);

/**
 * Return the original source row moved to the requested factor row.
 * The output is unchanged on invalid input or out-of-bounds access.
 */
numerus_matrix_status numerus_matrix_lu_get_permutation(
    const numerus_matrix_lu_factorization *factorization,
    size_t factor_row,
    size_t *source_row
);

/** Read L(row,column), where L has a unit diagonal. */
numerus_matrix_status numerus_matrix_lu_get_lower(
    const numerus_matrix_lu_factorization *factorization,
    size_t row,
    size_t column,
    double *value
);

/** Read U(row,column), the upper-triangular factor. */
numerus_matrix_status numerus_matrix_lu_get_upper(
    const numerus_matrix_lu_factorization *factorization,
    size_t row,
    size_t column,
    double *value
);

#endif
