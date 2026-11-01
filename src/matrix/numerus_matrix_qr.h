/**
 * @file numerus_matrix_qr.h
 * @brief Private QR factorization helpers for Matrix algorithms.
 */
#ifndef NUMERUS_MATRIX_QR_H
#define NUMERUS_MATRIX_QR_H

#include "numerus_matrix.h"

/**
 * Compute a reduced column-pivoted Householder QR factorization A P = Q R.
 *
 * permutation[j] is the original source column moved to factor column j.
 * permutation_count must equal the input column count. numerical_rank uses a
 * scale-aware diagonal threshold. Q and R are independent dense Matrices.
 * The permutation and numerical_rank outputs are unchanged on failure.
 */
numerus_matrix_status numerus_matrix_qr_decompose_pivoted(
    const numerus_matrix *matrix,
    numerus_matrix **q,
    numerus_matrix **r,
    size_t *permutation,
    size_t permutation_count,
    size_t *numerical_rank
);

#endif
