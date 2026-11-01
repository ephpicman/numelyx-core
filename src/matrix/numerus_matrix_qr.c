/**
 * @file numerus_matrix_qr.c
 * @brief QR factorization and orthogonalization helpers.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_qr.h"
#include "numerus_size.h"
#include "numerus_numeric.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_qr_alloc(size) emalloc(size)
# define matrix_qr_free(pointer) efree(pointer)
#else
# define matrix_qr_alloc(size) malloc(size)
# define matrix_qr_free(pointer) free(pointer)
#endif

static numerus_matrix_status matrix_qr_decompose_internal(
    const numerus_matrix *matrix,
    bool pivot_columns,
    numerus_matrix **q,
    numerus_matrix **r,
    size_t *permutation_out,
    size_t *numerical_rank
)
{
    size_t rows;
    size_t columns;
    size_t rank_bound;
    size_t work_count;
    size_t q_count;
    size_t r_count;
    size_t work_bytes;
    size_t tau_bytes;
    size_t q_bytes;
    size_t r_bytes;
    size_t permutation_bytes = 0;
    size_t row;
    size_t column;
    size_t step;
    double *work = NULL;
    double *tau = NULL;
    double *q_values = NULL;
    double *r_values = NULL;
    size_t *permutation_values = NULL;
    double scale = 0.0;
    double relative_threshold;
    size_t rank_result = 0;
    numerus_matrix_status status = NUMERUS_MATRIX_SUCCESS;

    if (q != NULL) {
        *q = NULL;
    }
    if (r != NULL && r != q) {
        *r = NULL;
    }
    if (q == NULL || r == NULL || q == r || matrix == NULL ||
        (pivot_columns && (permutation_out == NULL || numerical_rank == NULL))) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    rank_bound = rows < columns ? rows : columns;

    if (!numerus_size_multiply(rows, columns, &work_count) ||
        !numerus_size_multiply(work_count, sizeof(*work), &work_bytes) ||
        !numerus_size_multiply(rows, rank_bound, &q_count) ||
        !numerus_size_multiply(q_count, sizeof(*q_values), &q_bytes) ||
        !numerus_size_multiply(rank_bound, columns, &r_count) ||
        !numerus_size_multiply(r_count, sizeof(*r_values), &r_bytes) ||
        !numerus_size_multiply(rank_bound, sizeof(*tau), &tau_bytes) ||
        (pivot_columns && !numerus_size_multiply(
            columns, sizeof(*permutation_values), &permutation_bytes
        ))) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    work = matrix_qr_alloc(work_bytes);
    tau = matrix_qr_alloc(tau_bytes);
    q_values = matrix_qr_alloc(q_bytes);
    r_values = matrix_qr_alloc(r_bytes);
    if (pivot_columns) {
        permutation_values = matrix_qr_alloc(permutation_bytes);
    }
    if (work == NULL || tau == NULL || q_values == NULL || r_values == NULL ||
        (pivot_columns && permutation_values == NULL)) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;
            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row * columns + column] = value;
            if (fabs(value) > scale) {
                scale = fabs(value);
            }
        }
    }
    /* Track the current-to-original column mapping separately from the
     * numerical work buffer; this is the P in A P = Q R.
     */
    if (pivot_columns) {
        for (column = 0; column < columns; column++) {
            permutation_values[column] = column;
        }
    }

    /*
     * Store each Householder vector below R's diagonal. The normalized
     * denominator avoids forming alpha - beta, which can overflow even when
     * both alpha and the column norm are finite.
     */
    for (step = 0; step < rank_bound; step++) {
        double norm = 0.0;
        double alpha;
        double beta;
        double denominator_ratio;

        /* Column pivoting chooses the largest remaining trailing-column norm.
         * This improves rank revelation and numerical stability; the same
         * swap must be applied to the permutation map as to the data columns.
         */
        if (pivot_columns) {
            size_t pivot_column = step;
            double best_norm = -1.0;
            size_t candidate_column;

            for (candidate_column = step;
                 candidate_column < columns;
                 candidate_column++) {
                double candidate_norm = 0.0;

                for (row = step; row < rows; row++) {
                    candidate_norm = hypot(
                        candidate_norm, work[row * columns + candidate_column]
                    );
                    if (!isfinite(candidate_norm)) {
                        status = NUMERUS_MATRIX_NON_FINITE;
                        goto cleanup;
                    }
                }
                if (candidate_norm > best_norm) {
                    best_norm = candidate_norm;
                    pivot_column = candidate_column;
                }
            }

            if (pivot_column != step) {
                size_t swap_row;
                size_t temporary_index;

                for (swap_row = 0; swap_row < rows; swap_row++) {
                    double temporary = work[swap_row * columns + step];
                    work[swap_row * columns + step] =
                        work[swap_row * columns + pivot_column];
                    work[swap_row * columns + pivot_column] = temporary;
                }
                temporary_index = permutation_values[step];
                permutation_values[step] = permutation_values[pivot_column];
                permutation_values[pivot_column] = temporary_index;
            }
        }

        alpha = work[step * columns + step];
        for (row = step; row < rows; row++) {
            norm = hypot(norm, work[row * columns + step]);
            if (!isfinite(norm)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }

        if (norm == 0.0) {
            tau[step] = 0.0;
            continue;
        }

        /* Choose beta with sign opposite alpha to avoid cancellation in the
         * reflector construction. tau and the stored subdiagonal entries
         * encode H = I - tau*v*vᵀ without allocating each reflector.
         */
        beta = -copysign(norm, alpha);
        tau[step] = 1.0 - alpha / beta;
        denominator_ratio = alpha / norm - beta / norm;
        if (!isfinite(tau[step]) || denominator_ratio == 0.0 ||
            !isfinite(denominator_ratio)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }

        for (row = step + 1; row < rows; row++) {
            work[row * columns + step] =
                (work[row * columns + step] / norm) / denominator_ratio;
        }
        work[step * columns + step] = beta;

        for (column = step + 1; column < columns; column++) {
            double dot = work[step * columns + column];

            for (row = step + 1; row < rows; row++) {
                dot += work[row * columns + step] *
                    work[row * columns + column];
                if (!isfinite(dot)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }

            dot *= tau[step];
            if (!isfinite(dot)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[step * columns + column] -= dot;
            if (!isfinite(work[step * columns + column])) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }

            for (row = step + 1; row < rows; row++) {
                work[row * columns + column] -=
                    work[row * columns + step] * dot;
                if (!isfinite(work[row * columns + column])) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
        }
    }

    /* Q = H0 H1 ... H(k-1); apply reflectors to basis vectors in reverse. */
    for (column = 0; column < rank_bound; column++) {
        for (row = 0; row < rows; row++) {
            q_values[row * rank_bound + column] = row == column ? 1.0 : 0.0;
        }

        step = rank_bound;
        while (step > 0) {
            double dot;
            size_t index;

            step--;
            if (tau[step] == 0.0) {
                continue;
            }

            dot = q_values[step * rank_bound + column];
            for (row = step + 1; row < rows; row++) {
                dot += work[row * columns + step] *
                    q_values[row * rank_bound + column];
                if (!isfinite(dot)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
            dot *= tau[step];
            if (!isfinite(dot)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }

            q_values[step * rank_bound + column] -= dot;
            for (index = step + 1; index < rows; index++) {
                q_values[index * rank_bound + column] -=
                    work[index * columns + step] * dot;
                if (!isfinite(q_values[index * rank_bound + column])) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
        }
    }

    /* Rank is estimated from the R diagonal relative to the largest input
     * magnitude and matrix dimensions. It is a numerical threshold, not an
     * exact symbolic rank test; unpivoted QR deliberately skips this estimate.
     */
    if (pivot_columns && scale > 0.0) {
        relative_threshold = NUMERUS_EPSILON *
            (double) (rows > columns ? rows : columns);
        for (step = 0; step < rank_bound; step++) {
            if (fabs(work[step * columns + step]) / scale <=
                relative_threshold) {
                break;
            }
            rank_result++;
        }
    }

    for (row = 0; row < rank_bound; row++) {
        for (column = 0; column < columns; column++) {
            r_values[row * columns + column] =
                row <= column ? work[row * columns + column] : 0.0;
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, rank_bound, q_values, q
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rank_bound, columns, r_values, r
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(*q);
        *q = NULL;
        goto cleanup;
    }
    if (pivot_columns) {
        for (column = 0; column < columns; column++) {
            permutation_out[column] = permutation_values[column];
        }
        *numerical_rank = rank_result;
    }

cleanup:
    /* Q and R are published as a pair: if construction of R fails, Q is
     * destroyed above. Temporary buffers are always released on this path.
     */
    if (r_values != NULL) matrix_qr_free(r_values);
    if (q_values != NULL) matrix_qr_free(q_values);
    if (tau != NULL) matrix_qr_free(tau);
    if (work != NULL) matrix_qr_free(work);
    if (permutation_values != NULL) matrix_qr_free(permutation_values);
    return status;
}

numerus_matrix_status numerus_matrix_qr_decompose(
    const numerus_matrix *matrix,
    numerus_matrix **q,
    numerus_matrix **r
)
{
    return matrix_qr_decompose_internal(
        matrix, false, q, r, NULL, NULL
    );
}

numerus_matrix_status numerus_matrix_qr_decompose_pivoted(
    const numerus_matrix *matrix,
    numerus_matrix **q,
    numerus_matrix **r,
    size_t *permutation,
    size_t permutation_count,
    size_t *numerical_rank
)
{
    if (q != NULL) *q = NULL;
    if (r != NULL && r != q) *r = NULL;
    if (matrix == NULL || permutation == NULL || numerical_rank == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (permutation_count != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    return matrix_qr_decompose_internal(
        matrix, true, q, r, permutation, numerical_rank
    );
}
