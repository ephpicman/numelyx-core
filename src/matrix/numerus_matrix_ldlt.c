/**
 * @file numerus_matrix_ldlt.c
 * @brief Symmetric LDLᵀ factorization and related solve helpers.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_ldlt_alloc(size) emalloc(size)
# define matrix_ldlt_free(pointer) efree(pointer)
#else
# define matrix_ldlt_alloc(size) malloc(size)
# define matrix_ldlt_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_ldlt_decompose(
    const numerus_matrix *matrix,
    numerus_matrix **lower,
    numerus_matrix **diagonal
)
{
    size_t size;
    size_t element_count;
    size_t matrix_bytes;
    size_t diagonal_bytes;
    size_t row;
    size_t column;
    double scale = 0.0;
    double relative_threshold;
    double pivot_threshold;
    double *work = NULL;
    double *lower_values = NULL;
    double *diagonal_values = NULL;
    numerus_matrix_status status;

    if (lower != NULL) *lower = NULL;
    if (diagonal != NULL && diagonal != lower) *diagonal = NULL;
    if (matrix == NULL || lower == NULL || diagonal == NULL || lower == diagonal) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);
    if (!numerus_size_multiply(size, size, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*work), &matrix_bytes) ||
        !numerus_size_multiply(size, sizeof(*diagonal_values), &diagonal_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    work = matrix_ldlt_alloc(matrix_bytes);
    lower_values = matrix_ldlt_alloc(matrix_bytes);
    diagonal_values = matrix_ldlt_alloc(diagonal_bytes);
    if (work == NULL || lower_values == NULL || diagonal_values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;
            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row * size + column] = value;
            lower_values[row * size + column] = row == column ? 1.0 : 0.0;
            if (fabs(value) > scale) scale = fabs(value);
        }
    }

    /* LDLᵀ is only valid for symmetric input. Check symmetry relative to the
     * input scale before factorization; unlike Cholesky, the diagonal D may
     * contain negative pivots, but near-zero pivots still make division unsafe.
     */
    relative_threshold = NUMERUS_EPSILON * (double) size;
    pivot_threshold = scale * relative_threshold;

    if (scale > 0.0) {
        for (row = 0; row < size; row++) {
            for (column = row + 1; column < size; column++) {
                double upper = work[row * size + column] / scale;
                double lower_value = work[column * size + row] / scale;

                if (fabs(upper - lower_value) > relative_threshold) {
                    status = NUMERUS_MATRIX_NOT_SYMMETRIC;
                    goto cleanup;
                }
            }
        }
    }

    /* Unpivoted LDLᵀ recurrence. For each column, subtract contributions
     * from previously computed L and D to obtain D[j,j], then compute the
     * entries below it in L. The unit diagonal of L was initialized above.
     */
    for (column = 0; column < size; column++) {
        size_t k;
        double pivot = work[column * size + column];

        for (k = 0; k < column; k++) {
            double factor = lower_values[column * size + k];
            pivot -= factor * factor * diagonal_values[k];
            if (!isfinite(pivot)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }

        if (fabs(pivot) <= pivot_threshold) {
            status = NUMERUS_MATRIX_PIVOT_TOO_SMALL;
            goto cleanup;
        }
        diagonal_values[column] = pivot;

        for (row = column + 1; row < size; row++) {
            double value = work[row * size + column];

            for (k = 0; k < column; k++) {
                value -= lower_values[row * size + k] *
                    diagonal_values[k] * lower_values[column * size + k];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }

            value /= pivot;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            lower_values[row * size + column] = value;
        }
    }

    /* Publish both factors only after the numeric recurrence succeeds. If
     * creation of D fails after L was created, destroy L and reset its output
     * so callers never receive a half-formed factorization.
     */
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, size, lower_values, lower
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = (numerus_matrix_status) numerus_matrix_create_diagonal(
        size, diagonal_values, diagonal
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(*lower);
        *lower = NULL;
    }

cleanup:
    if (diagonal_values != NULL) matrix_ldlt_free(diagonal_values);
    if (lower_values != NULL) matrix_ldlt_free(lower_values);
    if (work != NULL) matrix_ldlt_free(work);
    return status;
}
