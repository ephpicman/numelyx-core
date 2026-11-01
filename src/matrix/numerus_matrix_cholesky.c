/**
 * @file numerus_matrix_cholesky.c
 * @brief Cholesky factorization for symmetric positive-definite matrices.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_cholesky_alloc(size) emalloc(size)
# define matrix_cholesky_free(pointer) efree(pointer)
#else
# define matrix_cholesky_alloc(size) malloc(size)
# define matrix_cholesky_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_cholesky(
    const numerus_matrix *matrix,
    numerus_matrix **lower
)
{
    size_t size;
    size_t element_count;
    size_t value_bytes;
    size_t row;
    size_t column;
    double scale = 0.0;
    double relative_threshold;
    double pivot_threshold;
    double *work = NULL;
    double *lower_values = NULL;
    numerus_matrix_status status;

    if (lower != NULL) {
        *lower = NULL;
    }
    if (matrix == NULL || lower == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);
    if (!numerus_size_multiply(size, size, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*work), &value_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    work = matrix_cholesky_alloc(value_bytes);
    lower_values = matrix_cholesky_alloc(value_bytes);
    if (work == NULL || lower_values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;
            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row * size + column] = value;
            if (fabs(value) > scale) {
                scale = fabs(value);
            }
            lower_values[row * size + column] = 0.0;
        }
    }

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

    for (row = 0; row < size; row++) {
        for (column = 0; column <= row; column++) {
            size_t k;
            double value = work[row * size + column];

            for (k = 0; k < column; k++) {
                value -= lower_values[row * size + k] *
                    lower_values[column * size + k];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }

            if (row == column) {
                if (value <= pivot_threshold) {
                    status = NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE;
                    goto cleanup;
                }
                lower_values[row * size + column] = sqrt(value);
            } else {
                double diagonal = lower_values[column * size + column];
                double factor;

                if (diagonal == 0.0) {
                    status = NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE;
                    goto cleanup;
                }
                factor = value / diagonal;
                if (!isfinite(factor)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
                lower_values[row * size + column] = factor;
            }
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, size, lower_values, lower
    );

cleanup:
    if (lower_values != NULL) matrix_cholesky_free(lower_values);
    if (work != NULL) matrix_cholesky_free(work);
    return status;
}
