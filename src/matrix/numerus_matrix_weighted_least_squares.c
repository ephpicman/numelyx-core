/**
 * @file numerus_matrix_weighted_least_squares.c
 * @brief Weighted least-squares systems and weight validation.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_weighted_alloc(size) emalloc(size)
# define matrix_weighted_free(pointer) efree(pointer)
#else
# define matrix_weighted_alloc(size) malloc(size)
# define matrix_weighted_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_weighted_least_squares(
    const numerus_matrix *matrix,
    const numerus_matrix *right_hand_side,
    const double *weights,
    size_t weights_count,
    numerus_matrix **solution
)
{
    numerus_matrix *weighted_matrix = NULL;
    numerus_matrix *weighted_rhs = NULL;
    size_t rows;
    size_t columns;
    size_t rhs_columns;
    size_t matrix_count;
    size_t matrix_bytes;
    size_t rhs_count;
    size_t rhs_bytes;
    size_t row;
    size_t column;
    double *matrix_values = NULL;
    double *rhs_values = NULL;
    numerus_matrix_status status;

    if (solution == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *solution = NULL;
    if (matrix == NULL || right_hand_side == NULL || weights == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    rhs_columns = numerus_matrix_columns(right_hand_side);
    if (numerus_matrix_rows(right_hand_side) != rows ||
        weights_count != rows) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    if (!numerus_size_multiply(rows, columns, &matrix_count) ||
        !numerus_size_multiply(matrix_count, sizeof(*matrix_values), &matrix_bytes) ||
        !numerus_size_multiply(rows, rhs_columns, &rhs_count) ||
        !numerus_size_multiply(rhs_count, sizeof(*rhs_values), &rhs_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    matrix_values = matrix_weighted_alloc(matrix_bytes);
    rhs_values = matrix_weighted_alloc(rhs_bytes);
    if (matrix_values == NULL || rhs_values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < rows; row++) {
        double weight = weights[row];
        double row_scale;

        if (!isfinite(weight)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
        if (weight < 0.0) {
            status = NUMERUS_MATRIX_INVALID_ARGUMENT;
            goto cleanup;
        }
        row_scale = sqrt(weight);
        if (!isfinite(row_scale)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }

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
            value *= row_scale;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            matrix_values[row * columns + column] = value;
        }

        for (column = 0; column < rhs_columns; column++) {
            double value;
            status = numerus_matrix_get(
                right_hand_side, row, column, &value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            value *= row_scale;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            rhs_values[row * rhs_columns + column] = value;
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, columns, matrix_values, &weighted_matrix
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, rhs_columns, rhs_values, &weighted_rhs
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    status = numerus_matrix_least_squares(
        weighted_matrix, weighted_rhs, solution
    );

cleanup:
    if (rhs_values != NULL) matrix_weighted_free(rhs_values);
    if (matrix_values != NULL) matrix_weighted_free(matrix_values);
    numerus_matrix_destroy(weighted_rhs);
    numerus_matrix_destroy(weighted_matrix);
    return status;
}
