/**
 * @file numerus_matrix_solve_triangular.c
 * @brief Forward and backward substitution for triangular systems.
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
# define matrix_triangular_alloc(size) emalloc(size)
# define matrix_triangular_free(pointer) efree(pointer)
#else
# define matrix_triangular_alloc(size) malloc(size)
# define matrix_triangular_free(pointer) free(pointer)
#endif

static numerus_matrix_status read_triangular_coefficient(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    bool transpose,
    double *value
)
{
    if (transpose) {
        return numerus_matrix_get(matrix, column, row, value);
    }

    return numerus_matrix_get(matrix, row, column, value);
}

numerus_matrix_status numerus_matrix_solve_triangular(
    const numerus_matrix *triangular,
    const numerus_matrix *right_hand_side,
    bool lower,
    bool transpose,
    numerus_matrix **solution
)
{
    size_t size;
    size_t rhs_columns;
    size_t element_count;
    size_t value_bytes;
    size_t work_bytes;
    size_t row;
    size_t column;
    double *values = NULL;
    double *work = NULL;
    bool effective_lower;
    numerus_matrix_status status;

    if (solution == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *solution = NULL;

    if (triangular == NULL || right_hand_side == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (numerus_matrix_rows(triangular) !=
        numerus_matrix_columns(triangular)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(triangular);
    if (numerus_matrix_rows(right_hand_side) != size) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }
    rhs_columns = numerus_matrix_columns(right_hand_side);

    if (!numerus_size_multiply(size, rhs_columns, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*values), &value_bytes) ||
        !numerus_size_multiply(size, sizeof(*work), &work_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = matrix_triangular_alloc(value_bytes);
    work = matrix_triangular_alloc(work_bytes);
    if (values == NULL || work == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    /*
     * Validate the stored orientation before considering the optional
     * transpose. The unused half must contain exact zeros; silently ignoring
     * small non-zero values would solve a different system than the caller
     * supplied.
     */
    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;

            status = numerus_matrix_get(
                triangular, row, column, &value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            if (row == column && value == 0.0) {
                status = NUMERUS_MATRIX_SINGULAR;
                goto cleanup;
            }
            if ((lower && row < column && value != 0.0) ||
                (!lower && row > column && value != 0.0)) {
                status = NUMERUS_MATRIX_INVALID_ARGUMENT;
                goto cleanup;
            }
        }
    }

    effective_lower = lower != transpose;

    for (column = 0; column < rhs_columns; column++) {
        if (effective_lower) {
            for (row = 0; row < size; row++) {
                size_t k;
                double value;
                double diagonal;

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

                for (k = 0; k < row; k++) {
                    double coefficient;

                    status = read_triangular_coefficient(
                        triangular, row, k, transpose, &coefficient
                    );
                    if (status != NUMERUS_MATRIX_SUCCESS) {
                        goto cleanup;
                    }
                    value -= coefficient * work[k];
                    if (!isfinite(value)) {
                        status = NUMERUS_MATRIX_NON_FINITE;
                        goto cleanup;
                    }
                }

                status = numerus_matrix_get(
                    triangular, row, row, &diagonal
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                value /= diagonal;
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }

                work[row] = value;
                values[row * rhs_columns + column] = value;
            }
        } else {
            row = size;
            while (row > 0) {
                size_t k;
                double value;
                double diagonal;

                row--;
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

                for (k = row + 1; k < size; k++) {
                    double coefficient;

                    status = read_triangular_coefficient(
                        triangular, row, k, transpose, &coefficient
                    );
                    if (status != NUMERUS_MATRIX_SUCCESS) {
                        goto cleanup;
                    }
                    value -= coefficient * work[k];
                    if (!isfinite(value)) {
                        status = NUMERUS_MATRIX_NON_FINITE;
                        goto cleanup;
                    }
                }

                status = numerus_matrix_get(
                    triangular, row, row, &diagonal
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                value /= diagonal;
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }

                work[row] = value;
                values[row * rhs_columns + column] = value;
            }
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, rhs_columns, values, solution
    );

cleanup:
    if (work != NULL) {
        matrix_triangular_free(work);
    }
    if (values != NULL) {
        matrix_triangular_free(values);
    }
    return status;
}
