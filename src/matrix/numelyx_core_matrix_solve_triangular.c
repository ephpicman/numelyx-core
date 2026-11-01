/**
 * @file numelyx_core_matrix_solve_triangular.c
 * @brief Forward and backward substitution for triangular systems.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_size.h"

#include <math.h>
#include <limits.h>
#include <stdlib.h>

#ifdef _WIN32
# include <mkl.h>
#else
# include <lapacke.h>
#endif

# define matrix_triangular_alloc(size) malloc(size)
# define matrix_triangular_free(pointer) free(pointer)

static numelyx_core_matrix_status read_triangular_coefficient(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    bool transpose,
    double *value
)
{
    if (transpose) {
        return numelyx_core_matrix_get(matrix, column, row, value);
    }

    return numelyx_core_matrix_get(matrix, row, column, value);
}

numelyx_core_matrix_status numelyx_core_matrix_solve_triangular(
    const numelyx_core_matrix *triangular,
    const numelyx_core_matrix *right_hand_side,
    bool lower,
    bool transpose,
    numelyx_core_matrix **solution
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
    double *triangular_values = NULL;
    bool effective_lower;
    numelyx_core_matrix_status status;

    if (solution == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *solution = NULL;

    if (triangular == NULL || right_hand_side == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (numelyx_core_matrix_rows(triangular) !=
        numelyx_core_matrix_columns(triangular)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numelyx_core_matrix_rows(triangular);
    if (numelyx_core_matrix_rows(right_hand_side) != size) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }
    rhs_columns = numelyx_core_matrix_columns(right_hand_side);

    if (!numelyx_core_size_multiply(size, rhs_columns, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(*values), &value_bytes) ||
        !numelyx_core_size_multiply(size, sizeof(*work), &work_bytes)) {
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

            status = numelyx_core_matrix_get(
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

    if (size <= (size_t) INT_MAX && rhs_columns <= (size_t) INT_MAX) {
        size_t matrix_count;
        size_t matrix_bytes;
        lapack_int info;

        if (!numelyx_core_size_multiply(size, size, &matrix_count) ||
            !numelyx_core_size_multiply(
                matrix_count, sizeof(*triangular_values), &matrix_bytes
            )) {
            status = NUMERUS_MATRIX_OVERFLOW;
            goto cleanup;
        }

        triangular_values = matrix_triangular_alloc(matrix_bytes);
        if (triangular_values == NULL) {
            status = NUMERUS_MATRIX_OUT_OF_MEMORY;
            goto cleanup;
        }

        for (row = 0; row < size; row++) {
            for (column = 0; column < size; column++) {
                status = numelyx_core_matrix_get(
                    triangular, row, column,
                    &triangular_values[row * size + column]
                );
                if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
            }
        }
        for (row = 0; row < size; row++) {
            for (column = 0; column < rhs_columns; column++) {
                double value;
                status = numelyx_core_matrix_get(
                    right_hand_side, row, column, &value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
                values[row * rhs_columns + column] = value;
            }
        }

        info = LAPACKE_dtrtrs(
            LAPACK_ROW_MAJOR,
            lower ? 'L' : 'U',
            transpose ? 'T' : 'N',
            'N',
            (lapack_int) size,
            (lapack_int) rhs_columns,
            triangular_values,
            (lapack_int) size,
            values,
            (lapack_int) rhs_columns
        );
        if (info == LAPACK_WORK_MEMORY_ERROR ||
            info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
            status = NUMERUS_MATRIX_OUT_OF_MEMORY;
            goto cleanup;
        }
        if (info > 0) {
            status = NUMERUS_MATRIX_SINGULAR;
            goto cleanup;
        }
        if (info < 0) {
            status = NUMERUS_MATRIX_INVALID_ARGUMENT;
            goto cleanup;
        }
        for (row = 0; row < element_count; row++) {
            if (!isfinite(values[row])) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }

        status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
            size, rhs_columns, values, solution
        );
        goto cleanup;
    }

    for (column = 0; column < rhs_columns; column++) {
        if (effective_lower) {
            for (row = 0; row < size; row++) {
                size_t k;
                double value;
                double diagonal;

                status = numelyx_core_matrix_get(
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

                status = numelyx_core_matrix_get(
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
                status = numelyx_core_matrix_get(
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

                status = numelyx_core_matrix_get(
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

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        size, rhs_columns, values, solution
    );

cleanup:
    if (triangular_values != NULL) {
        matrix_triangular_free(triangular_values);
    }
    if (work != NULL) {
        matrix_triangular_free(work);
    }
    if (values != NULL) {
        matrix_triangular_free(values);
    }
    return status;
}
