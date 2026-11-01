/**
 * @file numerus_matrix_least_squares.c
 * @brief Least-squares solvers and residual-oriented result construction.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_qr.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_least_squares_alloc(size) emalloc(size)
# define matrix_least_squares_free(pointer) efree(pointer)
#else
# define matrix_least_squares_alloc(size) malloc(size)
# define matrix_least_squares_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_least_squares(
    const numerus_matrix *matrix,
    const numerus_matrix *right_hand_side,
    numerus_matrix **solution
)
{
    numerus_matrix *q = NULL;
    numerus_matrix *r = NULL;
    size_t rows;
    size_t columns;
    size_t rhs_columns;
    size_t rank = 0;
    size_t permutation_bytes;
    size_t projection_bytes;
    size_t result_count;
    size_t result_bytes;
    size_t row;
    size_t column;
    size_t *permutation = NULL;
    double *projection = NULL;
    double *values = NULL;
    numerus_matrix_status status;

    if (solution == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *solution = NULL;
    if (matrix == NULL || right_hand_side == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    rhs_columns = numerus_matrix_columns(right_hand_side);
    if (numerus_matrix_rows(right_hand_side) != rows) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    if (!numerus_size_multiply(
        columns, sizeof(*permutation), &permutation_bytes
    ) || !numerus_size_multiply(
        columns, sizeof(*projection), &projection_bytes
    ) || !numerus_size_multiply(
        columns, rhs_columns, &result_count
    ) || !numerus_size_multiply(
        result_count, sizeof(*values), &result_bytes
    )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    permutation = matrix_least_squares_alloc(permutation_bytes);
    projection = matrix_least_squares_alloc(projection_bytes);
    values = matrix_least_squares_alloc(result_bytes);
    if (permutation == NULL || projection == NULL || values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    /* Pivoted QR provides both Q/R and the column permutation. This solver
     * deliberately requires full column rank; rank-deficient and underdetermined
     * minimum-norm semantics belong to a different solver contract.
     */
    status = numerus_matrix_qr_decompose_pivoted(
        matrix, &q, &r, permutation, columns, &rank
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    if (rank != columns) {
        status = NUMERUS_MATRIX_RANK_DEFICIENT;
        goto cleanup;
    }

    /* Solve each right-hand side independently: first project b onto Q using
     * Qᵀb, then solve the upper-triangular system Rz = Qᵀb by back substitution.
     * The final vector is scattered through permutation because QR factored A P.
     */
    for (column = 0; column < rhs_columns; column++) {
        for (row = 0; row < columns; row++) {
            size_t source_row;
            double dot = 0.0;

            for (source_row = 0; source_row < rows; source_row++) {
                double q_value;
                double rhs_value;

                status = numerus_matrix_get(
                    q, source_row, row, &q_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                status = numerus_matrix_get(
                    right_hand_side, source_row, column, &rhs_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                if (!isfinite(rhs_value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }

                dot += q_value * rhs_value;
                if (!isfinite(dot)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
            projection[row] = dot;
        }

        row = columns;
        while (row > 0) {
            size_t upper_column;
            double value;
            double diagonal;

            row--;
            value = projection[row];
            for (upper_column = row + 1;
                 upper_column < columns;
                 upper_column++) {
                double upper_value;
                status = numerus_matrix_get(
                    r, row, upper_column, &upper_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                value -= upper_value * projection[upper_column];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }

            status = numerus_matrix_get(r, row, row, &diagonal);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (diagonal == 0.0) {
                status = NUMERUS_MATRIX_RANK_DEFICIENT;
                goto cleanup;
            }

            value /= diagonal;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            projection[row] = value;
            values[permutation[row] * rhs_columns + column] = value;
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        columns, rhs_columns, values, solution
    );

cleanup:
    if (values != NULL) matrix_least_squares_free(values);
    if (projection != NULL) matrix_least_squares_free(projection);
    if (permutation != NULL) matrix_least_squares_free(permutation);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    return status;
}
