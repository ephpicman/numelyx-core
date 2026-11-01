/**
 * @file numerus_matrix_solve.c
 * @brief Linear-system solving and algorithm selection for Matrix operands.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_internal.h"
#include "numerus_matrix_lu.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

numerus_matrix_status numerus_matrix_solve(
    const numerus_matrix *matrix,
    const numerus_matrix *right_hand_side,
    numerus_matrix **solution
)
{
    numerus_matrix_lu_factorization *factorization = NULL;
    size_t size;
    size_t right_hand_sides;
    size_t element_count;
    size_t value_bytes;
    size_t work_bytes;
    size_t column;
    size_t rank;
    double *values = NULL;
    double *work = NULL;
    numerus_matrix_status status;

    if (solution == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *solution = NULL;

    if (matrix == NULL || right_hand_side == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);
    if (numerus_matrix_rows(right_hand_side) != size) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }
    right_hand_sides = numerus_matrix_columns(right_hand_side);

    status = numerus_matrix_rank(matrix, &rank);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }
    if (rank != size) {
        return NUMERUS_MATRIX_SINGULAR;
    }

    if (!numerus_size_multiply(size, right_hand_sides, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*values), &value_bytes) ||
        !numerus_size_multiply(size, sizeof(*work), &work_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = malloc(value_bytes);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    work = malloc(work_bytes);
    if (work == NULL) {
        free(values);
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    status = numerus_matrix_get_or_factorize_lu(matrix, &factorization);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        free(work);
        free(values);
        return status;
    }
    if (numerus_matrix_lu_rank(factorization) != size) {
        numerus_matrix_lu_destroy(factorization);
        free(work);
        free(values);
        return NUMERUS_MATRIX_SINGULAR;
    }

    for (column = 0; column < right_hand_sides; column++) {
        size_t row;

        /* Forward substitution: L y = P b. */
        for (row = 0; row < size; row++) {
            size_t source_row;
            size_t k;
            double value;

            status = numerus_matrix_lu_get_permutation(
                factorization, row, &source_row
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            status = numerus_matrix_get(
                right_hand_side, source_row, column, &value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto failure;
            }

            for (k = 0; k < row; k++) {
                double lower_value;
                status = numerus_matrix_lu_get_lower(
                    factorization, row, k, &lower_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
                value -= lower_value * work[k];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto failure;
                }
            }

            work[row] = value;
        }

        /* Back substitution: U x = y. */
        row = size;
        while (row > 0) {
            size_t k;
            double value;
            double diagonal;

            row--;
            value = work[row];

            for (k = row + 1; k < size; k++) {
                double upper_value;
                status = numerus_matrix_lu_get_upper(
                    factorization, row, k, &upper_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
                value -= upper_value * work[k];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto failure;
                }
            }

            status = numerus_matrix_lu_get_upper(
                factorization, row, row, &diagonal
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            if (diagonal == 0.0) {
                status = NUMERUS_MATRIX_SINGULAR;
                goto failure;
            }

            value /= diagonal;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto failure;
            }
            work[row] = value;
            values[row * right_hand_sides + column] = value;
        }
    }

    numerus_matrix_lu_destroy(factorization);
    free(work);
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, right_hand_sides, values, solution
    );
    free(values);
    return status;

failure:
    numerus_matrix_lu_destroy(factorization);
    free(work);
    free(values);
    return status;
}
