/**
 * @file numerus_matrix_system.c
 * @brief Matrix system-level numerical routines and shared solve orchestration.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_qr.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_classify_alloc(size) emalloc(size)
# define matrix_classify_free(pointer) efree(pointer)
#else
# define matrix_classify_alloc(size) malloc(size)
# define matrix_classify_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_classify_system(
    const numerus_matrix *matrix,
    const numerus_matrix *right_hand_side,
    numerus_matrix_solution_kind *classification
)
{
    numerus_matrix *q = NULL;
    numerus_matrix *r = NULL;
    size_t rows;
    size_t columns;
    size_t rhs_columns;
    size_t rank = 0;
    size_t maximum_dimension;
    size_t permutation_bytes;
    size_t vector_bytes;
    size_t projection_bytes = 0;
    size_t column;
    size_t row;
    size_t *permutation = NULL;
    double *rhs = NULL;
    double *projection = NULL;
    numerus_matrix_solution_kind result_kind;
    numerus_matrix_status status;

    if (matrix == NULL || right_hand_side == NULL || classification == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    rhs_columns = numerus_matrix_columns(right_hand_side);
    if (numerus_matrix_rows(right_hand_side) != rows) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    maximum_dimension = rows > columns ? rows : columns;
    if (!numerus_size_multiply(
        columns, sizeof(*permutation), &permutation_bytes
    ) || !numerus_size_multiply(
        rows, sizeof(*rhs), &vector_bytes
    )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    permutation = matrix_classify_alloc(permutation_bytes);
    rhs = matrix_classify_alloc(vector_bytes);
    if (permutation == NULL || rhs == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = numerus_matrix_qr_decompose_pivoted(
        matrix, &q, &r, permutation, columns, &rank
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    if (rank > 0) {
        if (!numerus_size_multiply(rank, sizeof(*projection), &projection_bytes)) {
            status = NUMERUS_MATRIX_OVERFLOW;
            goto cleanup;
        }
        projection = matrix_classify_alloc(projection_bytes);
        if (projection == NULL) {
            status = NUMERUS_MATRIX_OUT_OF_MEMORY;
            goto cleanup;
        }
    }

    result_kind = rank == columns
        ? NUMERUS_MATRIX_SOLUTION_UNIQUE
        : NUMERUS_MATRIX_SOLUTION_INFINITE;

    for (column = 0; column < rhs_columns; column++) {
        double rhs_norm = 0.0;
        double residual_norm = 0.0;
        double tolerance;

        for (row = 0; row < rows; row++) {
            status = numerus_matrix_get(
                right_hand_side, row, column, &rhs[row]
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (!isfinite(rhs[row])) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            rhs_norm = hypot(rhs_norm, rhs[row]);
            if (!isfinite(rhs_norm)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }

        for (size_t basis_column = 0; basis_column < rank; basis_column++) {
            double dot = 0.0;

            for (row = 0; row < rows; row++) {
                double q_value;
                status = numerus_matrix_get(q, row, basis_column, &q_value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                dot += q_value * rhs[row];
                if (!isfinite(dot)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
            projection[basis_column] = dot;
        }

        for (row = 0; row < rows; row++) {
            double value = rhs[row];
            size_t basis_column;

            for (basis_column = 0; basis_column < rank; basis_column++) {
                double q_value;
                status = numerus_matrix_get(q, row, basis_column, &q_value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                value -= q_value * projection[basis_column];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
            residual_norm = hypot(residual_norm, value);
            if (!isfinite(residual_norm)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }

        tolerance = NUMERUS_EPSILON * (double) maximum_dimension * rhs_norm;
        if (residual_norm > tolerance) {
            result_kind = NUMERUS_MATRIX_SOLUTION_INCONSISTENT;
        }
    }

    *classification = result_kind;
    status = NUMERUS_MATRIX_SUCCESS;

cleanup:
    if (projection != NULL) matrix_classify_free(projection);
    if (rhs != NULL) matrix_classify_free(rhs);
    if (permutation != NULL) matrix_classify_free(permutation);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    return status;
}
