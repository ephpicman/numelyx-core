/**
 * @file numerus_matrix_inverse.c
 * @brief Matrix inversion using the selected factorization strategy.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_internal.h"
#include "numerus_matrix_lu.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

numerus_matrix_status numerus_matrix_inverse(
    const numerus_matrix *matrix,
    numerus_matrix **inverse
)
{
    numerus_matrix_lu_factorization *factorization = NULL;
    size_t size;
    size_t rank;
    size_t element_count;
    size_t value_bytes;
    size_t work_bytes;
    size_t column;
    double *values = NULL;
    double *work = NULL;
    numerus_matrix_status status;

    if (inverse == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *inverse = NULL;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    status = numerus_matrix_get_cached_inverse(matrix, inverse);
    if (status != NUMERUS_MATRIX_SUCCESS || *inverse != NULL) {
        return status;
    }

    size = numerus_matrix_rows(matrix);
    status = numerus_matrix_rank(matrix, &rank);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }
    if (rank != size) {
        return NUMERUS_MATRIX_SINGULAR;
    }

    if (!numerus_size_multiply(size, size, &element_count) ||
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

    /* Solve A x = e_j for each basis vector, reusing one LU factorization. */
    for (column = 0; column < size; column++) {
        size_t row;

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
            value = source_row == column ? 1.0 : 0.0;

            for (k = 0; k < row; k++) {
                double lower_value;
                status = numerus_matrix_lu_get_lower(
                    factorization, row, k, &lower_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
                value -= lower_value * work[k];
            }

            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto failure;
            }
            work[row] = value;
        }

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
            values[row * size + column] = value;
        }
    }

    numerus_matrix_lu_destroy(factorization);
    free(work);
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, size, values, inverse
    );
    if (status == NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_store_inverse_cache(matrix, values);
    }
    free(values);
    return status;

failure:
    numerus_matrix_lu_destroy(factorization);
    free(work);
    free(values);
    return status;
}
