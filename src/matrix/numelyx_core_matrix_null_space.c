/**
 * @file numelyx_core_matrix_null_space.c
 * @brief Null-space computation and rank-aware basis construction.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_matrix_qr.h"
#include "numelyx_core_size.h"

#include <math.h>
#include <stdlib.h>

# define matrix_null_alloc(size) malloc(size)
# define matrix_null_free(pointer) free(pointer)

numelyx_core_matrix_status numelyx_core_matrix_null_space(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **basis,
    size_t *nullity
)
{
    numelyx_core_matrix *q = NULL;
    numelyx_core_matrix *r = NULL;
    size_t columns;
    size_t rank = 0;
    size_t nullity_result;
    size_t permutation_bytes;
    size_t vector_bytes;
    size_t basis_count;
    size_t basis_bytes;
    size_t free_column;
    size_t *permutation = NULL;
    double *work = NULL;
    double *values = NULL;
    numelyx_core_matrix_status status;

    if (basis != NULL) {
        *basis = NULL;
    }
    if (basis == NULL || nullity == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    columns = numelyx_core_matrix_columns(matrix);
    if (!numelyx_core_size_multiply(
        columns, sizeof(*permutation), &permutation_bytes
    ) || !numelyx_core_size_multiply(
        columns, sizeof(*work), &vector_bytes
    )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    permutation = matrix_null_alloc(permutation_bytes);
    work = matrix_null_alloc(vector_bytes);
    if (permutation == NULL || work == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = numelyx_core_matrix_qr_decompose_pivoted(
        matrix, &q, &r, permutation, columns, &rank
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    nullity_result = columns - rank;
    if (nullity_result == 0) {
        *nullity = 0;
        status = NUMERUS_MATRIX_SUCCESS;
        goto cleanup;
    }

    if (!numelyx_core_size_multiply(
        columns, nullity_result, &basis_count
    ) || !numelyx_core_size_multiply(
        basis_count, sizeof(*values), &basis_bytes
    )) {
        status = NUMERUS_MATRIX_OVERFLOW;
        goto cleanup;
    }

    values = matrix_null_alloc(basis_bytes);
    if (values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (free_column = rank; free_column < columns; free_column++) {
        size_t row;
        size_t basis_column = free_column - rank;

        for (row = 0; row < columns; row++) {
            work[row] = row == free_column ? 1.0 : 0.0;
        }

        /* Solve R11*y1 = -R12*y2 by back substitution. */
        row = rank;
        while (row > 0) {
            size_t column;
            double value;
            double diagonal;

            row--;
            status = numelyx_core_matrix_get(r, row, free_column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            value = -value;

            for (column = row + 1; column < rank; column++) {
                double upper_value;
                status = numelyx_core_matrix_get(r, row, column, &upper_value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                value -= upper_value * work[column];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }

            status = numelyx_core_matrix_get(r, row, row, &diagonal);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (diagonal == 0.0) {
                status = NUMERUS_MATRIX_SINGULAR;
                goto cleanup;
            }

            value /= diagonal;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row] = value;
        }

        /* Undo the column permutation: x[permutation[j]] = y[j]. */
        for (row = 0; row < columns; row++) {
            values[permutation[row] * nullity_result + basis_column] = work[row];
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        columns, nullity_result, values, basis
    );
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *nullity = nullity_result;
    }

cleanup:
    if (values != NULL) matrix_null_free(values);
    if (work != NULL) matrix_null_free(work);
    if (permutation != NULL) matrix_null_free(permutation);
    numelyx_core_matrix_destroy(r);
    numelyx_core_matrix_destroy(q);
    return status;
}
