/**
 * @file numelyx_core_matrix_spaces.c
 * @brief Matrix range, row-space, and column-space operations.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_matrix_qr.h"
#include "numelyx_core_size.h"

#include <stdlib.h>

# define matrix_space_alloc(size) malloc(size)
# define matrix_space_free(pointer) free(pointer)

numelyx_core_matrix_status numelyx_core_matrix_column_space_basis(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **basis,
    size_t *dimension
)
{
    numelyx_core_matrix *q = NULL;
    numelyx_core_matrix *r = NULL;
    size_t rows;
    size_t columns;
    size_t rank = 0;
    size_t permutation_bytes;
    size_t value_count;
    size_t value_bytes;
    size_t row;
    size_t column;
    size_t *permutation = NULL;
    double *values = NULL;
    numelyx_core_matrix_status status;

    if (basis != NULL) {
        *basis = NULL;
    }
    if (matrix == NULL || basis == NULL || dimension == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numelyx_core_matrix_rows(matrix);
    columns = numelyx_core_matrix_columns(matrix);
    if (!numelyx_core_size_multiply(
        columns, sizeof(*permutation), &permutation_bytes
    )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    permutation = matrix_space_alloc(permutation_bytes);
    if (permutation == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = numelyx_core_matrix_qr_decompose_pivoted(
        matrix, &q, &r, permutation, columns, &rank
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    if (rank == 0) {
        *dimension = 0;
        status = NUMERUS_MATRIX_SUCCESS;
        goto cleanup;
    }

    if (!numelyx_core_size_multiply(rows, rank, &value_count) ||
        !numelyx_core_size_multiply(value_count, sizeof(*values), &value_bytes)) {
        status = NUMERUS_MATRIX_OVERFLOW;
        goto cleanup;
    }
    values = matrix_space_alloc(value_bytes);
    if (values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < rank; column++) {
            status = numelyx_core_matrix_get(
                matrix, row, permutation[column],
                &values[row * rank + column]
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, rank, values, basis
    );
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *dimension = rank;
    }

cleanup:
    if (values != NULL) matrix_space_free(values);
    if (permutation != NULL) matrix_space_free(permutation);
    numelyx_core_matrix_destroy(r);
    numelyx_core_matrix_destroy(q);
    return status;
}

numelyx_core_matrix_status numelyx_core_matrix_row_space_basis(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **basis,
    size_t *dimension
)
{
    numelyx_core_matrix *transpose = NULL;
    numelyx_core_matrix *q = NULL;
    numelyx_core_matrix *r = NULL;
    size_t rows;
    size_t columns;
    size_t rank = 0;
    size_t permutation_bytes;
    size_t value_count;
    size_t value_bytes;
    size_t row;
    size_t column;
    size_t *permutation = NULL;
    double *values = NULL;
    numelyx_core_matrix_status status;

    if (basis != NULL) {
        *basis = NULL;
    }
    if (matrix == NULL || basis == NULL || dimension == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numelyx_core_matrix_rows(matrix);
    columns = numelyx_core_matrix_columns(matrix);
    if (!numelyx_core_size_multiply(
        rows, sizeof(*permutation), &permutation_bytes
    )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    permutation = matrix_space_alloc(permutation_bytes);
    if (permutation == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    /*
     * View constructors currently accept a mutable pointer despite never
     * mutating their immutable parent. The cast only adapts that legacy
     * signature; this function performs no writes through the parent.
     */
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_transpose(
        (numelyx_core_matrix *) matrix, &transpose
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = numelyx_core_matrix_qr_decompose_pivoted(
        transpose, &q, &r, permutation, rows, &rank
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    if (rank == 0) {
        *dimension = 0;
        status = NUMERUS_MATRIX_SUCCESS;
        goto cleanup;
    }

    if (!numelyx_core_size_multiply(rank, columns, &value_count) ||
        !numelyx_core_size_multiply(value_count, sizeof(*values), &value_bytes)) {
        status = NUMERUS_MATRIX_OVERFLOW;
        goto cleanup;
    }
    values = matrix_space_alloc(value_bytes);
    if (values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < rank; row++) {
        for (column = 0; column < columns; column++) {
            status = numelyx_core_matrix_get(
                matrix, permutation[row], column,
                &values[row * columns + column]
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rank, columns, values, basis
    );
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *dimension = rank;
    }

cleanup:
    if (values != NULL) matrix_space_free(values);
    if (permutation != NULL) matrix_space_free(permutation);
    numelyx_core_matrix_destroy(r);
    numelyx_core_matrix_destroy(q);
    numelyx_core_matrix_destroy(transpose);
    return status;
}
