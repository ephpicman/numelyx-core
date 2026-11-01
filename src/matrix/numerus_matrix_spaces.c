/**
 * @file numerus_matrix_spaces.c
 * @brief Matrix range, row-space, and column-space operations.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_qr.h"
#include "numerus_size.h"

#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_space_alloc(size) emalloc(size)
# define matrix_space_free(pointer) efree(pointer)
#else
# define matrix_space_alloc(size) malloc(size)
# define matrix_space_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_column_space_basis(
    const numerus_matrix *matrix,
    numerus_matrix **basis,
    size_t *dimension
)
{
    numerus_matrix *q = NULL;
    numerus_matrix *r = NULL;
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
    numerus_matrix_status status;

    if (basis != NULL) {
        *basis = NULL;
    }
    if (matrix == NULL || basis == NULL || dimension == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    if (!numerus_size_multiply(
        columns, sizeof(*permutation), &permutation_bytes
    )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    permutation = matrix_space_alloc(permutation_bytes);
    if (permutation == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = numerus_matrix_qr_decompose_pivoted(
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

    if (!numerus_size_multiply(rows, rank, &value_count) ||
        !numerus_size_multiply(value_count, sizeof(*values), &value_bytes)) {
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
            status = numerus_matrix_get(
                matrix, row, permutation[column],
                &values[row * rank + column]
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, rank, values, basis
    );
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *dimension = rank;
    }

cleanup:
    if (values != NULL) matrix_space_free(values);
    if (permutation != NULL) matrix_space_free(permutation);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    return status;
}

numerus_matrix_status numerus_matrix_row_space_basis(
    const numerus_matrix *matrix,
    numerus_matrix **basis,
    size_t *dimension
)
{
    numerus_matrix *transpose = NULL;
    numerus_matrix *q = NULL;
    numerus_matrix *r = NULL;
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
    numerus_matrix_status status;

    if (basis != NULL) {
        *basis = NULL;
    }
    if (matrix == NULL || basis == NULL || dimension == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    if (!numerus_size_multiply(
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
    status = (numerus_matrix_status) numerus_matrix_create_transpose(
        (numerus_matrix *) matrix, &transpose
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = numerus_matrix_qr_decompose_pivoted(
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

    if (!numerus_size_multiply(rank, columns, &value_count) ||
        !numerus_size_multiply(value_count, sizeof(*values), &value_bytes)) {
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
            status = numerus_matrix_get(
                matrix, permutation[row], column,
                &values[row * columns + column]
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rank, columns, values, basis
    );
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *dimension = rank;
    }

cleanup:
    if (values != NULL) matrix_space_free(values);
    if (permutation != NULL) matrix_space_free(permutation);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    numerus_matrix_destroy(transpose);
    return status;
}
