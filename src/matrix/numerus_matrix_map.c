/**
 * @file numerus_matrix_map.c
 * @brief Matrix mapping and callback-driven transformation operations.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"
#include <stdlib.h>
#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define map_alloc(size) emalloc(size)
# define map_free(pointer) efree(pointer)
#else
# define map_alloc(size) malloc(size)
# define map_free(pointer) free(pointer)
#endif

int numerus_matrix_create_map(const numerus_matrix *parent,
    numerus_value_transform_fn transform, const void *context,
    numerus_matrix **matrix)
{
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (parent == NULL || transform == NULL)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    return numerus_matrix_create_from_parent_with_transforms(
        (numerus_matrix *) parent, numerus_matrix_rows(parent),
        numerus_matrix_columns(parent), NULL, transform, context, matrix);
}

int numerus_matrix_apply(const numerus_matrix *source,
    numerus_value_transform_fn transform, const void *context,
    numerus_matrix **matrix)
{
    size_t rows, columns, count, bytes, row, column;
    double *values;
    numerus_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (source == NULL || transform == NULL)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numerus_matrix_rows(source);
    columns = numerus_matrix_columns(source);
    if (!numerus_size_multiply(rows, columns, &count) ||
        !numerus_size_multiply(count, sizeof(*values), &bytes))
        return NUMERUS_MATRIX_OVERFLOW;
    values = map_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double input, output;
            status = numerus_matrix_get(source, row, column, &input);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                map_free(values); return status;
            }
            status = transform(row, column, input, &output, context);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                map_free(values); return status;
            }
            values[row * columns + column] = output;
        }
    }
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, columns, values, matrix);
    map_free(values);
    return status;
}
