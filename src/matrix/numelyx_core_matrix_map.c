/**
 * @file numelyx_core_matrix_map.c
 * @brief Matrix mapping and callback-driven transformation operations.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_size.h"
#include <stdlib.h>
# define map_alloc(size) malloc(size)
# define map_free(pointer) free(pointer)

int numelyx_core_matrix_create_map(const numelyx_core_matrix *parent,
    numelyx_core_value_transform_fn transform, const void *context,
    numelyx_core_matrix **matrix)
{
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (parent == NULL || transform == NULL)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    return numelyx_core_matrix_create_from_parent_with_transforms(
        (numelyx_core_matrix *) parent, numelyx_core_matrix_rows(parent),
        numelyx_core_matrix_columns(parent), NULL, transform, context, matrix);
}

int numelyx_core_matrix_apply(const numelyx_core_matrix *source,
    numelyx_core_value_transform_fn transform, const void *context,
    numelyx_core_matrix **matrix)
{
    size_t rows, columns, count, bytes, row, column;
    double *values;
    numelyx_core_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (source == NULL || transform == NULL)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numelyx_core_matrix_rows(source);
    columns = numelyx_core_matrix_columns(source);
    if (!numelyx_core_size_multiply(rows, columns, &count) ||
        !numelyx_core_size_multiply(count, sizeof(*values), &bytes))
        return NUMERUS_MATRIX_OVERFLOW;
    values = map_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double input, output;
            status = numelyx_core_matrix_get(source, row, column, &input);
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
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, columns, values, matrix);
    map_free(values);
    return status;
}
