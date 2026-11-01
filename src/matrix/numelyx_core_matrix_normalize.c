/**
 * @file numelyx_core_matrix_normalize.c
 * @brief Matrix normalization and scale-aware transformations.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_size.h"
#include <math.h>
#include <stdlib.h>
# define normalize_alloc(size) malloc(size)
# define normalize_free(pointer) free(pointer)

static double vector_norm(const double *values, size_t rows, size_t columns,
    numelyx_core_matrix_normalize_axis axis, numelyx_core_matrix_normalize_norm norm,
    size_t vector)
{
    size_t length = axis == NUMERUS_MATRIX_NORMALIZE_WHOLE
        ? rows * columns
        : axis == NUMERUS_MATRIX_NORMALIZE_ROWS ? columns : rows;
    size_t i;
    double result = 0.0;
    for (i = 0; i < length; i++) {
        size_t index = axis == NUMERUS_MATRIX_NORMALIZE_WHOLE ? i
            : axis == NUMERUS_MATRIX_NORMALIZE_ROWS ? vector * columns + i
            : i * columns + vector;
        double magnitude = fabs(values[index]);
        if (norm == NUMERUS_MATRIX_NORMALIZE_L1) result += magnitude;
        else if (norm == NUMERUS_MATRIX_NORMALIZE_L2) result = hypot(result, magnitude);
        else if (magnitude > result) result = magnitude;
    }
    return result;
}

int numelyx_core_matrix_create_normalized(const numelyx_core_matrix *source,
    numelyx_core_matrix_normalize_axis axis, numelyx_core_matrix_normalize_norm norm,
    numelyx_core_matrix **matrix)
{
    size_t rows, columns, count, bytes, row, column, i, vectors;
    double *values;
    numelyx_core_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (source == NULL ||
        (axis != NUMERUS_MATRIX_NORMALIZE_WHOLE &&
         axis != NUMERUS_MATRIX_NORMALIZE_ROWS &&
         axis != NUMERUS_MATRIX_NORMALIZE_COLUMNS) ||
        (norm != NUMERUS_MATRIX_NORMALIZE_L1 &&
         norm != NUMERUS_MATRIX_NORMALIZE_L2 &&
         norm != NUMERUS_MATRIX_NORMALIZE_INFINITY))
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numelyx_core_matrix_rows(source);
    columns = numelyx_core_matrix_columns(source);
    if (!numelyx_core_size_multiply(rows, columns, &count) ||
        !numelyx_core_size_multiply(count, sizeof(*values), &bytes))
        return NUMERUS_MATRIX_OVERFLOW;
    values = normalize_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            status = numelyx_core_matrix_get(source, row, column,
                &values[row * columns + column]);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                normalize_free(values); return status;
            }
            if (!isfinite(values[row * columns + column])) {
                normalize_free(values); return NUMERUS_MATRIX_NON_FINITE;
            }
        }
    }
    vectors = axis == NUMERUS_MATRIX_NORMALIZE_WHOLE ? 1
        : axis == NUMERUS_MATRIX_NORMALIZE_ROWS ? rows : columns;
    for (i = 0; i < vectors; i++) {
        double length = vector_norm(values, rows, columns, axis, norm, i);
        size_t j;
        size_t vector_length = axis == NUMERUS_MATRIX_NORMALIZE_WHOLE ? count
            : axis == NUMERUS_MATRIX_NORMALIZE_ROWS ? columns : rows;
        if (!isfinite(length)) {
            normalize_free(values); return NUMERUS_MATRIX_NON_FINITE;
        }
        if (length == 0.0) {
            normalize_free(values); return NUMERUS_MATRIX_DIVISION_BY_ZERO;
        }
        for (j = 0; j < vector_length; j++) {
            size_t index = axis == NUMERUS_MATRIX_NORMALIZE_WHOLE ? j
                : axis == NUMERUS_MATRIX_NORMALIZE_ROWS ? i * columns + j
                : j * columns + i;
            values[index] /= length;
        }
    }
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, columns, values, matrix);
    normalize_free(values);
    return status;
}
