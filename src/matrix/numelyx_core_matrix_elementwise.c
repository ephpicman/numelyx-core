/**
 * @file numelyx_core_matrix_elementwise.c
 * @brief Elementwise arithmetic and scalar broadcasting for Matrix values.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_size.h"
#include <math.h>
#include <stdlib.h>
# define elementwise_alloc(size) malloc(size)
# define elementwise_free(pointer) free(pointer)

typedef enum { ELEMENTWISE_MIN, ELEMENTWISE_MAX } elementwise_operation;

static int create_extreme(const numelyx_core_matrix *left, const numelyx_core_matrix *right,
    numelyx_core_matrix **matrix, elementwise_operation operation)
{
    size_t rows, columns, count, bytes, row, column;
    double *values;
    numelyx_core_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (left == NULL || right == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numelyx_core_matrix_rows(left);
    columns = numelyx_core_matrix_columns(left);
    if (rows != numelyx_core_matrix_rows(right) ||
        columns != numelyx_core_matrix_columns(right)) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }
    if (!numelyx_core_size_multiply(rows, columns, &count) ||
        !numelyx_core_size_multiply(count, sizeof(*values), &bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }
    values = elementwise_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double a, b;
            status = numelyx_core_matrix_get(left, row, column, &a);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                elementwise_free(values); return status;
            }
            status = numelyx_core_matrix_get(right, row, column, &b);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                elementwise_free(values); return status;
            }
            if (isnan(a) || isnan(b)) values[row * columns + column] = NAN;
            else if (operation == ELEMENTWISE_MIN)
                values[row * columns + column] = a < b ? a : b;
            else values[row * columns + column] = a > b ? a : b;
        }
    }
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, columns, values, matrix);
    elementwise_free(values);
    return status;
}

int numelyx_core_matrix_create_elementwise_min(const numelyx_core_matrix *left,
    const numelyx_core_matrix *right, numelyx_core_matrix **matrix)
{
    return create_extreme(left, right, matrix, ELEMENTWISE_MIN);
}

int numelyx_core_matrix_create_elementwise_max(const numelyx_core_matrix *left,
    const numelyx_core_matrix *right, numelyx_core_matrix **matrix)
{
    return create_extreme(left, right, matrix, ELEMENTWISE_MAX);
}

int numelyx_core_matrix_create_clamp(const numelyx_core_matrix *source, double lower,
    double upper, numelyx_core_matrix **matrix)
{
    size_t rows, columns, count, bytes, row, column;
    double *values;
    numelyx_core_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (source == NULL || isnan(lower) || isnan(upper) || lower > upper)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numelyx_core_matrix_rows(source);
    columns = numelyx_core_matrix_columns(source);
    if (!numelyx_core_size_multiply(rows, columns, &count) ||
        !numelyx_core_size_multiply(count, sizeof(*values), &bytes))
        return NUMERUS_MATRIX_OVERFLOW;
    values = elementwise_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;
            status = numelyx_core_matrix_get(source, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                elementwise_free(values); return status;
            }
            if (isnan(value)) values[row * columns + column] = NAN;
            else if (value < lower) values[row * columns + column] = lower;
            else if (value > upper) values[row * columns + column] = upper;
            else values[row * columns + column] = value;
        }
    }
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, columns, values, matrix);
    elementwise_free(values);
    return status;
}
