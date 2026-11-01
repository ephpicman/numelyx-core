/**
 * @file numerus_matrix_elementwise.c
 * @brief Elementwise arithmetic and scalar broadcasting for Matrix values.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"
#include <math.h>
#include <stdlib.h>
#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define elementwise_alloc(size) emalloc(size)
# define elementwise_free(pointer) efree(pointer)
#else
# define elementwise_alloc(size) malloc(size)
# define elementwise_free(pointer) free(pointer)
#endif

typedef enum { ELEMENTWISE_MIN, ELEMENTWISE_MAX } elementwise_operation;

static int create_extreme(const numerus_matrix *left, const numerus_matrix *right,
    numerus_matrix **matrix, elementwise_operation operation)
{
    size_t rows, columns, count, bytes, row, column;
    double *values;
    numerus_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (left == NULL || right == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numerus_matrix_rows(left);
    columns = numerus_matrix_columns(left);
    if (rows != numerus_matrix_rows(right) ||
        columns != numerus_matrix_columns(right)) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }
    if (!numerus_size_multiply(rows, columns, &count) ||
        !numerus_size_multiply(count, sizeof(*values), &bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }
    values = elementwise_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double a, b;
            status = numerus_matrix_get(left, row, column, &a);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                elementwise_free(values); return status;
            }
            status = numerus_matrix_get(right, row, column, &b);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                elementwise_free(values); return status;
            }
            if (isnan(a) || isnan(b)) values[row * columns + column] = NAN;
            else if (operation == ELEMENTWISE_MIN)
                values[row * columns + column] = a < b ? a : b;
            else values[row * columns + column] = a > b ? a : b;
        }
    }
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, columns, values, matrix);
    elementwise_free(values);
    return status;
}

int numerus_matrix_create_elementwise_min(const numerus_matrix *left,
    const numerus_matrix *right, numerus_matrix **matrix)
{
    return create_extreme(left, right, matrix, ELEMENTWISE_MIN);
}

int numerus_matrix_create_elementwise_max(const numerus_matrix *left,
    const numerus_matrix *right, numerus_matrix **matrix)
{
    return create_extreme(left, right, matrix, ELEMENTWISE_MAX);
}

int numerus_matrix_create_clamp(const numerus_matrix *source, double lower,
    double upper, numerus_matrix **matrix)
{
    size_t rows, columns, count, bytes, row, column;
    double *values;
    numerus_matrix_status status;
    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (source == NULL || isnan(lower) || isnan(upper) || lower > upper)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numerus_matrix_rows(source);
    columns = numerus_matrix_columns(source);
    if (!numerus_size_multiply(rows, columns, &count) ||
        !numerus_size_multiply(count, sizeof(*values), &bytes))
        return NUMERUS_MATRIX_OVERFLOW;
    values = elementwise_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;
            status = numerus_matrix_get(source, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                elementwise_free(values); return status;
            }
            if (isnan(value)) values[row * columns + column] = NAN;
            else if (value < lower) values[row * columns + column] = lower;
            else if (value > upper) values[row * columns + column] = upper;
            else values[row * columns + column] = value;
        }
    }
    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, columns, values, matrix);
    elementwise_free(values);
    return status;
}
