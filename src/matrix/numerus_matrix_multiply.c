/**
 * @file numerus_matrix_multiply.c
 * @brief Matrix multiplication kernels and dimension validation.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_multiply_alloc(size) emalloc(size)
# define numerus_multiply_free(ptr) efree(ptr)
#else
# define numerus_multiply_alloc(size) malloc(size)
# define numerus_multiply_free(ptr) free(ptr)
#endif

static numerus_matrix_status matrix_matches_identity(
    const numerus_matrix *matrix,
    bool *matches
)
{
    size_t row;
    size_t column;

    if (matrix == NULL || matches == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        *matches = false;
        return NUMERUS_MATRIX_SUCCESS;
    }

    for (row = 0; row < numerus_matrix_rows(matrix); row++) {
        for (column = 0; column < numerus_matrix_columns(matrix); column++) {
            double value;
            double expected = row == column ? 1.0 : 0.0;
            numerus_matrix_status status =
                numerus_matrix_get(matrix, row, column, &value);

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (value != expected || (value == 0.0 && signbit(value))) {
                *matches = false;
                return NUMERUS_MATRIX_SUCCESS;
            }
        }
    }

    *matches = true;
    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status matrix_matches_zero(
    const numerus_matrix *matrix,
    bool *matches
)
{
    size_t row;
    size_t column;

    if (matrix == NULL || matches == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    for (row = 0; row < numerus_matrix_rows(matrix); row++) {
        for (column = 0; column < numerus_matrix_columns(matrix); column++) {
            double value;
            numerus_matrix_status status =
                numerus_matrix_get(matrix, row, column, &value);

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (value != 0.0 || signbit(value)) {
                *matches = false;
                return NUMERUS_MATRIX_SUCCESS;
            }
        }
    }

    *matches = true;
    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status create_dense_zero_result(
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
)
{
    size_t count;
    size_t bytes;
    size_t index;
    double *values;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;
    if (!numerus_size_multiply(rows, columns, &count) ||
        !numerus_size_multiply(count, sizeof(*values), &bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = numerus_multiply_alloc(bytes);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    for (index = 0; index < count; index++) {
        values[index] = 0.0;
    }

    status = numerus_matrix_create_dense(rows, columns, values, matrix);
    numerus_multiply_free(values);
    return status;
}

/*
 * The current multiplication loop evaluates zero-times-value terms. Only
 * nonnegative finite values make identity/zero shortcuts preserve those
 * IEEE-754 results, including the sign of zero.
 */
static numerus_matrix_status matrix_is_finite_nonnegative(
    const numerus_matrix *matrix,
    bool *eligible
)
{
    size_t row;
    size_t column;

    if (matrix == NULL || eligible == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    for (row = 0; row < numerus_matrix_rows(matrix); row++) {
        for (column = 0; column < numerus_matrix_columns(matrix); column++) {
            double value;
            numerus_matrix_status status =
                numerus_matrix_get(matrix, row, column, &value);

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (!isfinite(value) || value < 0.0 ||
                (value == 0.0 && signbit(value))) {
                *eligible = false;
                return NUMERUS_MATRIX_SUCCESS;
            }
        }
    }

    *eligible = true;
    return NUMERUS_MATRIX_SUCCESS;
}

/*
 * Try a row-compressed multiplication path for sparse right operands. This
 * path is only equivalent to the generic loop when both operands are finite
 * and nonnegative: omitted implicit-zero terms then contribute +0 and cannot
 * change cancellation or NaN propagation. Scratch-allocation failures fall
 * back to the generic implementation.
 */
static numerus_matrix_status matrix_try_sparse_right_product(
    const numerus_matrix *left,
    const numerus_matrix *right,
    size_t rows,
    size_t inner_dimension,
    size_t columns,
    size_t output_element_count,
    size_t output_allocation_size,
    numerus_matrix **matrix,
    bool *used
)
{
    size_t maximum_entries;
    size_t row_offsets_bytes;
    size_t column_indices_bytes;
    size_t sparse_values_bytes;
    size_t *row_offsets = NULL;
    size_t *column_indices = NULL;
    double *sparse_values = NULL;
    double *output = NULL;
    size_t nonzero_count = 0;
    size_t row;
    size_t column;
    size_t inner;
    bool eligible;
    numerus_matrix_status status;

    *used = false;
    if (rows < 2 ||
        numerus_matrix_storage_kind(right) != NUMERUS_STORAGE_SPARSE) {
        return NUMERUS_MATRIX_SUCCESS;
    }

    status = matrix_is_finite_nonnegative(left, &eligible);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }
    if (!eligible) {
        return NUMERUS_MATRIX_SUCCESS;
    }

    if (inner_dimension == SIZE_MAX ||
        !numerus_size_multiply(inner_dimension, columns, &maximum_entries) ||
        !numerus_size_multiply(
            inner_dimension + 1, sizeof(*row_offsets), &row_offsets_bytes
        )) {
        return NUMERUS_MATRIX_SUCCESS;
    }

    /* Count nonzero entries before allocating storage proportional to nnz. */
    for (inner = 0; inner < inner_dimension; inner++) {
        for (column = 0; column < columns; column++) {
            double value;

            status = numerus_matrix_get(right, inner, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (!isfinite(value) || value < 0.0 ||
                (value == 0.0 && signbit(value))) {
                return NUMERUS_MATRIX_SUCCESS;
            }
            if (value != 0.0) {
                nonzero_count++;
            }
        }
    }

    if (nonzero_count == 0 || nonzero_count > maximum_entries / 4) {
        return NUMERUS_MATRIX_SUCCESS;
    }
    if (!numerus_size_multiply(
            nonzero_count, sizeof(*column_indices), &column_indices_bytes
        ) ||
        !numerus_size_multiply(
            nonzero_count, sizeof(*sparse_values), &sparse_values_bytes
        )) {
        return NUMERUS_MATRIX_SUCCESS;
    }

    row_offsets = numerus_multiply_alloc(row_offsets_bytes);
    column_indices = numerus_multiply_alloc(column_indices_bytes);
    sparse_values = numerus_multiply_alloc(sparse_values_bytes);
    if (row_offsets == NULL || column_indices == NULL || sparse_values == NULL) {
        goto fallback;
    }

    {
        size_t entry_count = 0;
        for (inner = 0; inner < inner_dimension; inner++) {
            row_offsets[inner] = entry_count;
            for (column = 0; column < columns; column++) {
                double value;

                status = numerus_matrix_get(right, inner, column, &value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto error;
                }
                if (!isfinite(value) || value < 0.0 ||
                    (value == 0.0 && signbit(value))) {
                    goto fallback;
                }
                if (value != 0.0) {
                    if (entry_count >= nonzero_count) {
                        goto fallback;
                    }
                    column_indices[entry_count] = column;
                    sparse_values[entry_count] = value;
                    entry_count++;
                }
            }
            row_offsets[inner + 1] = entry_count;
        }
        if (entry_count != nonzero_count) {
            goto fallback;
        }
    }

    output = numerus_multiply_alloc(output_allocation_size);
    if (output == NULL) {
        goto fallback;
    }
    for (row = 0; row < output_element_count; row++) {
        output[row] = 0.0;
    }

    for (row = 0; row < rows; row++) {
        for (inner = 0; inner < inner_dimension; inner++) {
            double left_value;

            status = numerus_matrix_get(left, row, inner, &left_value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto error;
            }
            for (size_t entry = row_offsets[inner];
                 entry < row_offsets[inner + 1];
                 entry++) {
                output[row * columns + column_indices[entry]] +=
                    left_value * sparse_values[entry];
            }
        }
    }

    status = numerus_matrix_create_dense(rows, columns, output, matrix);
    numerus_multiply_free(output);
    numerus_multiply_free(sparse_values);
    numerus_multiply_free(column_indices);
    numerus_multiply_free(row_offsets);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *used = true;
    }
    return status;

fallback:
    numerus_multiply_free(output);
    numerus_multiply_free(sparse_values);
    numerus_multiply_free(column_indices);
    numerus_multiply_free(row_offsets);
    *used = false;
    return NUMERUS_MATRIX_SUCCESS;

error:
    numerus_multiply_free(output);
    numerus_multiply_free(sparse_values);
    numerus_multiply_free(column_indices);
    numerus_multiply_free(row_offsets);
    return status;
}

/**
 * Compute a materialized matrix product using the standard triple loop.
 *
 * The result is independent dense Storage. Parent read errors are propagated,
 * and no partial result is published. Dot products accumulate in double
 * precision using the natural left-to-right order.
 */
int numerus_matrix_multiply(
    const numerus_matrix *left,
    const numerus_matrix *right,
    numerus_matrix **matrix
)
{
    size_t rows;
    size_t inner_dimension;
    size_t columns;
    size_t element_count;
    size_t allocation_size;
    size_t row;
    size_t column;
    double *values;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (left == NULL || right == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(left);
    inner_dimension = numerus_matrix_columns(left);
    columns = numerus_matrix_columns(right);

    if (inner_dimension != numerus_matrix_rows(right)) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    if (rows == 0 || inner_dimension == 0 || columns == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(rows, columns, &element_count) ||
        !numerus_size_multiply(
            element_count, sizeof(*values), &allocation_size
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    {
        bool matches;
        bool eligible;

        status = matrix_matches_identity(right, &matches);
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        if (matches) {
            status = matrix_is_finite_nonnegative(left, &eligible);
            if (status != NUMERUS_MATRIX_SUCCESS) return status;
            if (eligible) return numerus_matrix_materialize(left, matrix);
        }

        status = matrix_matches_identity(left, &matches);
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        if (matches) {
            status = matrix_is_finite_nonnegative(right, &eligible);
            if (status != NUMERUS_MATRIX_SUCCESS) return status;
            if (eligible) return numerus_matrix_materialize(right, matrix);
        }

        status = matrix_matches_zero(right, &matches);
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        if (matches) {
            status = matrix_is_finite_nonnegative(left, &eligible);
            if (status != NUMERUS_MATRIX_SUCCESS) return status;
            if (eligible) return create_dense_zero_result(rows, columns, matrix);
        }

        status = matrix_matches_zero(left, &matches);
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        if (matches) {
            status = matrix_is_finite_nonnegative(right, &eligible);
            if (status != NUMERUS_MATRIX_SUCCESS) return status;
            if (eligible) return create_dense_zero_result(rows, columns, matrix);
        }
    }

    {
        bool used;
        status = matrix_try_sparse_right_product(
            left, right, rows, inner_dimension, columns,
            element_count, allocation_size, matrix, &used
        );
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        if (used) return NUMERUS_MATRIX_SUCCESS;
    }

    values = numerus_multiply_alloc(allocation_size);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            size_t inner;
            double sum = 0.0;

            for (inner = 0; inner < inner_dimension; inner++) {
                double left_value;
                double right_value;

                status = numerus_matrix_get(
                    left, row, inner, &left_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    numerus_multiply_free(values);
                    return status;
                }

                status = numerus_matrix_get(
                    right, inner, column, &right_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    numerus_multiply_free(values);
                    return status;
                }

                sum += left_value * right_value;
            }

            values[row * columns + column] = sum;
        }
    }

    status = numerus_matrix_create_dense(rows, columns, values, matrix);
    numerus_multiply_free(values);

    return status;
}
