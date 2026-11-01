/**
 * @file numerus_matrix_kronecker.c
 * @brief Kronecker products and their Matrix shape calculations.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"

#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_kronecker_alloc(size) emalloc(size)
# define numerus_kronecker_free(ptr) efree(ptr)
#else
# define numerus_kronecker_alloc(size) malloc(size)
# define numerus_kronecker_free(ptr) free(ptr)
#endif

/**
 * Compute the Kronecker product A ⊗ B into independent dense Storage.
 *
 * C[i * B.rows + k, j * B.columns + l] = A[i,j] * B[k,l].
 */
int numerus_matrix_kronecker_product(
    const numerus_matrix *left,
    const numerus_matrix *right,
    numerus_matrix **matrix
)
{
    size_t left_rows;
    size_t left_columns;
    size_t right_rows;
    size_t right_columns;
    size_t rows;
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

    left_rows = numerus_matrix_rows(left);
    left_columns = numerus_matrix_columns(left);
    right_rows = numerus_matrix_rows(right);
    right_columns = numerus_matrix_columns(right);

    if (!numerus_size_multiply(left_rows, right_rows, &rows) ||
        !numerus_size_multiply(left_columns, right_columns, &columns) ||
        !numerus_size_multiply(rows, columns, &element_count) ||
        !numerus_size_multiply(
            element_count, sizeof(*values), &allocation_size
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = numerus_kronecker_alloc(allocation_size);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < left_rows; row++) {
        for (column = 0; column < left_columns; column++) {
            double left_value;
            size_t right_row;
            size_t right_column;
            size_t output_row_base;
            size_t output_column_base;

            status = numerus_matrix_get(
                left, row, column, &left_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numerus_kronecker_free(values);
                return status;
            }

            output_row_base = row * right_rows;
            output_column_base = column * right_columns;

            for (right_row = 0; right_row < right_rows; right_row++) {
                for (right_column = 0; right_column < right_columns;
                     right_column++) {
                    double right_value;
                    size_t output_row = output_row_base + right_row;
                    size_t output_column =
                        output_column_base + right_column;

                    status = numerus_matrix_get(
                        right, right_row, right_column, &right_value
                    );
                    if (status != NUMERUS_MATRIX_SUCCESS) {
                        numerus_kronecker_free(values);
                        return status;
                    }

                    values[output_row * columns + output_column] =
                        left_value * right_value;
                }
            }
        }
    }

    status = numerus_matrix_create_dense(rows, columns, values, matrix);
    numerus_kronecker_free(values);

    return status;
}


/**
 * Compute the Kronecker sum A ⊕ B = A ⊗ I + I ⊗ B.
 *
 * Both inputs must be square. The result is an independent dense Matrix of
 * dimension left.rows * right.rows. IEEE-754 arithmetic is preserved by
 * evaluating both products, including zero-times-non-finite cases.
 */
int numerus_matrix_kronecker_sum(
    const numerus_matrix *left,
    const numerus_matrix *right,
    numerus_matrix **matrix
)
{
    size_t left_dimension;
    size_t right_dimension;
    size_t dimension;
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

    left_dimension = numerus_matrix_rows(left);
    right_dimension = numerus_matrix_rows(right);
    if (left_dimension != numerus_matrix_columns(left) ||
        right_dimension != numerus_matrix_columns(right)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    if (!numerus_size_multiply(left_dimension, right_dimension, &dimension) ||
        !numerus_size_multiply(dimension, dimension, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*values),
            &allocation_size)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = numerus_kronecker_alloc(allocation_size);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < dimension; row++) {
        size_t left_row = row / right_dimension;
        size_t right_row = row % right_dimension;

        for (column = 0; column < dimension; column++) {
            size_t left_column = column / right_dimension;
            size_t right_column = column % right_dimension;
            double left_value;
            double right_value;

            status = numerus_matrix_get(
                left, left_row, left_column, &left_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numerus_kronecker_free(values);
                return status;
            }

            status = numerus_matrix_get(
                right, right_row, right_column, &right_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numerus_kronecker_free(values);
                return status;
            }

            values[row * dimension + column] =
                left_value * (right_row == right_column ? 1.0 : 0.0) +
                (left_row == left_column ? 1.0 : 0.0) * right_value;
        }
    }

    status = numerus_matrix_create_dense(dimension, dimension, values, matrix);
    numerus_kronecker_free(values);
    return status;
}
