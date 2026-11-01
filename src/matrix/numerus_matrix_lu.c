/**
 * @file numerus_matrix_lu.c
 * @brief LU factorization, pivoting, and reusable factorization handles.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix_lu.h"
#include "numerus_size.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_lu_alloc(size) emalloc(size)
# define matrix_lu_free(ptr) efree(ptr)
#else
# define matrix_lu_alloc(size) malloc(size)
# define matrix_lu_free(ptr) free(ptr)
#endif

struct numerus_matrix_lu_factorization {
    size_t size;
    size_t rank;
    size_t *permutation;
    double *values;
    int permutation_sign;
    size_t reference_count;
};

numerus_matrix_status numerus_matrix_lu_retain(
    numerus_matrix_lu_factorization *factorization
)
{
    if (factorization == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (factorization->reference_count == SIZE_MAX) {
        return NUMERUS_MATRIX_OVERFLOW;
    }
    factorization->reference_count++;
    return NUMERUS_MATRIX_SUCCESS;
}

void numerus_matrix_lu_destroy(
    numerus_matrix_lu_factorization *factorization
)
{
    if (factorization == NULL) {
        return;
    }
    if (factorization->reference_count > 1) {
        factorization->reference_count--;
        return;
    }

    matrix_lu_free(factorization->values);
    matrix_lu_free(factorization->permutation);
    matrix_lu_free(factorization);
}

numerus_matrix_status numerus_matrix_lu_factorize(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization **factorization
)
{
    numerus_matrix_lu_factorization *result;
    size_t size;
    size_t element_count;
    size_t value_bytes;
    size_t permutation_bytes;
    size_t row;
    size_t column;

    if (factorization == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *factorization = NULL;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);
    if (!numerus_size_multiply(size, size, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(double), &value_bytes) ||
        !numerus_size_multiply(size, sizeof(size_t), &permutation_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    result = matrix_lu_alloc(sizeof(*result));
    if (result == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    result->size = size;
    result->rank = 0;
    result->reference_count = 1;
    result->permutation = NULL;
    result->values = NULL;
    result->permutation_sign = 1;

    result->values = matrix_lu_alloc(value_bytes);
    if (result->values == NULL) {
        numerus_matrix_lu_destroy(result);
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    result->permutation = matrix_lu_alloc(permutation_bytes);
    if (result->permutation == NULL) {
        numerus_matrix_lu_destroy(result);
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < size; row++) {
        result->permutation[row] = row;
        for (column = 0; column < size; column++) {
            numerus_matrix_status status = numerus_matrix_get(
                matrix, row, column,
                &result->values[row * size + column]
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                numerus_matrix_lu_destroy(result);
                return status;
            }
        }
    }

    for (column = 0; column < size; column++) {
        size_t pivot_row = column;
        size_t candidate_row;
        double pivot_magnitude = fabs(
            result->values[column * size + column]
        );

        for (candidate_row = column + 1;
             candidate_row < size;
             candidate_row++) {
            double magnitude = fabs(
                result->values[candidate_row * size + column]
            );

            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = candidate_row;
            }
        }

        if (result->values[pivot_row * size + column] == 0.0) {
            continue;
        }

        if (pivot_row != column) {
            size_t swap_column;
            size_t temporary_permutation = result->permutation[column];

            result->permutation[column] = result->permutation[pivot_row];
            result->permutation[pivot_row] = temporary_permutation;

            for (swap_column = 0; swap_column < size; swap_column++) {
                double temporary = result->values[
                    column * size + swap_column
                ];
                result->values[column * size + swap_column] =
                    result->values[pivot_row * size + swap_column];
                result->values[pivot_row * size + swap_column] = temporary;
            }

            result->permutation_sign = -result->permutation_sign;
        }

        {
            double pivot = result->values[column * size + column];

            result->rank++;
            for (row = column + 1; row < size; row++) {
                size_t update_column;
                double multiplier = result->values[row * size + column] / pivot;

                result->values[row * size + column] = multiplier;
                for (update_column = column + 1;
                     update_column < size;
                     update_column++) {
                    result->values[row * size + update_column] -=
                        multiplier * result->values[
                            column * size + update_column
                        ];
                }
            }
        }
    }

    *factorization = result;
    return NUMERUS_MATRIX_SUCCESS;
}

size_t numerus_matrix_lu_size(
    const numerus_matrix_lu_factorization *factorization
)
{
    return factorization == NULL ? 0 : factorization->size;
}

size_t numerus_matrix_lu_rank(
    const numerus_matrix_lu_factorization *factorization
)
{
    return factorization == NULL ? 0 : factorization->rank;
}

int numerus_matrix_lu_permutation_sign(
    const numerus_matrix_lu_factorization *factorization
)
{
    return factorization == NULL ? 0 : factorization->permutation_sign;
}

numerus_matrix_status numerus_matrix_lu_get_permutation(
    const numerus_matrix_lu_factorization *factorization,
    size_t factor_row,
    size_t *source_row
)
{
    if (factorization == NULL || source_row == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (factor_row >= factorization->size) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    *source_row = factorization->permutation[factor_row];
    return NUMERUS_MATRIX_SUCCESS;
}

numerus_matrix_status numerus_matrix_lu_get_lower(
    const numerus_matrix_lu_factorization *factorization,
    size_t row,
    size_t column,
    double *value
)
{
    double result;

    if (factorization == NULL || value == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (row >= factorization->size || column >= factorization->size) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    if (row == column) {
        result = 1.0;
    } else if (row < column) {
        result = 0.0;
    } else {
        result = factorization->values[row * factorization->size + column];
    }

    *value = result;
    return NUMERUS_MATRIX_SUCCESS;
}

numerus_matrix_status numerus_matrix_lu_get_upper(
    const numerus_matrix_lu_factorization *factorization,
    size_t row,
    size_t column,
    double *value
)
{
    double result;

    if (factorization == NULL || value == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (row >= factorization->size || column >= factorization->size) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    result = row > column
        ? 0.0
        : factorization->values[row * factorization->size + column];

    *value = result;
    return NUMERUS_MATRIX_SUCCESS;
}
