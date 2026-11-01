/**
 * @file numerus_matrix_inverse_gauss_jordan.c
 * @brief Gauss–Jordan matrix inversion and pivot management.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

numerus_matrix_status numerus_matrix_inverse_gauss_jordan(
    const numerus_matrix *matrix,
    numerus_matrix **inverse
)
{
    size_t size;
    size_t augmented_columns;
    size_t augmented_count;
    size_t augmented_bytes;
    size_t result_count;
    size_t result_bytes;
    size_t rank;
    size_t row;
    size_t column;
    double scale = 0.0;
    double relative_threshold;
    double *augmented = NULL;
    double *result = NULL;
    numerus_matrix_status status;

    if (inverse == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *inverse = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);
    if (size == 0 ||
        !numerus_size_multiply(size, 2, &augmented_columns) ||
        !numerus_size_multiply(size, augmented_columns, &augmented_count) ||
        !numerus_size_multiply(augmented_count, sizeof(*augmented), &augmented_bytes) ||
        !numerus_size_multiply(size, size, &result_count) ||
        !numerus_size_multiply(result_count, sizeof(*result), &result_bytes)) {
        return size == 0 ? NUMERUS_MATRIX_INVALID_ARGUMENT : NUMERUS_MATRIX_OVERFLOW;
    }

    status = numerus_matrix_rank(matrix, &rank);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }
    if (rank != size) {
        return NUMERUS_MATRIX_SINGULAR;
    }

    augmented = malloc(augmented_bytes);
    if (augmented == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    result = malloc(result_bytes);
    if (result == NULL) {
        free(augmented);
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;
            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto failure;
            }
            augmented[row * augmented_columns + column] = value;
            if (fabs(value) > scale) {
                scale = fabs(value);
            }
            augmented[row * augmented_columns + size + column] =
                row == column ? 1.0 : 0.0;
        }
    }

    relative_threshold = NUMERUS_EPSILON * (double) size;
    for (column = 0; column < size; column++) {
        size_t pivot_row = column;
        size_t candidate;
        double pivot_magnitude = 0.0;

        for (candidate = column; candidate < size; candidate++) {
            double magnitude = fabs(
                augmented[candidate * augmented_columns + column]
            );
            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = candidate;
            }
        }

        if (scale == 0.0 || pivot_magnitude / scale <= relative_threshold ||
            !isfinite(pivot_magnitude)) {
            status = NUMERUS_MATRIX_SINGULAR;
            goto failure;
        }

        if (pivot_row != column) {
            size_t swap_column;
            for (swap_column = 0; swap_column < augmented_columns; swap_column++) {
                double temporary = augmented[column * augmented_columns + swap_column];
                augmented[column * augmented_columns + swap_column] =
                    augmented[pivot_row * augmented_columns + swap_column];
                augmented[pivot_row * augmented_columns + swap_column] = temporary;
            }
        }

        {
            size_t pivot_index = column * augmented_columns;
            double pivot = augmented[pivot_index + column];

            for (size_t j = 0; j < augmented_columns; j++) {
                double normalized = augmented[pivot_index + j] / pivot;
                if (!isfinite(normalized)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto failure;
                }
                augmented[pivot_index + j] = normalized;
            }
            augmented[pivot_index + column] = 1.0;
        }

        for (row = 0; row < size; row++) {
            size_t row_index;
            double multiplier;

            if (row == column) {
                continue;
            }
            row_index = row * augmented_columns;
            multiplier = augmented[row_index + column];
            if (multiplier == 0.0) {
                continue;
            }
            for (size_t j = 0; j < augmented_columns; j++) {
                double updated;
                if (j == column) {
                    continue;
                }
                updated = augmented[row_index + j] -
                    multiplier * augmented[column * augmented_columns + j];
                if (!isfinite(updated)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto failure;
                }
                augmented[row_index + j] = updated;
            }
            augmented[row_index + column] = 0.0;
        }
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            result[row * size + column] =
                augmented[row * augmented_columns + size + column];
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, size, result, inverse
    );
    free(result);
    free(augmented);
    return status;

failure:
    free(result);
    free(augmented);
    return status;
}
