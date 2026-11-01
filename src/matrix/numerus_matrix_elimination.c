/**
 * @file numerus_matrix_elimination.c
 * @brief Gaussian elimination and row-operation based Matrix routines.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"
#include "numerus_numeric.h"

#include <math.h>
#include <stdlib.h>

static numerus_matrix_status matrix_elimination_form(
    const numerus_matrix *matrix,
    bool reduced,
    numerus_matrix **result
)
{
    size_t rows;
    size_t columns;
    size_t element_count;
    size_t allocation_size;
    size_t row;
    size_t column;
    size_t pivot_row = 0;
    size_t pivot_column;
    size_t pivot_limit;
    double scale = 0.0;
    double relative_threshold;
    double *values;
    numerus_matrix_status status;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = NULL;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    if (!numerus_size_multiply(rows, columns, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*values), &allocation_size)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = malloc(allocation_size);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;

            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                free(values);
                return status;
            }
            if (!isfinite(value)) {
                free(values);
                return NUMERUS_MATRIX_NON_FINITE;
            }

            values[row * columns + column] = value;
            if (fabs(value) > scale) {
                scale = fabs(value);
            }
        }
    }

    if (scale != 0.0) {
        relative_threshold = NUMERUS_EPSILON *
            (double) (rows > columns ? rows : columns);
        pivot_limit = rows < columns ? rows : columns;

        for (pivot_column = 0;
             pivot_column < columns && pivot_row < pivot_limit;
             pivot_column++) {
            size_t candidate_row;
            size_t selected_row = pivot_row;
            double pivot_magnitude = 0.0;

            for (candidate_row = pivot_row;
                 candidate_row < rows;
                 candidate_row++) {
                double magnitude = fabs(
                    values[candidate_row * columns + pivot_column]
                );

                if (magnitude > pivot_magnitude) {
                    pivot_magnitude = magnitude;
                    selected_row = candidate_row;
                }
            }

            if (pivot_magnitude / scale <= relative_threshold) {
                for (row = pivot_row; row < rows; row++) {
                    values[row * columns + pivot_column] = 0.0;
                }
                continue;
            }

            if (selected_row != pivot_row) {
                size_t swap_column;

                for (swap_column = 0; swap_column < columns; swap_column++) {
                    double temporary = values[
                        pivot_row * columns + swap_column
                    ];
                    values[pivot_row * columns + swap_column] =
                        values[selected_row * columns + swap_column];
                    values[selected_row * columns + swap_column] = temporary;
                }
            }

            {
                double pivot = values[pivot_row * columns + pivot_column];

                if (reduced) {
                    for (column = 0; column < columns; column++) {
                        double normalized = values[
                            pivot_row * columns + column
                        ] / pivot;

                        if (!isfinite(normalized)) {
                            free(values);
                            return NUMERUS_MATRIX_NON_FINITE;
                        }
                        values[pivot_row * columns + column] = normalized;
                    }
                    values[pivot_row * columns + pivot_column] = 1.0;

                    for (row = 0; row < rows; row++) {
                        double multiplier;

                        if (row == pivot_row) {
                            continue;
                        }
                        multiplier = values[row * columns + pivot_column];
                        if (multiplier == 0.0) {
                            continue;
                        }

                        for (column = 0; column < columns; column++) {
                            double updated = values[row * columns + column] -
                                multiplier * values[
                                    pivot_row * columns + column
                                ];

                            if (!isfinite(updated)) {
                                free(values);
                                return NUMERUS_MATRIX_NON_FINITE;
                            }
                            values[row * columns + column] = updated;
                        }
                        values[row * columns + pivot_column] = 0.0;
                    }
                } else {
                    for (row = pivot_row + 1; row < rows; row++) {
                        double multiplier = values[
                            row * columns + pivot_column
                        ] / pivot;

                        values[row * columns + pivot_column] = 0.0;
                        for (column = pivot_column + 1;
                             column < columns;
                             column++) {
                            double updated = values[row * columns + column] -
                                multiplier * values[
                                    pivot_row * columns + column
                                ];

                            if (!isfinite(updated)) {
                                free(values);
                                return NUMERUS_MATRIX_NON_FINITE;
                            }
                            values[row * columns + column] = updated;
                        }
                    }
                }
            }

            pivot_row++;
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, columns, values, result
    );
    free(values);
    return status;
}

numerus_matrix_status numerus_matrix_row_echelon_form(
    const numerus_matrix *matrix,
    numerus_matrix **result
)
{
    return matrix_elimination_form(matrix, false, result);
}

numerus_matrix_status numerus_matrix_reduced_row_echelon_form(
    const numerus_matrix *matrix,
    numerus_matrix **result
)
{
    return matrix_elimination_form(matrix, true, result);
}
