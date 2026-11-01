/**
 * @file numelyx_core_matrix_analysis.c
 * @brief Matrix structural and numerical analysis routines.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_size.h"
#include "numelyx_core_numeric.h"

#include <math.h>
#include <stdlib.h>

static numelyx_core_matrix_status matrix_reduce(
    const numelyx_core_matrix *matrix,
    double *sum_result,
    double *minimum_result,
    double *maximum_result
)
{
    double sum = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    bool has_finite_or_infinite_value = false;
    bool saw_nan = false;
    size_t row;
    size_t column;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    for (row = 0; row < numelyx_core_matrix_rows(matrix); row++) {
        for (column = 0; column < numelyx_core_matrix_columns(matrix); column++) {
            double value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            sum += value;
            if (isnan(value)) {
                saw_nan = true;
                continue;
            }

            if (!has_finite_or_infinite_value) {
                minimum = value;
                maximum = value;
                has_finite_or_infinite_value = true;
            } else {
                if (value < minimum) {
                    minimum = value;
                }
                if (value > maximum) {
                    maximum = value;
                }
            }
        }
    }

    if (sum_result != NULL) {
        *sum_result = sum;
    }
    if (minimum_result != NULL) {
        *minimum_result = saw_nan ? NAN : minimum;
    }
    if (maximum_result != NULL) {
        *maximum_result = saw_nan ? NAN : maximum;
    }

    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_trace(
    const numelyx_core_matrix *matrix,
    double *trace
)
{
    double result = 0.0;
    size_t index;
    size_t size;

    if (matrix == NULL || trace == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numelyx_core_matrix_rows(matrix);
    for (index = 0; index < size; index++) {
        double value;
        numelyx_core_matrix_status status = numelyx_core_matrix_get(
            matrix, index, index, &value
        );

        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        result += value;
    }

    *trace = result;
    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_sum(
    const numelyx_core_matrix *matrix,
    double *sum
)
{
    if (matrix == NULL || sum == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return matrix_reduce(matrix, sum, NULL, NULL);
}

numelyx_core_matrix_status numelyx_core_matrix_min(
    const numelyx_core_matrix *matrix,
    double *minimum
)
{
    if (matrix == NULL || minimum == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return matrix_reduce(matrix, NULL, minimum, NULL);
}

numelyx_core_matrix_status numelyx_core_matrix_max(
    const numelyx_core_matrix *matrix,
    double *maximum
)
{
    if (matrix == NULL || maximum == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return matrix_reduce(matrix, NULL, NULL, maximum);
}

numelyx_core_matrix_status numelyx_core_matrix_mean(
    const numelyx_core_matrix *matrix,
    double *mean
)
{
    double sum;
    size_t element_count;
    numelyx_core_matrix_status status;

    if (matrix == NULL || mean == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numelyx_core_size_multiply(
            numelyx_core_matrix_rows(matrix),
            numelyx_core_matrix_columns(matrix),
            &element_count
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = matrix_reduce(matrix, &sum, NULL, NULL);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    *mean = sum / (double) element_count;
    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status matrix_axis_aggregates(
    const numelyx_core_matrix *matrix,
    bool aggregate_rows,
    bool calculate_mean,
    numelyx_core_matrix **result
)
{
    size_t output_count;
    size_t reduction_count;
    size_t bytes;
    size_t outer;
    size_t inner;
    double *values;
    numelyx_core_matrix_status status;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = NULL;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    output_count = aggregate_rows
        ? numelyx_core_matrix_rows(matrix)
        : numelyx_core_matrix_columns(matrix);
    reduction_count = aggregate_rows
        ? numelyx_core_matrix_columns(matrix)
        : numelyx_core_matrix_rows(matrix);

    if (!numelyx_core_size_multiply(output_count, sizeof(*values), &bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = malloc(bytes);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (outer = 0; outer < output_count; outer++) {
        double aggregate = 0.0;

        for (inner = 0; inner < reduction_count; inner++) {
            size_t row = aggregate_rows ? outer : inner;
            size_t column = aggregate_rows ? inner : outer;
            double value;

            status = numelyx_core_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                free(values);
                return status;
            }

            aggregate += value;
        }

        values[outer] = calculate_mean
            ? aggregate / (double) reduction_count
            : aggregate;
    }

    status = numelyx_core_matrix_create_dense(
        aggregate_rows ? output_count : 1,
        aggregate_rows ? 1 : output_count,
        values,
        result
    );
    free(values);

    return (numelyx_core_matrix_status) status;
}

numelyx_core_matrix_status numelyx_core_matrix_row_sums(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **sums
)
{
    return matrix_axis_aggregates(matrix, true, false, sums);
}

numelyx_core_matrix_status numelyx_core_matrix_column_sums(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **sums
)
{
    return matrix_axis_aggregates(matrix, false, false, sums);
}

numelyx_core_matrix_status numelyx_core_matrix_row_means(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **means
)
{
    return matrix_axis_aggregates(matrix, true, true, means);
}

numelyx_core_matrix_status numelyx_core_matrix_column_means(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **means
)
{
    return matrix_axis_aggregates(matrix, false, true, means);
}


/*
 * Frobenius norm uses hypot() to avoid the avoidable overflow and underflow
 * of summing squared elements. NaN is deliberately propagated even when an
 * infinity is also present, consistently with the aggregate APIs.
 */
numelyx_core_matrix_status numelyx_core_matrix_norm_frobenius(
    const numelyx_core_matrix *matrix,
    double *norm
)
{
    double result = 0.0;
    bool saw_nan = false;
    size_t row;
    size_t column;

    if (matrix == NULL || norm == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    for (row = 0; row < numelyx_core_matrix_rows(matrix); row++) {
        for (column = 0; column < numelyx_core_matrix_columns(matrix); column++) {
            double value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (isnan(value)) {
                saw_nan = true;
                continue;
            }
            result = hypot(result, value);
        }
    }

    *norm = saw_nan ? NAN : result;
    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status matrix_induced_norm(
    const numelyx_core_matrix *matrix,
    bool reduce_columns,
    double *norm
)
{
    size_t outer_count;
    size_t inner_count;
    size_t outer;
    double maximum = 0.0;
    bool saw_nan = false;

    if (matrix == NULL || norm == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    outer_count = reduce_columns
        ? numelyx_core_matrix_columns(matrix)
        : numelyx_core_matrix_rows(matrix);
    inner_count = reduce_columns
        ? numelyx_core_matrix_rows(matrix)
        : numelyx_core_matrix_columns(matrix);

    for (outer = 0; outer < outer_count; outer++) {
        double sum = 0.0;
        size_t inner;

        for (inner = 0; inner < inner_count; inner++) {
            size_t row = reduce_columns ? inner : outer;
            size_t column = reduce_columns ? outer : inner;
            double value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (isnan(value)) {
                saw_nan = true;
                continue;
            }
            sum += fabs(value);
        }

        if (sum > maximum) {
            maximum = sum;
        }
    }

    *norm = saw_nan ? NAN : maximum;
    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_norm_one(
    const numelyx_core_matrix *matrix,
    double *norm
)
{
    return matrix_induced_norm(matrix, true, norm);
}

numelyx_core_matrix_status numelyx_core_matrix_norm_infinity(
    const numelyx_core_matrix *matrix,
    double *norm
)
{
    return matrix_induced_norm(matrix, false, norm);
}


typedef enum {
    MATRIX_FINITE_PREDICATE_HAS_NAN,
    MATRIX_FINITE_PREDICATE_HAS_INFINITY,
    MATRIX_FINITE_PREDICATE_ALL_FINITE
} matrix_finite_predicate;

static numelyx_core_matrix_status matrix_evaluate_finite_predicate(
    const numelyx_core_matrix *matrix,
    matrix_finite_predicate predicate,
    bool *result
)
{
    size_t row;
    size_t column;

    if (matrix == NULL || result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    for (row = 0; row < numelyx_core_matrix_rows(matrix); row++) {
        for (column = 0; column < numelyx_core_matrix_columns(matrix); column++) {
            double value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            if (predicate == MATRIX_FINITE_PREDICATE_HAS_NAN && isnan(value)) {
                *result = true;
                return NUMERUS_MATRIX_SUCCESS;
            }
            if (predicate == MATRIX_FINITE_PREDICATE_HAS_INFINITY && isinf(value)) {
                *result = true;
                return NUMERUS_MATRIX_SUCCESS;
            }
            if (predicate == MATRIX_FINITE_PREDICATE_ALL_FINITE && !isfinite(value)) {
                *result = false;
                return NUMERUS_MATRIX_SUCCESS;
            }
        }
    }

    *result = predicate == MATRIX_FINITE_PREDICATE_ALL_FINITE;
    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_has_nan(
    const numelyx_core_matrix *matrix,
    bool *has_nan
)
{
    return matrix_evaluate_finite_predicate(
        matrix, MATRIX_FINITE_PREDICATE_HAS_NAN, has_nan
    );
}

numelyx_core_matrix_status numelyx_core_matrix_has_infinity(
    const numelyx_core_matrix *matrix,
    bool *has_infinity
)
{
    return matrix_evaluate_finite_predicate(
        matrix, MATRIX_FINITE_PREDICATE_HAS_INFINITY, has_infinity
    );
}

numelyx_core_matrix_status numelyx_core_matrix_is_finite(
    const numelyx_core_matrix *matrix,
    bool *is_finite
)
{
    return matrix_evaluate_finite_predicate(
        matrix, MATRIX_FINITE_PREDICATE_ALL_FINITE, is_finite
    );
}


numelyx_core_matrix_status numelyx_core_matrix_is_skew_symmetric(
    const numelyx_core_matrix *matrix,
    bool *is_skew_symmetric
)
{
    size_t row;
    size_t column;
    bool result = true;

    if (matrix == NULL || is_skew_symmetric == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        *is_skew_symmetric = false;
        return NUMERUS_MATRIX_SUCCESS;
    }

    for (row = 0; row < numelyx_core_matrix_rows(matrix); row++) {
        for (column = row; column < numelyx_core_matrix_columns(matrix); column++) {
            double value;
            double transposed_value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            if (row == column) {
                if (!numelyx_core_double_is_zero(value)) {
                    result = false;
                    goto done;
                }
                continue;
            }

            status = numelyx_core_matrix_get(
                matrix, column, row, &transposed_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            if (!numelyx_core_double_equals(value, -transposed_value)) {
                result = false;
                goto done;
            }
        }
    }

done:
    *is_skew_symmetric = result;
    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_is_orthogonal(
    const numelyx_core_matrix *matrix,
    bool *is_orthogonal
)
{
    size_t size;
    size_t row;
    size_t other_row;

    if (matrix == NULL || is_orthogonal == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    size = numelyx_core_matrix_rows(matrix);
    if (size != numelyx_core_matrix_columns(matrix)) {
        *is_orthogonal = false;
        return NUMERUS_MATRIX_SUCCESS;
    }

    for (row = 0; row < size; row++) {
        for (other_row = row; other_row < size; other_row++) {
            double dot_product = 0.0;
            size_t column;

            for (column = 0; column < size; column++) {
                double left_value;
                double right_value;
                numelyx_core_matrix_status status = numelyx_core_matrix_get(
                    matrix, row, column, &left_value
                );

                if (status != NUMERUS_MATRIX_SUCCESS) {
                    return status;
                }

                status = numelyx_core_matrix_get(
                    matrix, other_row, column, &right_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    return status;
                }

                dot_product += left_value * right_value;
            }

            if (!numelyx_core_double_equals(
                    dot_product, row == other_row ? 1.0 : 0.0
                )) {
                *is_orthogonal = false;
                return NUMERUS_MATRIX_SUCCESS;
            }
        }
    }

    *is_orthogonal = true;
    return NUMERUS_MATRIX_SUCCESS;
}


numelyx_core_matrix_status numelyx_core_matrix_rank(
    const numelyx_core_matrix *matrix,
    size_t *rank
)
{
    size_t rows;
    size_t columns;
    size_t element_count;
    size_t allocation_size;
    size_t row;
    size_t column;
    size_t pivot_index = 0;
    size_t pivot_limit;
    size_t rank_result = 0;
    bool is_diagonal = true;
    double scale = 0.0;
    double relative_threshold;
    double *values;

    if (matrix == NULL || rank == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numelyx_core_matrix_rows(matrix);
    columns = numelyx_core_matrix_columns(matrix);
    if (!numelyx_core_size_multiply(rows, columns, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(*values), &allocation_size)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = malloc(allocation_size);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                free(values);
                return status;
            }
            if (!isfinite(value)) {
                free(values);
                return NUMERUS_MATRIX_NON_FINITE;
            }

            values[row * columns + column] = value;
            if (row != column && value != 0.0) {
                is_diagonal = false;
            }
            if (fabs(value) > scale) {
                scale = fabs(value);
            }
        }
    }

    if (scale == 0.0) {
        free(values);
        *rank = 0;
        return NUMERUS_MATRIX_SUCCESS;
    }

    relative_threshold = NUMERUS_EPSILON *
        (double) (rows > columns ? rows : columns);
    pivot_limit = rows < columns ? rows : columns;

    /*
     * Complete pivoting on an exactly diagonal matrix only counts diagonal
     * entries above the same global, scale-relative threshold. Avoid the
     * cubic elimination pass for this common structured case.
     */
    if (is_diagonal) {
        for (row = 0; row < pivot_limit; row++) {
            if (fabs(values[row * columns + row]) / scale >
                relative_threshold) {
                rank_result++;
            }
        }
        free(values);
        *rank = rank_result;
        return NUMERUS_MATRIX_SUCCESS;
    }

    while (pivot_index < pivot_limit) {
        size_t pivot_row = pivot_index;
        size_t pivot_column = pivot_index;
        size_t candidate_row;
        double pivot_magnitude = 0.0;

        /*
         * Complete pivoting makes the rank estimate less dependent on the
         * original row/column ordering than a diagonal-only pivot search.
         */
        for (candidate_row = pivot_index;
             candidate_row < rows;
             candidate_row++) {
            size_t candidate_column;

            for (candidate_column = pivot_index;
                 candidate_column < columns;
                 candidate_column++) {
                double magnitude = fabs(
                    values[candidate_row * columns + candidate_column]
                );

                if (magnitude > pivot_magnitude) {
                    pivot_magnitude = magnitude;
                    pivot_row = candidate_row;
                    pivot_column = candidate_column;
                }
            }
        }

        if (pivot_magnitude / scale <= relative_threshold) {
            break;
        }

        if (pivot_row != pivot_index) {
            size_t swap_column;

            for (swap_column = 0; swap_column < columns; swap_column++) {
                double temporary = values[
                    pivot_index * columns + swap_column
                ];
                values[pivot_index * columns + swap_column] =
                    values[pivot_row * columns + swap_column];
                values[pivot_row * columns + swap_column] = temporary;
            }
        }

        if (pivot_column != pivot_index) {
            size_t swap_row;

            for (swap_row = 0; swap_row < rows; swap_row++) {
                double temporary = values[
                    swap_row * columns + pivot_index
                ];
                values[swap_row * columns + pivot_index] =
                    values[swap_row * columns + pivot_column];
                values[swap_row * columns + pivot_column] = temporary;
            }
        }

        {
            double pivot = values[pivot_index * columns + pivot_index];

            for (row = pivot_index + 1; row < rows; row++) {
                size_t update_column;
                double multiplier = values[row * columns + pivot_index] / pivot;

                values[row * columns + pivot_index] = 0.0;
                for (update_column = pivot_index + 1;
                     update_column < columns;
                     update_column++) {
                    double updated = values[row * columns + update_column] -
                        multiplier * values[pivot_index * columns + update_column];

                    if (!isfinite(updated)) {
                        free(values);
                        return NUMERUS_MATRIX_NON_FINITE;
                    }
                    values[row * columns + update_column] = updated;
                }
            }
        }

        rank_result++;
        pivot_index++;
    }

    free(values);
    *rank = rank_result;
    return NUMERUS_MATRIX_SUCCESS;
}
