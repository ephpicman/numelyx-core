/**
 * @file numerus_matrix_statistics.c
 * @brief Descriptive statistical reductions over Matrix values.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numerus_matrix.h"
#include <math.h>
static numerus_matrix_status compute_variance(const numerus_matrix *matrix,
    bool sample, double *result)
{
    size_t rows, columns, row, column, count = 0;
    double mean = 0.0, sum_squares = 0.0, variance;
    if (matrix == NULL || result == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    rows = numerus_matrix_rows(matrix); columns = numerus_matrix_columns(matrix);
    for (row = 0; row < rows; row++) for (column = 0; column < columns; column++) {
        double value, delta, delta_after;
        numerus_matrix_status status = numerus_matrix_get(matrix,row,column,&value);
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        if (!isfinite(value)) return NUMERUS_MATRIX_NON_FINITE;
        count++; delta = value - mean;
        if (!isfinite(delta)) return NUMERUS_MATRIX_NON_FINITE;
        mean += delta / (double) count; delta_after = value - mean;
        sum_squares += delta * delta_after;
        if (!isfinite(mean) || !isfinite(sum_squares)) return NUMERUS_MATRIX_NON_FINITE;
    }
    if (count == 0 || (sample && count < 2)) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    variance = sum_squares / (double)(sample ? count - 1 : count);
    if (!isfinite(variance) || variance < 0.0) return NUMERUS_MATRIX_NON_FINITE;
    *result = variance; return NUMERUS_MATRIX_SUCCESS;
}
numerus_matrix_status numerus_matrix_variance(const numerus_matrix *matrix,
    bool sample, double *variance)
{
    return compute_variance(matrix, sample, variance);
}
numerus_matrix_status numerus_matrix_standard_deviation(
    const numerus_matrix *matrix, bool sample, double *standard_deviation)
{
    double variance; numerus_matrix_status status;
    if (standard_deviation == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    status = compute_variance(matrix, sample, &variance);
    if (status != NUMERUS_MATRIX_SUCCESS) return status;
    *standard_deviation = sqrt(variance); return NUMERUS_MATRIX_SUCCESS;
}