/**
 * @file numerus_matrix_log_determinant.c
 * @brief Log-determinant evaluation with sign and overflow handling.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix.h"
#include "numerus_matrix_lu.h"

#include <math.h>

numerus_matrix_status numerus_matrix_log_determinant(
    const numerus_matrix *matrix,
    int *sign,
    double *log_abs_determinant
)
{
    numerus_matrix_lu_factorization *factorization = NULL;
    size_t size;
    size_t row;
    size_t column;
    int result_sign;
    double sum = 0.0;
    double compensation = 0.0;
    numerus_matrix_status status;

    if (matrix == NULL || sign == NULL || log_abs_determinant == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);

    /*
     * LU currently does not reject non-finite source entries itself. Validate
     * the complete input before factorization so NaN pivot comparisons cannot
     * be mistaken for a singular or successful factorization.
     */
    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;

            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (!isfinite(value)) {
                return NUMERUS_MATRIX_NON_FINITE;
            }
        }
    }

    status = numerus_matrix_lu_factorize(matrix, &factorization);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    /*
     * Check the packed factors as well: finite source values can still
     * overflow during elimination. Inspect L below the diagonal and U on
     * and above it without exposing the packed factorization storage.
     */
    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;

            if (row > column) {
                status = numerus_matrix_lu_get_lower(
                    factorization, row, column, &value
                );
            } else {
                status = numerus_matrix_lu_get_upper(
                    factorization, row, column, &value
                );
            }
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto cleanup;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }
    }

    if (numerus_matrix_lu_rank(factorization) != size) {
        status = NUMERUS_MATRIX_SINGULAR;
        goto cleanup;
    }

    result_sign = numerus_matrix_lu_permutation_sign(factorization);
    for (row = 0; row < size; row++) {
        double diagonal;
        double term;
        double adjusted;
        double next_sum;

        status = numerus_matrix_lu_get_upper(
            factorization, row, row, &diagonal
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            goto cleanup;
        }
        if (diagonal == 0.0) {
            status = NUMERUS_MATRIX_SINGULAR;
            goto cleanup;
        }
        if (!isfinite(diagonal)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }

        if (diagonal < 0.0) {
            result_sign = -result_sign;
        }

        term = log(fabs(diagonal));
        if (!isfinite(term)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }

        /* Kahan compensated summation reduces cancellation across scales. */
        adjusted = term - compensation;
        next_sum = sum + adjusted;
        if (!isfinite(next_sum)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
        compensation = (next_sum - sum) - adjusted;
        sum = next_sum;
    }

    *sign = result_sign;
    *log_abs_determinant = sum;
    status = NUMERUS_MATRIX_SUCCESS;

cleanup:
    numerus_matrix_lu_destroy(factorization);
    return status;
}
