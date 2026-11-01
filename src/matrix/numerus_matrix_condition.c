/**
 * @file numerus_matrix_condition.c
 * @brief Matrix condition-number estimation and numerical stability helpers.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numerus_matrix.h"

#include <math.h>

numerus_matrix_status numerus_matrix_condition_estimate_one(
    const numerus_matrix *matrix,
    double *condition_estimate
)
{
    numerus_matrix *inverse = NULL;
    double matrix_norm;
    double inverse_norm;
    numerus_matrix_status status;

    if (matrix == NULL || condition_estimate == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    status = numerus_matrix_inverse(matrix, &inverse);
    if (status == NUMERUS_MATRIX_SINGULAR) {
        *condition_estimate = INFINITY;
        return NUMERUS_MATRIX_SUCCESS;
    }
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    status = numerus_matrix_norm_one(matrix, &matrix_norm);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(inverse);
        return status;
    }

    status = numerus_matrix_norm_one(inverse, &inverse_norm);
    numerus_matrix_destroy(inverse);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    *condition_estimate = matrix_norm * inverse_norm;
    return NUMERUS_MATRIX_SUCCESS;
}
