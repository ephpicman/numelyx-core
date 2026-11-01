/**
 * @file numelyx_core_matrix_condition.c
 * @brief Matrix condition-number estimation and numerical stability helpers.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numelyx_core_matrix.h"

#include <math.h>

numelyx_core_matrix_status numelyx_core_matrix_condition_estimate_one(
    const numelyx_core_matrix *matrix,
    double *condition_estimate
)
{
    numelyx_core_matrix *inverse = NULL;
    double matrix_norm;
    double inverse_norm;
    numelyx_core_matrix_status status;

    if (matrix == NULL || condition_estimate == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    status = numelyx_core_matrix_inverse(matrix, &inverse);
    if (status == NUMERUS_MATRIX_SINGULAR) {
        *condition_estimate = INFINITY;
        return NUMERUS_MATRIX_SUCCESS;
    }
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    status = numelyx_core_matrix_norm_one(matrix, &matrix_norm);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numelyx_core_matrix_destroy(inverse);
        return status;
    }

    status = numelyx_core_matrix_norm_one(inverse, &inverse_norm);
    numelyx_core_matrix_destroy(inverse);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    *condition_estimate = matrix_norm * inverse_norm;
    return NUMERUS_MATRIX_SUCCESS;
}
