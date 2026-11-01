/**
 * @file numelyx_core_matrix_spectral.c
 * @brief Spectral Matrix operations and eigenvalue-related routines.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numelyx_core_matrix.h"

#include <math.h>

numelyx_core_matrix_status numelyx_core_matrix_spectral_norm(
    const numelyx_core_matrix *matrix,
    double *norm
)
{
    numelyx_core_matrix *u = NULL;
    numelyx_core_matrix *singular_values = NULL;
    numelyx_core_matrix *vt = NULL;
    double result;
    numelyx_core_matrix_status status;

    if (matrix == NULL || norm == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numelyx_core_matrix_svd(matrix, &u, &singular_values, &vt);
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = numelyx_core_matrix_get(singular_values, 0, 0, &result);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *norm = result;
    }

cleanup:
    numelyx_core_matrix_destroy(vt);
    numelyx_core_matrix_destroy(singular_values);
    numelyx_core_matrix_destroy(u);
    return status;
}

numelyx_core_matrix_status numelyx_core_matrix_symmetric_spectral_radius(
    const numelyx_core_matrix *matrix,
    double *radius
)
{
    numelyx_core_matrix *eigenvalues = NULL;
    numelyx_core_matrix *eigenvectors = NULL;
    double result = 0.0;
    size_t index;
    numelyx_core_matrix_status status;

    if (matrix == NULL || radius == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numelyx_core_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    for (index = 0; index < numelyx_core_matrix_rows(matrix); index++) {
        double value;
        status = numelyx_core_matrix_get(eigenvalues, index, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
        if (fabs(value) > result) result = fabs(value);
    }

    *radius = result;

cleanup:
    numelyx_core_matrix_destroy(eigenvectors);
    numelyx_core_matrix_destroy(eigenvalues);
    return status;
}
