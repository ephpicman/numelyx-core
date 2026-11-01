/**
 * @file numerus_matrix_spectral.c
 * @brief Spectral Matrix operations and eigenvalue-related routines.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numerus_matrix.h"

#include <math.h>

numerus_matrix_status numerus_matrix_spectral_norm(
    const numerus_matrix *matrix,
    double *norm
)
{
    numerus_matrix *u = NULL;
    numerus_matrix *singular_values = NULL;
    numerus_matrix *vt = NULL;
    double result;
    numerus_matrix_status status;

    if (matrix == NULL || norm == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_svd(matrix, &u, &singular_values, &vt);
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = numerus_matrix_get(singular_values, 0, 0, &result);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *norm = result;
    }

cleanup:
    numerus_matrix_destroy(vt);
    numerus_matrix_destroy(singular_values);
    numerus_matrix_destroy(u);
    return status;
}

numerus_matrix_status numerus_matrix_symmetric_spectral_radius(
    const numerus_matrix *matrix,
    double *radius
)
{
    numerus_matrix *eigenvalues = NULL;
    numerus_matrix *eigenvectors = NULL;
    double result = 0.0;
    size_t index;
    numerus_matrix_status status;

    if (matrix == NULL || radius == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    for (index = 0; index < numerus_matrix_rows(matrix); index++) {
        double value;
        status = numerus_matrix_get(eigenvalues, index, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
        if (fabs(value) > result) result = fabs(value);
    }

    *radius = result;

cleanup:
    numerus_matrix_destroy(eigenvectors);
    numerus_matrix_destroy(eigenvalues);
    return status;
}
