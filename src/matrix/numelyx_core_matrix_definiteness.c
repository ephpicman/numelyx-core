/**
 * @file numelyx_core_matrix_definiteness.c
 * @brief Positive-definiteness and related symmetric-matrix tests.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_numeric.h"
#include <math.h>

numelyx_core_matrix_status numelyx_core_matrix_classify_definiteness(
    const numelyx_core_matrix *matrix,
    int *positive_definite,
    int *positive_semidefinite
)
{
    numelyx_core_matrix *eigenvalues = NULL;
    numelyx_core_matrix *eigenvectors = NULL;
    size_t size, index;
    double scale = 0.0, threshold, minimum = 0.0;
    numelyx_core_matrix_status status;

    if (matrix == NULL || positive_definite == NULL ||
        positive_semidefinite == NULL ||
        positive_definite == positive_semidefinite) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    status = numelyx_core_matrix_symmetric_eigen(matrix, &eigenvalues, &eigenvectors);
    if (status != NUMERUS_MATRIX_SUCCESS) return status;

    size = numelyx_core_matrix_rows(matrix);
    for (index = 0; index < size; index++) {
        double value;
        status = numelyx_core_matrix_get(eigenvalues, index, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
        if (fabs(value) > scale) scale = fabs(value);
        if (index == 0 || value < minimum) minimum = value;
    }

    /* Scale-relative tolerance preserves classification at tiny magnitudes. */
    threshold = scale * NUMERUS_EPSILON * (double) size;
    *positive_definite = minimum > threshold;
    *positive_semidefinite = minimum >= -threshold;
    status = NUMERUS_MATRIX_SUCCESS;

cleanup:
    numelyx_core_matrix_destroy(eigenvectors);
    numelyx_core_matrix_destroy(eigenvalues);
    return status;
}
