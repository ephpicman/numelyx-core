/**
 * @file numerus_matrix_definiteness.c
 * @brief Positive-definiteness and related symmetric-matrix tests.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include <math.h>

numerus_matrix_status numerus_matrix_classify_definiteness(
    const numerus_matrix *matrix,
    int *positive_definite,
    int *positive_semidefinite
)
{
    numerus_matrix *eigenvalues = NULL;
    numerus_matrix *eigenvectors = NULL;
    size_t size, index;
    double scale = 0.0, threshold, minimum = 0.0;
    numerus_matrix_status status;

    if (matrix == NULL || positive_definite == NULL ||
        positive_semidefinite == NULL ||
        positive_definite == positive_semidefinite) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    status = numerus_matrix_symmetric_eigen(matrix, &eigenvalues, &eigenvectors);
    if (status != NUMERUS_MATRIX_SUCCESS) return status;

    size = numerus_matrix_rows(matrix);
    for (index = 0; index < size; index++) {
        double value;
        status = numerus_matrix_get(eigenvalues, index, 0, &value);
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
    numerus_matrix_destroy(eigenvectors);
    numerus_matrix_destroy(eigenvalues);
    return status;
}
