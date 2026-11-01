/**
 * @file numelyx_core_matrix_symmetric_eigen.c
 * @brief Eigenvalue decomposition for symmetric matrices.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_numeric.h"
#include "numelyx_core_size.h"

#include <math.h>
#include <limits.h>
#include <stdlib.h>

#ifdef _WIN32
# include <mkl.h>
#else
# include <lapacke.h>
#endif

# define matrix_eigen_alloc(size) malloc(size)
# define matrix_eigen_free(pointer) free(pointer)

static numelyx_core_matrix_status matrix_symmetric_eigen_lapack(
    const numelyx_core_matrix *matrix,
    size_t size,
    numelyx_core_matrix **eigenvalues,
    numelyx_core_matrix **eigenvectors
)
{
    size_t element_count;
    size_t matrix_bytes;
    size_t vector_bytes;
    size_t row;
    size_t column;
    double scale = 0.0;
    double relative_threshold = NUMERUS_EPSILON * (double) size;
    double *work = NULL;
    double *values = NULL;
    lapack_int info;
    numelyx_core_matrix_status status = NUMERUS_MATRIX_SUCCESS;

    if (size > (size_t) INT_MAX) return NUMERUS_MATRIX_NO_CONVERGENCE;
    if (!numelyx_core_size_multiply(size, size, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(*work), &matrix_bytes) ||
        !numelyx_core_size_multiply(size, sizeof(*values), &vector_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    work = matrix_eigen_alloc(matrix_bytes);
    values = matrix_eigen_alloc(vector_bytes);
    if (work == NULL || values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;
            status = numelyx_core_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row * size + column] = value;
            if (fabs(value) > scale) scale = fabs(value);
        }
    }

    if (scale > 0.0) {
        for (row = 0; row < size; row++) {
            for (column = row + 1; column < size; column++) {
                double upper = work[row * size + column] / scale;
                double lower = work[column * size + row] / scale;
                if (fabs(upper - lower) > relative_threshold) {
                    status = NUMERUS_MATRIX_NOT_SYMMETRIC;
                    goto cleanup;
                }
                work[row * size + column] =
                    0.5 * work[row * size + column] +
                    0.5 * work[column * size + row];
                work[column * size + row] = work[row * size + column];
            }
        }
    }

    info = LAPACKE_dsyevd(
        LAPACK_ROW_MAJOR, 'V', 'L',
        (lapack_int) size, work, (lapack_int) size, values
    );
    if (info == LAPACK_WORK_MEMORY_ERROR ||
        info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (info > 0) {
        status = NUMERUS_MATRIX_NO_CONVERGENCE;
        goto cleanup;
    }
    if (info < 0) {
        status = NUMERUS_MATRIX_INVALID_ARGUMENT;
        goto cleanup;
    }

    for (row = 0; row < size; row++) {
        if (!isfinite(values[row])) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
    }
    for (row = 0; row < element_count; row++) {
        if (!isfinite(work[row])) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
    }

    /* LAPACK returns ascending eigenvalues; the Matrix API specifies descending. */
    for (column = 0; column < size / 2; column++) {
        size_t opposite = size - 1 - column;
        double temporary = values[column];
        values[column] = values[opposite];
        values[opposite] = temporary;

        for (row = 0; row < size; row++) {
            temporary = work[row * size + column];
            work[row * size + column] = work[row * size + opposite];
            work[row * size + opposite] = temporary;
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        size, 1, values, eigenvalues
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        size, size, work, eigenvectors
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numelyx_core_matrix_destroy(*eigenvalues);
        *eigenvalues = NULL;
    }

cleanup:
    if (values != NULL) matrix_eigen_free(values);
    if (work != NULL) matrix_eigen_free(work);
    return status;
}

numelyx_core_matrix_status numelyx_core_matrix_symmetric_eigen(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **eigenvalues,
    numelyx_core_matrix **eigenvectors
)
{
    size_t size;
    size_t element_count;
    size_t matrix_bytes;
    size_t vector_bytes;
    size_t row;
    size_t column;
    size_t sweep;
    size_t max_sweeps_product;
    size_t max_sweeps;
    double scale = 0.0;
    double relative_threshold;
    double off_diagonal_threshold;
    double *work = NULL;
    double *values = NULL;
    double *vectors = NULL;
    bool converged = false;
    numelyx_core_matrix_status status;

    if (eigenvalues != NULL) *eigenvalues = NULL;
    if (eigenvectors != NULL && eigenvectors != eigenvalues) *eigenvectors = NULL;
    if (matrix == NULL || eigenvalues == NULL || eigenvectors == NULL ||
        eigenvalues == eigenvectors) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numelyx_core_matrix_rows(matrix);
    if (size <= (size_t) INT_MAX) {
        status = matrix_symmetric_eigen_lapack(
            matrix, size, eigenvalues, eigenvectors
        );
        if (status != NUMERUS_MATRIX_NO_CONVERGENCE) {
            return status;
        }
        /* Retain the cyclic Jacobi implementation as a non-convergence fallback. */
    }
    if (!numelyx_core_size_multiply(size, size, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(*work), &matrix_bytes) ||
        !numelyx_core_size_multiply(size, sizeof(*values), &vector_bytes) ||
        !numelyx_core_size_multiply(size, 5, &max_sweeps_product) ||
        !numelyx_core_size_add(30, max_sweeps_product, &max_sweeps)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    work = matrix_eigen_alloc(matrix_bytes);
    values = matrix_eigen_alloc(vector_bytes);
    vectors = matrix_eigen_alloc(matrix_bytes);
    if (work == NULL || values == NULL || vectors == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;
            status = numelyx_core_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row * size + column] = value;
            if (fabs(value) > scale) scale = fabs(value);
            vectors[row * size + column] = row == column ? 1.0 : 0.0;
        }
    }

    /* Scale-aware symmetry check: compare entries after dividing by the
     * largest magnitude so tolerance does not depend on the matrix's units.
     * Only round-off-sized asymmetry is averaged away before Jacobi sweeps.
     */
    relative_threshold = NUMERUS_EPSILON * (double) size;
    off_diagonal_threshold = scale * relative_threshold;

    if (scale > 0.0) {
        for (row = 0; row < size; row++) {
            for (column = row + 1; column < size; column++) {
                double upper = work[row * size + column] / scale;
                double lower = work[column * size + row] / scale;
                if (fabs(upper - lower) > relative_threshold) {
                    status = NUMERUS_MATRIX_NOT_SYMMETRIC;
                    goto cleanup;
                }
                /* Symmetrize only the tolerated round-off discrepancy. */
                work[row * size + column] =
                    0.5 * work[row * size + column] +
                    0.5 * work[column * size + row];
                work[column * size + row] = work[row * size + column];
            }
        }
    }

    /* Cyclic Jacobi iteration: each (p,q) rotation annihilates one symmetric
     * off-diagonal pair and applies the same orthogonal rotation to the columns
     * of vectors. Those accumulated columns are the eigenvectors of the input.
     */
    for (sweep = 0; sweep < max_sweeps; sweep++) {
        bool rotated = false;
        size_t p;

        for (p = 0; p < size; p++) {
            size_t q;

            for (q = p + 1; q < size; q++) {
                double app = work[p * size + p];
                double aqq = work[q * size + q];
                double apq = work[p * size + q];
                double tau;
                double tangent;
                double cosine;
                double sine;

                if (fabs(apq) <= off_diagonal_threshold) continue;

                tau = ((aqq / scale) - (app / scale)) /
                    (2.0 * (apq / scale));
                if (!isfinite(tau)) continue;
                tangent = copysign(1.0, tau) /
                    (fabs(tau) + hypot(1.0, tau));
                cosine = 1.0 / sqrt(1.0 + tangent * tangent);
                sine = cosine * tangent;

                work[p * size + p] = app - tangent * apq;
                work[q * size + q] = aqq + tangent * apq;
                work[p * size + q] = 0.0;
                work[q * size + p] = 0.0;
                if (!isfinite(work[p * size + p]) ||
                    !isfinite(work[q * size + q])) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }

                for (row = 0; row < size; row++) {
                    if (row == p || row == q) continue;

                    {
                        double left = work[row * size + p];
                        double right = work[row * size + q];
                        double new_left = cosine * left - sine * right;
                        double new_right = sine * left + cosine * right;

                        if (!isfinite(new_left) || !isfinite(new_right)) {
                            status = NUMERUS_MATRIX_NON_FINITE;
                            goto cleanup;
                        }
                        work[row * size + p] = new_left;
                        work[p * size + row] = new_left;
                        work[row * size + q] = new_right;
                        work[q * size + row] = new_right;
                    }
                }

                for (row = 0; row < size; row++) {
                    double left = vectors[row * size + p];
                    double right = vectors[row * size + q];
                    vectors[row * size + p] = cosine * left - sine * right;
                    vectors[row * size + q] = sine * left + cosine * right;
                }
                rotated = true;
            }
        }

        /* A sweep with no applied rotations terminates the iteration. In the
         * usual path this means every off-diagonal entry is below the threshold;
         * pairs whose rotation parameter is non-finite are also skipped, so
         * extreme inputs should exercise that path explicitly in regression tests.
         * Hitting the sweep limit instead returns NO_CONVERGENCE, never partial
         * eigenpairs.
         */
        if (!rotated) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        status = NUMERUS_MATRIX_NO_CONVERGENCE;
        goto cleanup;
    }

    for (row = 0; row < size; row++) {
        values[row] = work[row * size + row];
    }

    /* Sort eigenvalues in descending algebraic order with their eigenvectors. */
    for (column = 0; column < size; column++) {
        size_t best = column;
        size_t candidate;

        for (candidate = column + 1; candidate < size; candidate++) {
            if (values[candidate] > values[best]) best = candidate;
        }
        if (best != column) {
            double temporary = values[column];
            values[column] = values[best];
            values[best] = temporary;

            for (row = 0; row < size; row++) {
                temporary = vectors[row * size + column];
                vectors[row * size + column] = vectors[row * size + best];
                vectors[row * size + best] = temporary;
            }
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        size, 1, values, eigenvalues
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        size, size, vectors, eigenvectors
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numelyx_core_matrix_destroy(*eigenvalues);
        *eigenvalues = NULL;
    }

cleanup:
    if (vectors != NULL) matrix_eigen_free(vectors);
    if (values != NULL) matrix_eigen_free(values);
    if (work != NULL) matrix_eigen_free(work);
    return status;
}
