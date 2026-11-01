/**
 * @file numerus_matrix_symmetric_eigen.c
 * @brief Eigenvalue decomposition for symmetric matrices.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_eigen_alloc(size) emalloc(size)
# define matrix_eigen_free(pointer) efree(pointer)
#else
# define matrix_eigen_alloc(size) malloc(size)
# define matrix_eigen_free(pointer) free(pointer)
#endif

numerus_matrix_status numerus_matrix_symmetric_eigen(
    const numerus_matrix *matrix,
    numerus_matrix **eigenvalues,
    numerus_matrix **eigenvectors
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
    numerus_matrix_status status;

    if (eigenvalues != NULL) *eigenvalues = NULL;
    if (eigenvectors != NULL && eigenvectors != eigenvalues) *eigenvectors = NULL;
    if (matrix == NULL || eigenvalues == NULL || eigenvectors == NULL ||
        eigenvalues == eigenvectors) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(matrix) != numerus_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numerus_matrix_rows(matrix);
    if (!numerus_size_multiply(size, size, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*work), &matrix_bytes) ||
        !numerus_size_multiply(size, sizeof(*values), &vector_bytes) ||
        !numerus_size_multiply(size, 5, &max_sweeps_product) ||
        !numerus_size_add(30, max_sweeps_product, &max_sweeps)) {
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
            status = numerus_matrix_get(matrix, row, column, &value);
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

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, 1, values, eigenvalues
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, size, vectors, eigenvectors
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(*eigenvalues);
        *eigenvalues = NULL;
    }

cleanup:
    if (vectors != NULL) matrix_eigen_free(vectors);
    if (values != NULL) matrix_eigen_free(values);
    if (work != NULL) matrix_eigen_free(work);
    return status;
}
