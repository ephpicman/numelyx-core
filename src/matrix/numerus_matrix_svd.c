/**
 * @file numerus_matrix_svd.c
 * @brief Singular value decomposition using one-sided Jacobi orthogonalization.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define matrix_svd_alloc(size) emalloc(size)
# define matrix_svd_free(pointer) efree(pointer)
#else
# define matrix_svd_alloc(size) malloc(size)
# define matrix_svd_free(pointer) free(pointer)
#endif

/*
 * Return the largest absolute entry in one working column. Scaling each
 * column before dot products keeps the Jacobi correlation calculation from
 * overflowing solely because the column has a large magnitude.
 */
static double matrix_svd_column_scale(
    const double *values,
    size_t rows,
    size_t columns,
    size_t column
)
{
    size_t row;
    double scale = 0.0;

    for (row = 0; row < rows; row++) {
        double magnitude = fabs(values[row * columns + column]);
        if (magnitude > scale) scale = magnitude;
    }
    return scale;
}

/*
 * One-sided Jacobi SVD for rows >= columns. The working columns are
 * orthogonalized directly, avoiding the squared condition number of A^T A.
 */
static numerus_matrix_status matrix_svd_tall(
    const numerus_matrix *matrix,
    size_t rows,
    size_t columns,
    double **u_result,
    double **singular_result,
    double **vt_result
)
{
    size_t work_count;
    size_t square_count;
    size_t work_bytes;
    size_t square_bytes;
    size_t singular_bytes;
    size_t max_sweeps_product;
    size_t max_sweeps;
    size_t row;
    size_t column;
    size_t sweep;
    bool converged = false;
    double *work = NULL;
    double *v = NULL;
    double *u = NULL;
    double *singular = NULL;
    double *vt = NULL;
    numerus_matrix_status status = NUMERUS_MATRIX_SUCCESS;

    *u_result = NULL;
    *singular_result = NULL;
    *vt_result = NULL;

    if (rows < columns) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (!numerus_size_multiply(rows, columns, &work_count) ||
        !numerus_size_multiply(work_count, sizeof(*work), &work_bytes) ||
        !numerus_size_multiply(columns, columns, &square_count) ||
        !numerus_size_multiply(square_count, sizeof(*v), &square_bytes) ||
        !numerus_size_multiply(columns, sizeof(*singular), &singular_bytes) ||
        !numerus_size_multiply(columns, 5, &max_sweeps_product) ||
        !numerus_size_add(30, max_sweeps_product, &max_sweeps)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    work = matrix_svd_alloc(work_bytes);
    v = matrix_svd_alloc(square_bytes);
    u = matrix_svd_alloc(work_bytes);
    singular = matrix_svd_alloc(singular_bytes);
    vt = matrix_svd_alloc(square_bytes);
    if (work == NULL || v == NULL || u == NULL || singular == NULL || vt == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    /* Copy through the Matrix accessor so roots and lazy views share one path.
     * Reject non-finite input before it can contaminate the rotations.
     */
    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;
            status = numerus_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
            work[row * columns + column] = value;
        }
    }

    for (row = 0; row < columns; row++) {
        for (column = 0; column < columns; column++) {
            v[row * columns + column] = row == column ? 1.0 : 0.0;
        }
    }

    /*
     * Each Jacobi sweep visits every column pair. A rotation reduces their
     * normalized correlation while applying the same orthogonal transform to
     * V; when an entire sweep performs no rotations, the columns are treated
     * as orthogonal to the configured numeric tolerance.
     */
    for (sweep = 0; sweep < max_sweeps; sweep++) {
        bool rotated = false;
        size_t p;

        for (p = 0; p < columns; p++) {
            size_t q;

            for (q = p + 1; q < columns; q++) {
                double scale_p = matrix_svd_column_scale(
                    work, rows, columns, p
                );
                double scale_q = matrix_svd_column_scale(
                    work, rows, columns, q
                );
                double norm_p_squared = 0.0;
                double norm_q_squared = 0.0;
                double dot = 0.0;
                double correlation;
                double norm_p;
                double norm_q;
                double largest_norm;
                double p_ratio;
                double q_ratio;
                double gamma_ratio;
                double tau;
                double tangent;
                double cosine;
                double sine;

                if (scale_p == 0.0 || scale_q == 0.0) continue;

                for (row = 0; row < rows; row++) {
                    double left = work[row * columns + p] / scale_p;
                    double right = work[row * columns + q] / scale_q;
                    norm_p_squared += left * left;
                    norm_q_squared += right * right;
                    dot += left * right;
                }

                correlation = dot / sqrt(norm_p_squared * norm_q_squared);
                if (!isfinite(correlation)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
                if (fabs(correlation) <= NUMERUS_EPSILON) continue;

                norm_p = scale_p * sqrt(norm_p_squared);
                norm_q = scale_q * sqrt(norm_q_squared);
                if (!isfinite(norm_p) || !isfinite(norm_q)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }

                largest_norm = norm_p > norm_q ? norm_p : norm_q;
                p_ratio = norm_p / largest_norm;
                q_ratio = norm_q / largest_norm;
                gamma_ratio = p_ratio * q_ratio * correlation;
                if (gamma_ratio == 0.0) continue;

                tau = (q_ratio * q_ratio - p_ratio * p_ratio) /
                    (2.0 * gamma_ratio);
                if (!isfinite(tau)) continue;
                tangent = copysign(1.0, tau) /
                    (fabs(tau) + hypot(1.0, tau));
                cosine = 1.0 / sqrt(1.0 + tangent * tangent);
                sine = cosine * tangent;

                for (row = 0; row < rows; row++) {
                    double left = work[row * columns + p];
                    double right = work[row * columns + q];
                    work[row * columns + p] = cosine * left - sine * right;
                    work[row * columns + q] = sine * left + cosine * right;
                    if (!isfinite(work[row * columns + p]) ||
                        !isfinite(work[row * columns + q])) {
                        status = NUMERUS_MATRIX_NON_FINITE;
                        goto cleanup;
                    }
                }

                for (row = 0; row < columns; row++) {
                    double left = v[row * columns + p];
                    double right = v[row * columns + q];
                    v[row * columns + p] = cosine * left - sine * right;
                    v[row * columns + q] = sine * left + cosine * right;
                }
                rotated = true;
            }
        }

        if (!rotated) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        status = NUMERUS_MATRIX_NO_CONVERGENCE;
        goto cleanup;
    }

    for (column = 0; column < columns; column++) {
        double norm = 0.0;
        for (row = 0; row < rows; row++) {
            norm = hypot(norm, work[row * columns + column]);
            if (!isfinite(norm)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto cleanup;
            }
        }
        singular[column] = norm;
    }

    /*
     * Keep each singular value paired with its transformed A column and V
     * column. Sorting all three consistently is required because U, S, and Vᵀ
     * must describe the same decomposition after this point.
     */
    for (column = 0; column < columns; column++) {
        size_t best = column;
        size_t candidate;

        for (candidate = column + 1; candidate < columns; candidate++) {
            if (singular[candidate] > singular[best]) best = candidate;
        }
        if (best != column) {
            double temporary = singular[column];
            singular[column] = singular[best];
            singular[best] = temporary;

            for (row = 0; row < rows; row++) {
                temporary = work[row * columns + column];
                work[row * columns + column] = work[row * columns + best];
                work[row * columns + best] = temporary;
            }
            for (row = 0; row < columns; row++) {
                temporary = v[row * columns + column];
                v[row * columns + column] = v[row * columns + best];
                v[row * columns + best] = temporary;
            }
        }
    }

    /*
     * Normalize nonzero work columns to form U. For an exact zero singular
     * value, the corresponding U column is undefined by A; construct a
     * deterministic orthogonal completion by projecting coordinate vectors
     * against the U columns already accepted.
     */
    for (column = 0; column < columns; column++) {
        if (singular[column] > 0.0) {
            for (row = 0; row < rows; row++) {
                u[row * columns + column] =
                    work[row * columns + column] / singular[column];
            }
        } else {
            size_t candidate;
            bool found = false;

            for (candidate = 0; candidate < rows && !found; candidate++) {
                size_t previous;
                double norm = 0.0;

                for (row = 0; row < rows; row++) {
                    u[row * columns + column] = row == candidate ? 1.0 : 0.0;
                }

                for (previous = 0; previous < column; previous++) {
                    double dot = 0.0;
                    for (row = 0; row < rows; row++) {
                        dot += u[row * columns + previous] *
                            u[row * columns + column];
                    }
                    for (row = 0; row < rows; row++) {
                        u[row * columns + column] -=
                            dot * u[row * columns + previous];
                    }
                }

                for (row = 0; row < rows; row++) {
                    norm = hypot(norm, u[row * columns + column]);
                }
                if (norm > 0.5) {
                    for (row = 0; row < rows; row++) {
                        u[row * columns + column] /= norm;
                    }
                    found = true;
                }
            }

            if (!found) {
                status = NUMERUS_MATRIX_NO_CONVERGENCE;
                goto cleanup;
            }
        }
    }

    for (row = 0; row < columns; row++) {
        for (column = 0; column < columns; column++) {
            vt[row * columns + column] = v[column * columns + row];
        }
    }

    *u_result = u;
    *singular_result = singular;
    *vt_result = vt;
    u = NULL;
    singular = NULL;
    vt = NULL;

cleanup:
    if (vt != NULL) matrix_svd_free(vt);
    if (singular != NULL) matrix_svd_free(singular);
    if (u != NULL) matrix_svd_free(u);
    if (v != NULL) matrix_svd_free(v);
    if (work != NULL) matrix_svd_free(work);
    return status;
}

numerus_matrix_status numerus_matrix_svd(
    const numerus_matrix *matrix,
    numerus_matrix **u,
    numerus_matrix **singular_values,
    numerus_matrix **vt
)
{
    numerus_matrix *transpose = NULL;
    const numerus_matrix *tall_matrix = matrix;
    size_t rows;
    size_t columns;
    size_t k;
    size_t u_count;
    size_t vt_count;
    size_t singular_bytes;
    size_t u_bytes;
    size_t vt_bytes;
    size_t row;
    size_t column;
    double *tall_u = NULL;
    double *tall_singular = NULL;
    double *tall_vt = NULL;
    double *u_values = NULL;
    double *singular_values_data = NULL;
    double *vt_values = NULL;
    numerus_matrix_status status;

    if (u != NULL) *u = NULL;
    if (singular_values != NULL && singular_values != u) *singular_values = NULL;
    if (vt != NULL && vt != u && vt != singular_values) *vt = NULL;
    if (matrix == NULL || u == NULL || singular_values == NULL || vt == NULL ||
        u == singular_values || u == vt || singular_values == vt) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(matrix);
    columns = numerus_matrix_columns(matrix);
    k = rows < columns ? rows : columns;

    if (rows < columns) {
        status = (numerus_matrix_status) numerus_matrix_create_transpose(
            (numerus_matrix *) matrix, &transpose
        );
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        tall_matrix = transpose;
    }

    {
        size_t tall_rows = numerus_matrix_rows(tall_matrix);
        size_t tall_columns = numerus_matrix_columns(tall_matrix);
        status = matrix_svd_tall(
            tall_matrix, tall_rows, tall_columns,
            &tall_u, &tall_singular, &tall_vt
        );
    }
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    if (!numerus_size_multiply(rows, k, &u_count) ||
        !numerus_size_multiply(u_count, sizeof(*u_values), &u_bytes) ||
        !numerus_size_multiply(k, columns, &vt_count) ||
        !numerus_size_multiply(vt_count, sizeof(*vt_values), &vt_bytes) ||
        !numerus_size_multiply(k, sizeof(*singular_values_data), &singular_bytes)) {
        status = NUMERUS_MATRIX_OVERFLOW;
        goto cleanup;
    }

    u_values = matrix_svd_alloc(u_bytes);
    singular_values_data = matrix_svd_alloc(singular_bytes);
    vt_values = matrix_svd_alloc(vt_bytes);
    if (u_values == NULL || singular_values_data == NULL || vt_values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (column = 0; column < k; column++) {
        singular_values_data[column] = tall_singular[column];
    }

    if (rows >= columns) {
        for (row = 0; row < rows; row++) {
            for (column = 0; column < k; column++) {
                u_values[row * k + column] = tall_u[row * k + column];
            }
        }
        for (row = 0; row < k; row++) {
            for (column = 0; column < columns; column++) {
                vt_values[row * columns + column] =
                    tall_vt[row * k + column];
            }
        }
    } else {
        /* A^T = U_t S V_t^T, so A = V_t S U_t^T. */
        for (row = 0; row < rows; row++) {
            for (column = 0; column < k; column++) {
                u_values[row * k + column] = tall_vt[column * k + row];
            }
        }
        for (row = 0; row < k; row++) {
            for (column = 0; column < columns; column++) {
                vt_values[row * columns + column] = tall_u[column * k + row];
            }
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        rows, k, u_values, u
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto cleanup;

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        k, 1, singular_values_data, singular_values
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(*u);
        *u = NULL;
        goto cleanup;
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        k, columns, vt_values, vt
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(*singular_values);
        numerus_matrix_destroy(*u);
        *singular_values = NULL;
        *u = NULL;
    }

cleanup:
    if (vt_values != NULL) matrix_svd_free(vt_values);
    if (singular_values_data != NULL) matrix_svd_free(singular_values_data);
    if (u_values != NULL) matrix_svd_free(u_values);
    if (tall_vt != NULL) matrix_svd_free(tall_vt);
    if (tall_singular != NULL) matrix_svd_free(tall_singular);
    if (tall_u != NULL) matrix_svd_free(tall_u);
    numerus_matrix_destroy(transpose);
    return status;
}
