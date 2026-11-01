/**
 * @file numelyx_core_matrix_exponential.c
 * @brief Matrix exponential algorithms and convergence checks.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_size.h"

#include <math.h>
#include <stdlib.h>

# define numelyx_core_exponential_alloc(size) malloc(size)
# define numelyx_core_exponential_free(ptr) free(ptr)

/* The order-13 Padé approximant is used after scaling the matrix so its
 * one-norm is at most theta_13; the selected exponent is undone by repeated
 * squaring after evaluating the rational approximation.
 */
#define NUMERUS_PADE_THETA_13 5.371920351148152
#define NUMERUS_PADE_COEFFICIENT_0 64764752532480000.0
#define NUMERUS_PADE_COEFFICIENT_1 32382376266240000.0
#define NUMERUS_PADE_COEFFICIENT_2 7771770303897600.0
#define NUMERUS_PADE_COEFFICIENT_3 1187353796428800.0
#define NUMERUS_PADE_COEFFICIENT_4 129060195264000.0
#define NUMERUS_PADE_COEFFICIENT_5 10559470521600.0
#define NUMERUS_PADE_COEFFICIENT_6 670442572800.0
#define NUMERUS_PADE_COEFFICIENT_7 33522128640.0
#define NUMERUS_PADE_COEFFICIENT_8 1323241920.0
#define NUMERUS_PADE_COEFFICIENT_9 40840800.0
#define NUMERUS_PADE_COEFFICIENT_10 960960.0
#define NUMERUS_PADE_COEFFICIENT_11 16380.0
#define NUMERUS_PADE_COEFFICIENT_12 182.0
#define NUMERUS_PADE_COEFFICIENT_13 1.0

static numelyx_core_matrix_status matrix_one_norm(
    const numelyx_core_matrix *matrix,
    double *norm
)
{
    size_t rows;
    size_t columns;
    size_t column;
    double maximum = 0.0;

    if (matrix == NULL || norm == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    rows = numelyx_core_matrix_rows(matrix);
    columns = numelyx_core_matrix_columns(matrix);

    for (column = 0; column < columns; column++) {
        size_t row;
        double sum = 0.0;

        for (row = 0; row < rows; row++) {
            double value;
            numelyx_core_matrix_status status = (numelyx_core_matrix_status)
                numelyx_core_matrix_get(matrix, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (!isfinite(value)) {
                return NUMERUS_MATRIX_NON_FINITE;
            }
            sum += fabs(value);
            if (!isfinite(sum)) {
                return NUMERUS_MATRIX_NON_FINITE;
            }
        }
        if (sum > maximum) {
            maximum = sum;
        }
    }

    *norm = maximum;
    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status matrix_scale_copy(
    const numelyx_core_matrix *source,
    int exponent,
    numelyx_core_matrix **result
)
{
    size_t rows;
    size_t columns;
    size_t count;
    size_t bytes;
    size_t row;
    size_t column;
    double *values;
    numelyx_core_matrix_status status;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = NULL;
    if (source == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numelyx_core_matrix_rows(source);
    columns = numelyx_core_matrix_columns(source);
    if (!numelyx_core_size_multiply(rows, columns, &count) ||
        !numelyx_core_size_multiply(count, sizeof(*values), &bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }
    values = numelyx_core_exponential_alloc(bytes);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double value;
            status = (numelyx_core_matrix_status)
                numelyx_core_matrix_get(source, row, column, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numelyx_core_exponential_free(values);
                return status;
            }
            if (!isfinite(value)) {
                numelyx_core_exponential_free(values);
                return NUMERUS_MATRIX_NON_FINITE;
            }
            value = scalbn(value, exponent);
            if (!isfinite(value)) {
                numelyx_core_exponential_free(values);
                return NUMERUS_MATRIX_NON_FINITE;
            }
            values[row * columns + column] = value;
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, columns, values, result
    );
    numelyx_core_exponential_free(values);
    return status;
}

static numelyx_core_matrix_status matrix_linear_combination(
    const numelyx_core_matrix *const *matrices,
    const double *coefficients,
    size_t term_count,
    numelyx_core_matrix **result
)
{
    size_t rows;
    size_t columns;
    size_t count;
    size_t bytes;
    size_t row;
    size_t column;
    size_t term;
    double *values;
    numelyx_core_matrix_status status;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = NULL;
    if (matrices == NULL || coefficients == NULL || term_count == 0 ||
        matrices[0] == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numelyx_core_matrix_rows(matrices[0]);
    columns = numelyx_core_matrix_columns(matrices[0]);
    if (!numelyx_core_size_multiply(rows, columns, &count) ||
        !numelyx_core_size_multiply(count, sizeof(*values), &bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }
    for (term = 0; term < term_count; term++) {
        if (matrices[term] == NULL ||
            numelyx_core_matrix_rows(matrices[term]) != rows ||
            numelyx_core_matrix_columns(matrices[term]) != columns) {
            return NUMERUS_MATRIX_DIMENSION_MISMATCH;
        }
        if (!isfinite(coefficients[term])) {
            return NUMERUS_MATRIX_NON_FINITE;
        }
    }

    values = numelyx_core_exponential_alloc(bytes);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double sum = 0.0;
            for (term = 0; term < term_count; term++) {
                double value;
                status = (numelyx_core_matrix_status) numelyx_core_matrix_get(
                    matrices[term], row, column, &value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    numelyx_core_exponential_free(values);
                    return status;
                }
                if (!isfinite(value)) {
                    numelyx_core_exponential_free(values);
                    return NUMERUS_MATRIX_NON_FINITE;
                }
                sum += coefficients[term] * value;
                if (!isfinite(sum)) {
                    numelyx_core_exponential_free(values);
                    return NUMERUS_MATRIX_NON_FINITE;
                }
            }
            values[row * columns + column] = sum;
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        rows, columns, values, result
    );
    numelyx_core_exponential_free(values);
    return status;
}

static numelyx_core_matrix_status matrix_multiply_finite(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    numelyx_core_matrix **result
)
{
    size_t row;
    size_t column;
    numelyx_core_matrix_status status;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = NULL;
    status = (numelyx_core_matrix_status) numelyx_core_matrix_multiply(
        left, right, result
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    for (row = 0; row < numelyx_core_matrix_rows(*result); row++) {
        for (column = 0; column < numelyx_core_matrix_columns(*result); column++) {
            double value;
            status = (numelyx_core_matrix_status) numelyx_core_matrix_get(
                *result, row, column, &value
            );
            if (status != NUMERUS_MATRIX_SUCCESS || !isfinite(value)) {
                numelyx_core_matrix_destroy(*result);
                *result = NULL;
                return status == NUMERUS_MATRIX_SUCCESS
                    ? NUMERUS_MATRIX_NON_FINITE
                    : status;
            }
        }
    }
    return NUMERUS_MATRIX_SUCCESS;
}

int numelyx_core_matrix_exponential(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **exponential
)
{
    static const double coefficients_u[] = {
        NUMERUS_PADE_COEFFICIENT_13,
        NUMERUS_PADE_COEFFICIENT_11,
        NUMERUS_PADE_COEFFICIENT_9
    };
    static const double coefficients_v[] = {
        NUMERUS_PADE_COEFFICIENT_12,
        NUMERUS_PADE_COEFFICIENT_10,
        NUMERUS_PADE_COEFFICIENT_8
    };
    numelyx_core_matrix *scaled = NULL;
    numelyx_core_matrix *identity = NULL;
    numelyx_core_matrix *a2 = NULL;
    numelyx_core_matrix *a4 = NULL;
    numelyx_core_matrix *a6 = NULL;
    numelyx_core_matrix *inner_u = NULL;
    numelyx_core_matrix *product_u = NULL;
    numelyx_core_matrix *outer_u = NULL;
    numelyx_core_matrix *u = NULL;
    numelyx_core_matrix *inner_v = NULL;
    numelyx_core_matrix *product_v = NULL;
    numelyx_core_matrix *v = NULL;
    numelyx_core_matrix *denominator = NULL;
    numelyx_core_matrix *numerator = NULL;
    numelyx_core_matrix *result = NULL;
    numelyx_core_matrix *squared = NULL;
    const numelyx_core_matrix *terms[5];
    double values[5];
    double norm;
    double ratio;
    int scaling = 0;
    int step;
    numelyx_core_matrix_status status;

    if (exponential == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *exponential = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    /* Scaling-and-squaring controls the approximation error: reduce A to
     * A / 2^s before forming powers, then square exp(A / 2^s) exactly s times.
     * The one-norm and all intermediates are checked for non-finite values.
     */
    status = matrix_one_norm(matrix, &norm);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    if (norm > NUMERUS_PADE_THETA_13) {
        ratio = norm / NUMERUS_PADE_THETA_13;
        if (!isfinite(ratio)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
        scaling = (int) ceil(log2(ratio));
        if (scaling < 0) {
            scaling = 0;
        }
    }

    status = matrix_scale_copy(matrix, -scaling, &scaled);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_identity(
        numelyx_core_matrix_rows(matrix), &identity
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    /* Build A², A⁴, and A⁶ once. The [13/13] Padé numerator and denominator
     * are polynomials in these even powers; sharing them avoids recomputing
     * expensive matrix products in the two polynomial branches.
     */
    status = matrix_multiply_finite(scaled, scaled, &a2);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = matrix_multiply_finite(a2, a2, &a4);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = matrix_multiply_finite(a4, a2, &a6);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    terms[0] = a6;
    terms[1] = a4;
    terms[2] = a2;
    values[0] = coefficients_u[0];
    values[1] = NUMERUS_PADE_COEFFICIENT_11;
    values[2] = NUMERUS_PADE_COEFFICIENT_9;
    status = matrix_linear_combination(terms, values, 3, &inner_u);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = matrix_multiply_finite(a6, inner_u, &product_u);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    terms[0] = product_u;
    terms[1] = a6;
    terms[2] = a4;
    terms[3] = a2;
    terms[4] = identity;
    values[0] = 1.0;
    values[1] = NUMERUS_PADE_COEFFICIENT_7;
    values[2] = NUMERUS_PADE_COEFFICIENT_5;
    values[3] = NUMERUS_PADE_COEFFICIENT_3;
    values[4] = NUMERUS_PADE_COEFFICIENT_1;
    status = matrix_linear_combination(terms, values, 5, &outer_u);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = matrix_multiply_finite(scaled, outer_u, &u);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    terms[0] = a6;
    terms[1] = a4;
    terms[2] = a2;
    values[0] = coefficients_v[0];
    values[1] = NUMERUS_PADE_COEFFICIENT_10;
    values[2] = NUMERUS_PADE_COEFFICIENT_8;
    status = matrix_linear_combination(terms, values, 3, &inner_v);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = matrix_multiply_finite(a6, inner_v, &product_v);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    terms[0] = product_v;
    terms[1] = a6;
    terms[2] = a4;
    terms[3] = a2;
    terms[4] = identity;
    values[0] = 1.0;
    values[1] = NUMERUS_PADE_COEFFICIENT_6;
    values[2] = NUMERUS_PADE_COEFFICIENT_4;
    values[3] = NUMERUS_PADE_COEFFICIENT_2;
    values[4] = NUMERUS_PADE_COEFFICIENT_0;
    status = matrix_linear_combination(terms, values, 5, &v);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    /* For the diagonal Padé approximant, exp(A) ≈ (V + U) / (V - U).
     * Solve (V - U) X = V + U rather than explicitly forming an inverse;
     * this reuses the Matrix linear solver and avoids an extra inverse product.
     */
    terms[0] = v;
    terms[1] = u;
    values[0] = 1.0;
    values[1] = -1.0;
    status = matrix_linear_combination(terms, values, 2, &denominator);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    values[1] = 1.0;
    status = matrix_linear_combination(terms, values, 2, &numerator);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_solve(
        denominator, numerator, &result
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }
    status = matrix_one_norm(result, &norm);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    /* Undo the initial scaling: exp(A) = exp(A / 2^s)^(2^s). Each square
     * replaces the previous result only after multiplication succeeds, so a
     * failed step leaves the current owned result available for cleanup.
     */
    for (step = 0; step < scaling; step++) {
        status = matrix_multiply_finite(result, result, &squared);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            goto cleanup;
        }
        numelyx_core_matrix_destroy(result);
        result = squared;
        squared = NULL;
    }

    *exponential = result;
    result = NULL;
    status = NUMERUS_MATRIX_SUCCESS;

cleanup:
    numelyx_core_matrix_destroy(scaled);
    numelyx_core_matrix_destroy(identity);
    numelyx_core_matrix_destroy(a2);
    numelyx_core_matrix_destroy(a4);
    numelyx_core_matrix_destroy(a6);
    numelyx_core_matrix_destroy(inner_u);
    numelyx_core_matrix_destroy(product_u);
    numelyx_core_matrix_destroy(outer_u);
    numelyx_core_matrix_destroy(u);
    numelyx_core_matrix_destroy(inner_v);
    numelyx_core_matrix_destroy(product_v);
    numelyx_core_matrix_destroy(v);
    numelyx_core_matrix_destroy(denominator);
    numelyx_core_matrix_destroy(numerator);
    numelyx_core_matrix_destroy(result);
    numelyx_core_matrix_destroy(squared);
    return status;
}
