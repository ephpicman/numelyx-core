/**
 * @file numelyx_core_matrix_pseudoinverse.c
 * @brief Moore–Penrose pseudoinverse computation with rank tolerance.
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
#include <stdlib.h>

# define matrix_pseudoinverse_alloc(size) malloc(size)
# define matrix_pseudoinverse_free(pointer) free(pointer)

numelyx_core_matrix_status numelyx_core_matrix_pseudoinverse(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **pseudoinverse
)
{
    numelyx_core_matrix *u = NULL;
    numelyx_core_matrix *singular_values = NULL;
    numelyx_core_matrix *vt = NULL;
    size_t rows;
    size_t columns;
    size_t rank_bound;
    size_t result_count;
    size_t result_bytes;
    size_t row;
    size_t column;
    size_t component;
    double largest_singular_value = 0.0;
    double threshold;
    double *values = NULL;
    numelyx_core_matrix_status status;

    if (pseudoinverse == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *pseudoinverse = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numelyx_core_matrix_rows(matrix);
    columns = numelyx_core_matrix_columns(matrix);
    rank_bound = rows < columns ? rows : columns;
    if (!numelyx_core_size_multiply(columns, rows, &result_count) ||
        !numelyx_core_size_multiply(result_count, sizeof(*values), &result_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numelyx_core_matrix_svd(matrix, &u, &singular_values, &vt);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    for (component = 0; component < rank_bound; component++) {
        double singular_value;
        status = numelyx_core_matrix_get(
            singular_values, component, 0, &singular_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            goto cleanup;
        }
        if (!isfinite(singular_value) || singular_value < 0.0) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
        if (singular_value > largest_singular_value) {
            largest_singular_value = singular_value;
        }
    }

    threshold = NUMERUS_EPSILON *
        (double) (rows > columns ? rows : columns) *
        largest_singular_value;
    if (!isfinite(threshold)) {
        status = NUMERUS_MATRIX_NON_FINITE;
        goto cleanup;
    }

    values = matrix_pseudoinverse_alloc(result_bytes);
    if (values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    /*
     * A+ = V * Sigma+ * U^T. Singular values at or below the documented
     * scale-aware threshold are treated as zero.
     */
    for (row = 0; row < columns; row++) {
        for (column = 0; column < rows; column++) {
            double value = 0.0;

            for (component = 0; component < rank_bound; component++) {
                double singular_value;
                double u_value;
                double vt_value;

                status = numelyx_core_matrix_get(
                    singular_values, component, 0, &singular_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                if (singular_value <= threshold) {
                    continue;
                }

                status = numelyx_core_matrix_get(u, column, component, &u_value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                status = numelyx_core_matrix_get(vt, component, row, &vt_value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }

                value += vt_value * (u_value / singular_value);
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
            values[row * rows + column] = value;
        }
    }

    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        columns, rows, values, pseudoinverse
    );

cleanup:
    if (values != NULL) matrix_pseudoinverse_free(values);
    numelyx_core_matrix_destroy(vt);
    numelyx_core_matrix_destroy(singular_values);
    numelyx_core_matrix_destroy(u);
    return status;
}
