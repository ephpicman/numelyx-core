/**
 * @file numerus_matrix_symmetric_functions.c
 * @brief Matrix functions specialized for symmetric matrices.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"
#include "numerus_size.h"

#include <math.h>
#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define symmetric_function_alloc(size) emalloc(size)
# define symmetric_function_free(pointer) efree(pointer)
#else
# define symmetric_function_alloc(size) malloc(size)
# define symmetric_function_free(pointer) free(pointer)
#endif

typedef enum {
    SYMMETRIC_FUNCTION_SQUARE_ROOT = 0,
    SYMMETRIC_FUNCTION_LOGARITHM,
    SYMMETRIC_FUNCTION_SINE,
    SYMMETRIC_FUNCTION_COSINE
} symmetric_matrix_function;

static numerus_matrix_status symmetric_matrix_function_apply(
    const numerus_matrix *matrix,
    symmetric_matrix_function function,
    numerus_matrix **result
)
{
    numerus_matrix *eigenvalues = NULL;
    numerus_matrix *eigenvectors = NULL;
    size_t size;
    size_t count;
    size_t bytes;
    size_t transformed_bytes;
    size_t row;
    size_t column;
    size_t k;
    double scale = 0.0;
    double tolerance;
    double *transformed = NULL;
    double *values = NULL;
    numerus_matrix_status status;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_symmetric_eigen(
        matrix, &eigenvalues, &eigenvectors
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        goto cleanup;
    }

    size = numerus_matrix_rows(matrix);
    if (!numerus_size_multiply(size, size, &count) ||
        !numerus_size_multiply(count, sizeof(*values), &bytes) ||
        !numerus_size_multiply(size, sizeof(*transformed), &transformed_bytes)) {
        status = NUMERUS_MATRIX_OVERFLOW;
        goto cleanup;
    }
    transformed = symmetric_function_alloc(transformed_bytes);
    values = symmetric_function_alloc(bytes);
    if (transformed == NULL || values == NULL) {
        status = NUMERUS_MATRIX_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (k = 0; k < size; k++) {
        double eigenvalue;
        status = (numerus_matrix_status) numerus_matrix_get(
            eigenvalues, k, 0, &eigenvalue
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            goto cleanup;
        }
        if (!isfinite(eigenvalue)) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
        if (fabs(eigenvalue) > scale) {
            scale = fabs(eigenvalue);
        }
    }
    tolerance = NUMERUS_EPSILON * (double) size * scale;

    for (k = 0; k < size; k++) {
        double eigenvalue;
        status = (numerus_matrix_status) numerus_matrix_get(
            eigenvalues, k, 0, &eigenvalue
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            goto cleanup;
        }

        switch (function) {
            case SYMMETRIC_FUNCTION_SQUARE_ROOT:
                if (eigenvalue < -tolerance) {
                    status = NUMERUS_MATRIX_NOT_POSITIVE_SEMIDEFINITE;
                    goto cleanup;
                }
                if (eigenvalue < 0.0) {
                    eigenvalue = 0.0;
                }
                transformed[k] = sqrt(eigenvalue);
                break;
            case SYMMETRIC_FUNCTION_LOGARITHM:
                if (eigenvalue <= tolerance) {
                    status = NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE;
                    goto cleanup;
                }
                transformed[k] = log(eigenvalue);
                break;
            case SYMMETRIC_FUNCTION_SINE:
                transformed[k] = sin(eigenvalue);
                break;
            case SYMMETRIC_FUNCTION_COSINE:
                transformed[k] = cos(eigenvalue);
                break;
            default:
                status = NUMERUS_MATRIX_INVALID_ARGUMENT;
                goto cleanup;
        }
        if (!isfinite(transformed[k])) {
            status = NUMERUS_MATRIX_NON_FINITE;
            goto cleanup;
        }
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double sum = 0.0;
            for (k = 0; k < size; k++) {
                double left;
                double right;
                status = (numerus_matrix_status) numerus_matrix_get(
                    eigenvectors, row, k, &left
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                status = (numerus_matrix_status) numerus_matrix_get(
                    eigenvectors, column, k, &right
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto cleanup;
                }
                sum += left * transformed[k] * right;
                if (!isfinite(sum)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto cleanup;
                }
            }
            values[row * size + column] = sum;
        }
    }

    status = (numerus_matrix_status) numerus_matrix_create_dense(
        size, size, values, result
    );

cleanup:
    numerus_matrix_destroy(eigenvalues);
    numerus_matrix_destroy(eigenvectors);
    symmetric_function_free(transformed);
    symmetric_function_free(values);
    return status;
}

numerus_matrix_status numerus_matrix_symmetric_square_root(
    const numerus_matrix *matrix,
    numerus_matrix **square_root
)
{
    return symmetric_matrix_function_apply(
        matrix, SYMMETRIC_FUNCTION_SQUARE_ROOT, square_root
    );
}

numerus_matrix_status numerus_matrix_symmetric_logarithm(
    const numerus_matrix *matrix,
    numerus_matrix **logarithm
)
{
    return symmetric_matrix_function_apply(
        matrix, SYMMETRIC_FUNCTION_LOGARITHM, logarithm
    );
}

numerus_matrix_status numerus_matrix_symmetric_sine(
    const numerus_matrix *matrix,
    numerus_matrix **sine
)
{
    return symmetric_matrix_function_apply(
        matrix, SYMMETRIC_FUNCTION_SINE, sine
    );
}

numerus_matrix_status numerus_matrix_symmetric_cosine(
    const numerus_matrix *matrix,
    numerus_matrix **cosine
)
{
    return symmetric_matrix_function_apply(
        matrix, SYMMETRIC_FUNCTION_COSINE, cosine
    );
}
