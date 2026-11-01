/**
 * @file numerus_matrix_polynomial.c
 * @brief Matrix polynomial evaluation and polynomial transformations.
 *
 * @details This translation unit provides internal numerical primitives for
 * Matrix operations. Input validation, checked dimension arithmetic, and
 * status propagation are part of the contract and must remain consistent with
 * the declarations in the focused Matrix headers.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"
#include <stdlib.h>
#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define polynomial_alloc(size) emalloc(size)
# define polynomial_free(ptr) efree(ptr)
#else
# define polynomial_alloc(size) malloc(size)
# define polynomial_free(ptr) free(ptr)
#endif

int numerus_matrix_polynomial(const numerus_matrix *base,
    const double *coefficients, size_t coefficient_count,
    numerus_matrix **matrix)
{
    size_t n, count, bytes, i, j, k;
    double *values;
    numerus_matrix *current = NULL;
    int status;

    if (matrix == NULL) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *matrix = NULL;
    if (base == NULL || coefficients == NULL || coefficient_count == 0)
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    n = numerus_matrix_rows(base);
    if (n != numerus_matrix_columns(base)) return NUMERUS_MATRIX_NOT_SQUARE;
    if (n == 0) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    if (!numerus_size_multiply(n, n, &count) ||
        !numerus_size_multiply(count, sizeof(*values), &bytes))
        return NUMERUS_MATRIX_OVERFLOW;

    values = polynomial_alloc(bytes);
    if (values == NULL) return NUMERUS_MATRIX_OUT_OF_MEMORY;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            values[i * n + j] = i == j ? coefficients[coefficient_count - 1] : 0.0;
    status = numerus_matrix_create_dense(n, n, values, &current);
    polynomial_free(values);
    if (status != NUMERUS_MATRIX_SUCCESS) return status;

    for (k = coefficient_count - 1; k > 0; k--) {
        numerus_matrix *product = NULL, *next = NULL;
        status = numerus_matrix_multiply(current, base, &product);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            numerus_matrix_destroy(current);
            return status;
        }
        values = polynomial_alloc(bytes);
        if (values == NULL) {
            numerus_matrix_destroy(product);
            numerus_matrix_destroy(current);
            return NUMERUS_MATRIX_OUT_OF_MEMORY;
        }
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double value;
                status = numerus_matrix_get(product, i, j, &value);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    polynomial_free(values);
                    numerus_matrix_destroy(product);
                    numerus_matrix_destroy(current);
                    return status;
                }
                values[i * n + j] = value + (i == j ? coefficients[k - 1] : 0.0);
            }
        }
        status = numerus_matrix_create_dense(n, n, values, &next);
        polynomial_free(values);
        numerus_matrix_destroy(product);
        numerus_matrix_destroy(current);
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        current = next;
    }
    *matrix = current;
    return NUMERUS_MATRIX_SUCCESS;
}
