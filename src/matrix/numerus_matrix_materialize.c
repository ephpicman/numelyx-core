/**
 * @file numerus_matrix_materialize.c
 * @brief Materialization of lazy Matrix expressions into owned storage.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numerus_matrix.h"
#include "numerus_size.h"

#include <stdlib.h>

#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_materialize_alloc(size) emalloc(size)
# define numerus_materialize_free(ptr) efree(ptr)
#else
# define numerus_materialize_alloc(size) malloc(size)
# define numerus_materialize_free(ptr) free(ptr)
#endif

/**
 * Copy the logical values of any Matrix node into independent dense Storage.
 *
 * Reads go through the checked accessor so nested views, joins, and callback
 * failures are handled uniformly. No result is published until all reads and
 * the dense Storage construction succeed.
 */
int numerus_matrix_materialize(
    const numerus_matrix *source,
    numerus_matrix **matrix
)
{
    size_t rows;
    size_t columns;
    size_t element_count;
    size_t allocation_size;
    size_t row;
    size_t column;
    double *values;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (source == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(source);
    columns = numerus_matrix_columns(source);
    if (rows == 0 || columns == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(rows, columns, &element_count) ||
        !numerus_size_multiply(
            element_count, sizeof(*values), &allocation_size
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = numerus_materialize_alloc(allocation_size);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            status = numerus_matrix_get(
                source,
                row,
                column,
                &values[row * columns + column]
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numerus_materialize_free(values);
                return status;
            }
        }
    }

    status = numerus_matrix_create_dense(rows, columns, values, matrix);
    numerus_materialize_free(values);

    return status;
}
