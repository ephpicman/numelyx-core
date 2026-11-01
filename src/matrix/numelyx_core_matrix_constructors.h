/**
 * @file numelyx_core_matrix_constructors.h
 * @brief Matrix storage factories.
 */
#ifndef NUMERUS_MATRIX_CONSTRUCTORS_H
#define NUMERUS_MATRIX_CONSTRUCTORS_H

#include "numelyx_core_matrix_types.h"

/**
 * Create a 1×length row-vector Matrix from copied values.
 *
 * The result is an ordinary dense Matrix, not a separate Vector type.
 * Length must be positive and values must contain at least length elements.
 */
int numelyx_core_matrix_create_row_vector(
    size_t length,
    const double *values,
    numelyx_core_matrix **matrix
);

/**
 * Create a length×1 column-vector Matrix from copied values.
 *
 * The result is an ordinary dense Matrix, not a separate Vector type.
 * Length must be positive and values must contain at least length elements.
 */
int numelyx_core_matrix_create_column_vector(
    size_t length,
    const double *values,
    numelyx_core_matrix **matrix
);

/** Create a dense Matrix. */
int numelyx_core_matrix_create_dense(
    size_t rows,
    size_t columns,
    const double *values,
    numelyx_core_matrix **matrix
);

/** Create an identity Matrix. */
int numelyx_core_matrix_create_identity(size_t size, numelyx_core_matrix **matrix);

/** Create a zero Matrix. */
int numelyx_core_matrix_create_zero(
    size_t rows,
    size_t columns,
    numelyx_core_matrix **matrix
);

/** Create a square zero Matrix. */
int numelyx_core_matrix_create_zero_square(size_t size, numelyx_core_matrix **matrix);

/** Create a diagonal Matrix. */
int numelyx_core_matrix_create_diagonal(
    size_t size,
    const double *values,
    numelyx_core_matrix **matrix
);

/** Create a constant-value Matrix. */
int numelyx_core_matrix_create_constant(
    size_t rows,
    size_t columns,
    double value,
    numelyx_core_matrix **matrix
);

/** Create a scaled-identity Matrix. */
int numelyx_core_matrix_create_scaled_identity(
    size_t size,
    double value,
    numelyx_core_matrix **matrix
);

/** Create an upper-triangular Matrix. */
int numelyx_core_matrix_create_upper_triangular(
    size_t size,
    const double *values,
    numelyx_core_matrix **matrix
);

/** Create a lower-triangular Matrix. */
int numelyx_core_matrix_create_lower_triangular(
    size_t size,
    const double *values,
    numelyx_core_matrix **matrix
);

/** Create a sparse Matrix with a default value and explicit overrides. */
int numelyx_core_matrix_create_sparse(
    size_t rows,
    size_t columns,
    double default_value,
    const numelyx_core_storage_sparse_entry *entries,
    size_t count,
    numelyx_core_matrix **matrix
);

/** Create a symmetric Matrix. */
int numelyx_core_matrix_create_symmetric(
    size_t size,
    const double *values,
    numelyx_core_matrix **matrix
);

/** Create a banded Matrix. */
int numelyx_core_matrix_create_banded(
    size_t rows,
    size_t columns,
    size_t lower_bandwidth,
    size_t upper_bandwidth,
    const double *values,
    numelyx_core_matrix **matrix
);

#endif
