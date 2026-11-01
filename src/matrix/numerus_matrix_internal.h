/**
 * @file numerus_matrix_internal.h
 * @brief Private helpers shared by Matrix implementation units.
 */
#ifndef NUMERUS_MATRIX_INTERNAL_H
#define NUMERUS_MATRIX_INTERNAL_H

#include "numerus_matrix.h"
#include "numerus_matrix_lu.h"

typedef enum {
    NUMERUS_MATRIX_SELECTION_NONE = 0,
    NUMERUS_MATRIX_SELECTION_ROWS,
    NUMERUS_MATRIX_SELECTION_COLUMNS
} numerus_matrix_selection_axis;

typedef enum {
    NUMERUS_MATRIX_BINARY_NONE = 0,
    NUMERUS_MATRIX_BINARY_ADD,
    NUMERUS_MATRIX_BINARY_SUBTRACT,
    NUMERUS_MATRIX_BINARY_HADAMARD,
    NUMERUS_MATRIX_BINARY_DIVIDE
} numerus_matrix_binary_operation;

/** Allocate and initialize an internal Matrix node. */
int numerus_matrix_allocate(numerus_matrix **matrix);

/** Obtain a retained LU factorization, computing and caching it on a miss. */
numerus_matrix_status numerus_matrix_get_or_factorize_lu(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization **factorization
);

/** Internal LU-cache lookup; a successful miss returns *factorization == NULL. */
numerus_matrix_status numerus_matrix_get_cached_lu(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization **factorization
);

/** Retain an eligible LU factorization in the source Matrix, if possible. */
void numerus_matrix_store_lu_cache(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization *factorization
);

/** Internal inverse-cache lookup; a successful miss returns *inverse == NULL. */
numerus_matrix_status numerus_matrix_get_cached_inverse(
    const numerus_matrix *matrix,
    numerus_matrix **inverse
);

/** Copy inverse values into the source Matrix's cache when within its size cap. */
void numerus_matrix_store_inverse_cache(
    const numerus_matrix *matrix,
    const double *values
);

/**
 * Create a lazy two-parent element-wise node.
 *
 * This is an implementation helper, not part of the supported Matrix API.
 */
int numerus_matrix_create_binary_view(
    numerus_matrix *left,
    numerus_matrix *right,
    numerus_matrix_binary_operation operation,
    numerus_matrix **matrix
);

/** Private checked constructor for a row-major reshape view. */
int numerus_matrix_create_reshape_view(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
);

/** Private checked constructor for a lazy index-selection view. */
int numerus_matrix_create_selection_view(
    numerus_matrix *parent,
    const size_t *indices,
    size_t count,
    numerus_matrix_selection_axis axis,
    numerus_matrix **matrix
);

/** Private checked constructor for a constant-padded view. */
int numerus_matrix_create_padding_view(
    numerus_matrix *parent,
    size_t top,
    size_t bottom,
    size_t left,
    size_t right,
    double value,
    numerus_matrix **matrix
);

/** Private checked constructor for a tiled repetition view. */
int numerus_matrix_create_repeat_view(
    numerus_matrix *parent,
    size_t row_repetitions,
    size_t column_repetitions,
    numerus_matrix **matrix
);

/** Private checked constructor for a repeated block-diagonal view. */
int numerus_matrix_create_block_diagonal_view(
    numerus_matrix *parent,
    size_t repetitions,
    numerus_matrix **matrix
);

/** Private checked constructor for a general non-empty block grid. */
int numerus_matrix_create_block_grid_view(
    numerus_matrix *const *blocks,
    size_t block_row_count,
    size_t block_column_count,
    numerus_matrix **matrix
);

/** Private checked constructor for a full row/column permutation view. */
int numerus_matrix_create_permutation_view(
    numerus_matrix *parent,
    const size_t *permutation,
    size_t count,
    numerus_matrix_selection_axis axis,
    numerus_matrix **matrix
);

/** Private checked constructor for a row-vector diagonal extraction view. */
int numerus_matrix_create_diagonal_extract_view(
    numerus_matrix *parent,
    ptrdiff_t offset,
    numerus_matrix **matrix
);

/** Private checked constructor for a square diagonal view from a vector. */
int numerus_matrix_create_diagonal_from_vector_view(
    numerus_matrix *vector,
    ptrdiff_t offset,
    numerus_matrix **matrix
);

/** Private checked constructor for a lazy range slice. */
int numerus_matrix_create_slice_view(
    numerus_matrix *parent,
    size_t row_start,
    size_t row_count,
    size_t column_start,
    size_t column_count,
    numerus_matrix **matrix
);


#ifndef NUMERUS_MATRIX_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_matrix_alloc(size) emalloc(size)
# define numerus_matrix_free(ptr) efree(ptr)
#else
# include <stdlib.h>
# define numerus_matrix_alloc(size) malloc(size)
# define numerus_matrix_free(ptr) free(ptr)
#endif

struct numerus_matrix {
    size_t rows;
    size_t columns;
    numerus_storage *storage;
    numerus_matrix *parent;
    numerus_matrix *parent2;
    numerus_matrix_join_type join_type;
    numerus_matrix_binary_operation binary_operation;
    numerus_coordinate_transform_fn coordinate_transform;
    numerus_value_transform_fn value_transform;
    const void *transform_context;
    size_t transform_index1;
    size_t transform_index2;
    numerus_matrix_flags cached_flags;
    uint32_t flags_computed;
    int determinant_state;
    double cached_determinant;
    double *cached_inverse_values;
    numerus_matrix_lu_factorization *cached_lu_factorization;
    double transform_scalar;
    size_t transform_row_offset;
    size_t transform_column_offset;
    size_t transform_parent_columns;
    size_t *selection_indices;
    numerus_matrix_selection_axis selection_axis;
    bool padding_view;
    size_t padding_top;
    size_t padding_left;
    double padding_value;
    bool repeat_view;
    bool block_diagonal_view;
    bool block_grid_view;
    numerus_matrix **block_matrices;
    size_t *block_row_offsets;
    size_t *block_column_offsets;
    size_t block_row_count;
    size_t block_column_count;
    bool diagonal_matrix_view;
    bool diagonal_offset_positive;
    size_t diagonal_offset_magnitude;
    size_t diagonal_vector_length;
};

#endif
