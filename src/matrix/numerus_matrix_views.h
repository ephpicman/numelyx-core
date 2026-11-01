/**
 * @file numerus_matrix_views.h
 * @brief Lazy Matrix views and composition.
 */
#ifndef NUMERUS_MATRIX_VIEWS_H
#define NUMERUS_MATRIX_VIEWS_H

#include "numerus_matrix_types.h"

/**
 * @brief Create a lazy transpose view of an existing Matrix.
 *
 * The returned child exposes parent columns as rows and parent rows as
 * columns. No element data is copied. The parent is non-owning and must
 * remain alive for the lifetime of the transpose view.
 */
int numerus_matrix_create_transpose(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/** Create a lazy view with the order of rows reversed. */
int numerus_matrix_create_flip_rows(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/** Create a lazy view with the order of columns reversed. */
int numerus_matrix_create_flip_columns(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/** Create a lazy view that omits one row. */
int numerus_matrix_create_remove_row(
    numerus_matrix *parent,
    size_t row,
    numerus_matrix **matrix
);

/** Create a lazy view that omits one column. */
int numerus_matrix_create_remove_column(
    numerus_matrix *parent,
    size_t column,
    numerus_matrix **matrix
);

/** Create a lazy view that swaps two rows. */
int numerus_matrix_create_swap_rows(
    numerus_matrix *parent,
    size_t row1,
    size_t row2,
    numerus_matrix **matrix
);

/** Create a lazy view that swaps two columns. */
int numerus_matrix_create_swap_columns(
    numerus_matrix *parent,
    size_t column1,
    size_t column2,
    numerus_matrix **matrix
);

/** Create a lazy 90-degree clockwise rotation view. */
int numerus_matrix_create_rotate_90_clockwise(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/** Create a lazy 180-degree rotation view. */
int numerus_matrix_create_rotate_180(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/** Create a lazy 90-degree counter-clockwise rotation view. */
int numerus_matrix_create_rotate_90_counterclockwise(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/**
 * @brief Create a lazy horizontal join of two Matrices.
 *
 * Both parents must have the same number of rows. The second Matrix is
 * appended to the right of the first. Neither parent is owned by the result.
 */
int numerus_matrix_create_join_horizontal(
    numerus_matrix *parent,
    numerus_matrix *parent2,
    numerus_matrix **matrix
);

/**
 * @brief Create a lazy vertical join of two Matrices.
 *
 * Both parents must have the same number of columns. The second Matrix is
 * appended below the first. Neither parent is owned by the result.
 */
int numerus_matrix_create_join_vertical(
    numerus_matrix *parent,
    numerus_matrix *parent2,
    numerus_matrix **matrix
);

/**
 * Create a lazy range slice of a Matrix.
 *
 * The slice contains row_count × column_count values starting at the specified
 * zero-based parent coordinates. Counts must be positive and the complete
 * range must fit inside the parent. The parent is borrowed and must outlive
 * the result.
 */
int numerus_matrix_create_slice(
    numerus_matrix *parent,
    size_t row_start,
    size_t row_count,
    size_t column_start,
    size_t column_count,
    numerus_matrix **matrix
);

/**
 * Select rows by a copied array of zero-based indices.
 *
 * Index order is preserved and repeated indices are allowed. The result is a
 * lazy view; the index array is copied, while the parent remains borrowed.
 */
int numerus_matrix_create_select_rows(
    numerus_matrix *parent,
    const size_t *indices,
    size_t count,
    numerus_matrix **matrix
);

/**
 * Select columns by a copied array of zero-based indices.
 *
 * Index order is preserved and repeated indices are allowed. The result is a
 * lazy view; the index array is copied, while the parent remains borrowed.
 */
int numerus_matrix_create_select_columns(
    numerus_matrix *parent,
    const size_t *indices,
    size_t count,
    numerus_matrix **matrix
);

/**
 * Create a lazy row-major reshape view.
 *
 * The requested shape must contain exactly the same number of elements as
 * the parent. Logical traversal order is row-major. The parent is borrowed
 * and must outlive the view.
 */
int numerus_matrix_create_reshape(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
);

/**
 * Create a lazy row-vector view of all parent elements in row-major order.
 *
 * The result has shape 1 × (parent.rows * parent.columns). Element-count
 * overflow is reported as NUMERUS_MATRIX_OVERFLOW.
 */
int numerus_matrix_create_flatten(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/**
 * Create a lazy constant-padded view.
 *
 * top/bottom/left/right specify the number of rows or columns added on each
 * side. Existing values retain their positions; all padding cells use value.
 * The parent is borrowed and must outlive the view. Dimension overflow is
 * reported as NUMERUS_MATRIX_OVERFLOW.
 */
int numerus_matrix_create_pad(
    numerus_matrix *parent,
    size_t top,
    size_t bottom,
    size_t left,
    size_t right,
    double value,
    numerus_matrix **matrix
);

/** Create a lazy zero-padded view with the same shape and ownership rules. */
int numerus_matrix_create_zero_extend(
    numerus_matrix *parent,
    size_t top,
    size_t bottom,
    size_t left,
    size_t right,
    numerus_matrix **matrix
);

/**
 * Tile a Matrix lazily row_repetitions × column_repetitions times.
 *
 * Repetition counts must be positive. The parent is borrowed and must outlive
 * the view; output dimensions are checked for overflow.
 */
int numerus_matrix_create_repeat(
    numerus_matrix *parent,
    size_t row_repetitions,
    size_t column_repetitions,
    numerus_matrix **matrix
);

/**
 * Create a lazy block-diagonal Matrix containing repetitions copies of parent.
 *
 * Off-diagonal blocks are zero. The parent may be rectangular; result
 * dimensions are (parent.rows * repetitions) ×
 * (parent.columns * repetitions). Repetitions must be positive, and the
 * borrowed parent must outlive the view.
 */
int numerus_matrix_create_block_diagonal(
    numerus_matrix *parent,
    size_t repetitions,
    numerus_matrix **matrix
);

/**
 * Assemble a lazy block grid from a row-major array of Matrix pointers.
 *
 * Every block must be non-NULL. Blocks in each block row must have equal row
 * counts, and blocks in each block column must have equal column counts.
 * Missing/NULL blocks are rejected rather than implicitly treated as zeros.
 * The pointer array is copied; all block Matrices are borrowed and must
 * outlive the resulting view. Grid dimensions and cumulative output sizes are
 * checked for overflow.
 */
int numerus_matrix_create_block_grid(
    numerus_matrix *const *blocks,
    size_t block_row_count,
    size_t block_column_count,
    numerus_matrix **matrix
);

/**
 * Extract a diagonal into a 1×N row-vector view.
 *
 * Offset zero selects the main diagonal; positive offsets select diagonals
 * above it, and negative offsets select diagonals below it. An offset with no
 * elements in the Matrix returns NUMERUS_MATRIX_OUT_OF_BOUNDS. The parent is
 * borrowed and must outlive the result.
 */
int numerus_matrix_create_diagonal_extract(
    numerus_matrix *parent,
    ptrdiff_t offset,
    numerus_matrix **matrix
);

/**
 * Create a square diagonal view from a row or column vector.
 *
 * The main diagonal is selected by offset zero; positive offsets place values
 * above it and negative offsets below it. The output side length is the vector
 * length plus the absolute offset, checked for overflow. Off-diagonal elements
 * are zero. The vector is borrowed and must outlive the result.
 */
int numerus_matrix_create_diagonal_from_vector(
    numerus_matrix *vector,
    ptrdiff_t offset,
    numerus_matrix **matrix
);

/**
 * Permute all rows using a full zero-based permutation.
 *
 * The index count must equal the parent row count; every index must be in
 * range and appear exactly once. The index list is copied and the parent is
 * borrowed.
 */
int numerus_matrix_create_permute_rows(
    numerus_matrix *parent,
    const size_t *permutation,
    size_t count,
    numerus_matrix **matrix
);

/**
 * Permute all columns using a full zero-based permutation.
 *
 * The index count must equal the parent column count; every index must be in
 * range and appear exactly once. The index list is copied and the parent is
 * borrowed.
 */
int numerus_matrix_create_permute_columns(
    numerus_matrix *parent,
    const size_t *permutation,
    size_t count,
    numerus_matrix **matrix
);

#endif
