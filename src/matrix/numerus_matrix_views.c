/**
 * @file numerus_matrix_views.c
 * @brief Lazy Matrix views, coordinate remapping, and view construction.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix_internal.h"
#include "numerus_size.h"

/** Public wrapper around the private checked slice-view constructor. */
int numerus_matrix_create_slice(
    numerus_matrix *parent,
    size_t row_start,
    size_t row_count,
    size_t column_start,
    size_t column_count,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_slice_view(
        parent,
        row_start,
        row_count,
        column_start,
        column_count,
        matrix
    );
}

/** Select rows using an owned copy of the requested indices. */
int numerus_matrix_create_select_rows(
    numerus_matrix *parent,
    const size_t *indices,
    size_t count,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_selection_view(
        parent,
        indices,
        count,
        NUMERUS_MATRIX_SELECTION_ROWS,
        matrix
    );
}

/** Select columns using an owned copy of the requested indices. */
int numerus_matrix_create_select_columns(
    numerus_matrix *parent,
    const size_t *indices,
    size_t count,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_selection_view(
        parent,
        indices,
        count,
        NUMERUS_MATRIX_SELECTION_COLUMNS,
        matrix
    );
}

/** Create a lazy reshape view preserving row-major traversal order. */
int numerus_matrix_create_reshape(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_reshape_view(
        parent, rows, columns, matrix
    );
}

/** Flatten a Matrix into a single row in row-major order. */
int numerus_matrix_create_flatten(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    size_t element_count;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(
            numerus_matrix_rows(parent),
            numerus_matrix_columns(parent),
            &element_count
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    return numerus_matrix_create_reshape_view(
        parent, 1, element_count, matrix
    );
}

/** Create a lazy constant-padded view. */
int numerus_matrix_create_pad(
    numerus_matrix *parent,
    size_t top,
    size_t bottom,
    size_t left,
    size_t right,
    double value,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_padding_view(
        parent, top, bottom, left, right, value, matrix
    );
}

/** Create a lazy zero-padded view. */
int numerus_matrix_create_zero_extend(
    numerus_matrix *parent,
    size_t top,
    size_t bottom,
    size_t left,
    size_t right,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_padding_view(
        parent, top, bottom, left, right, 0.0, matrix
    );
}

/** Tile a Matrix lazily across rows and columns. */
int numerus_matrix_create_repeat(
    numerus_matrix *parent,
    size_t row_repetitions,
    size_t column_repetitions,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_repeat_view(
        parent, row_repetitions, column_repetitions, matrix
    );
}

/** Create a lazy block-diagonal Matrix from repeated copies of parent. */
int numerus_matrix_create_block_diagonal(
    numerus_matrix *parent,
    size_t repetitions,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_block_diagonal_view(
        parent, repetitions, matrix
    );
}

/** Assemble a lazy grid from a row-major array of borrowed Matrix blocks. */
int numerus_matrix_create_block_grid(
    numerus_matrix *const *blocks,
    size_t block_row_count,
    size_t block_column_count,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_block_grid_view(
        blocks, block_row_count, block_column_count, matrix
    );
}

/** Extract a main or offset diagonal as a lazy row-vector view. */
int numerus_matrix_create_diagonal_extract(
    numerus_matrix *parent,
    ptrdiff_t offset,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_diagonal_extract_view(
        parent, offset, matrix
    );
}

/** Create a lazy square diagonal view from a row or column vector. */
int numerus_matrix_create_diagonal_from_vector(
    numerus_matrix *vector,
    ptrdiff_t offset,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_diagonal_from_vector_view(
        vector, offset, matrix
    );
}

/** Permute all rows using a copied full permutation. */
int numerus_matrix_create_permute_rows(
    numerus_matrix *parent,
    const size_t *permutation,
    size_t count,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_permutation_view(
        parent, permutation, count, NUMERUS_MATRIX_SELECTION_ROWS, matrix
    );
}

/** Permute all columns using a copied full permutation. */
int numerus_matrix_create_permute_columns(
    numerus_matrix *parent,
    const size_t *permutation,
    size_t count,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_permutation_view(
        parent, permutation, count, NUMERUS_MATRIX_SELECTION_COLUMNS, matrix
    );
}
