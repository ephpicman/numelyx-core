/**
 * @file numerus_matrix_view_nodes.c
 * @brief Lazy view node representations and coordinate translation.
 *
 * @details This translation unit implements an internal Matrix algorithm.
 * Its callers rely on consistent dimension checks, explicit status returns,
 * and cleanup of temporary allocations on every exit path.
 */

#include "numerus_matrix_internal.h"
#include "numerus_size.h"
#include "numerus_numeric.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>





/*
 * Coordinate callbacks map a logical view index back to its parent without
 * materializing data. Constructors validate the view dimensions and ranges;
 * the callbacks therefore perform only the relevant coordinate remapping and
 * still validate their context/output pointers at the internal API boundary.
 */
static numerus_matrix_status slice_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    if (view == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = view->transform_row_offset + row;
    *parent_column = view->transform_column_offset + column;
    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status diagonal_extract_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    (void) row;

    if (view == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = view->transform_row_offset + column;
    *parent_column = view->transform_column_offset + column;
    return NUMERUS_MATRIX_SUCCESS;
}

/* Selection views store a private copy of their index list. This callback
 * translates only the selected axis; the other coordinate passes through.
 * Repeated indices intentionally remain valid for selection views.
 */
static numerus_matrix_status selection_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    if (view == NULL || view->selection_indices == NULL ||
        parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (view->selection_axis == NUMERUS_MATRIX_SELECTION_ROWS) {
        *parent_row = view->selection_indices[row];
        *parent_column = column;
        return NUMERUS_MATRIX_SUCCESS;
    }

    if (view->selection_axis == NUMERUS_MATRIX_SELECTION_COLUMNS) {
        *parent_row = row;
        *parent_column = view->selection_indices[column];
        return NUMERUS_MATRIX_SUCCESS;
    }

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

/* Reshape preserves row-major linear order rather than the parent's 2-D
 * coordinates. The inverse mapping uses division/remainder by the validated
 * parent column count; construction rejects zero-width parents for this path.
 */
static numerus_matrix_status reshape_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;
    size_t linear_index;

    if (view == NULL || parent_row == NULL || parent_column == NULL ||
        view->transform_parent_columns == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    /*
     * Construction verifies rows * columns is representable. Coordinates
     * reaching this callback are within those dimensions, so this linear
     * index is bounded by the validated logical element count.
     */
    linear_index = row * view->columns + column;
    *parent_row = linear_index / view->transform_parent_columns;
    *parent_column = linear_index % view->transform_parent_columns;

    return NUMERUS_MATRIX_SUCCESS;
}

/* The following callbacks implement pure index maps for common views. They
 * must not read or write element storage: numerus_matrix_get() composes the
 * mapping with the parent accessor, preserving laziness across nested views.
 */
static numerus_matrix_status transpose_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) context;

    if (parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = column;
    *parent_column = row;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status flip_rows_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *parent = context;

    if (parent == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = numerus_matrix_rows(parent) - 1 - row;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status flip_columns_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *parent = context;

    if (parent == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = numerus_matrix_columns(parent) - 1 - column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status remove_row_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    if (view == NULL || view->parent == NULL ||
        parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row >= view->transform_index1 ? row + 1 : row;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status remove_column_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    if (view == NULL || view->parent == NULL ||
        parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column >= view->transform_index1 ? column + 1 : column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status swap_rows_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    if (view == NULL || view->parent == NULL ||
        parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (row == view->transform_index1) {
        row = view->transform_index2;
    } else if (row == view->transform_index2) {
        row = view->transform_index1;
    }

    *parent_row = row;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status swap_columns_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *view = context;

    if (view == NULL || view->parent == NULL ||
        parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (column == view->transform_index1) {
        column = view->transform_index2;
    } else if (column == view->transform_index2) {
        column = view->transform_index1;
    }

    *parent_row = row;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status rotate_90_clockwise_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *parent = context;

    if (parent == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = numerus_matrix_rows(parent) - 1 - column;
    *parent_column = row;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status rotate_180_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *parent = context;

    if (parent == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = numerus_matrix_rows(parent) - 1 - row;
    *parent_column = numerus_matrix_columns(parent) - 1 - column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status rotate_90_counterclockwise_coordinate_transform(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    const numerus_matrix *parent = context;

    if (parent == NULL || parent_row == NULL || parent_column == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = column;
    *parent_column = numerus_matrix_columns(parent) - 1 - row;

    return NUMERUS_MATRIX_SUCCESS;
}

/*
 * Shared constructor for row/column selection views. It validates indices,
 * copies them into view-owned memory (so the caller's array may be released),
 * and stores the parent as a borrowed pointer. Destruction frees the copied
 * indices but never destroys the parent.
 */
int numerus_matrix_create_selection_view(
    numerus_matrix *parent,
    const size_t *indices,
    size_t count,
    numerus_matrix_selection_axis axis,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t *owned_indices;
    size_t index_bytes;
    size_t parent_rows;
    size_t parent_columns;
    size_t index;
    size_t selected_limit;
    size_t view_rows;
    size_t view_columns;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || indices == NULL || count == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (axis != NUMERUS_MATRIX_SELECTION_ROWS &&
        axis != NUMERUS_MATRIX_SELECTION_COLUMNS) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (!numerus_size_multiply(count, sizeof(*owned_indices), &index_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    parent_rows = numerus_matrix_rows(parent);
    parent_columns = numerus_matrix_columns(parent);
    selected_limit = axis == NUMERUS_MATRIX_SELECTION_ROWS
        ? parent_rows
        : parent_columns;

    for (index = 0; index < count; index++) {
        if (indices[index] >= selected_limit) {
            return NUMERUS_MATRIX_OUT_OF_BOUNDS;
        }
    }

    owned_indices = numerus_matrix_alloc(index_bytes);
    if (owned_indices == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    for (index = 0; index < count; index++) {
        owned_indices[index] = indices[index];
    }

    view_rows = axis == NUMERUS_MATRIX_SELECTION_ROWS
        ? count
        : parent_rows;
    view_columns = axis == NUMERUS_MATRIX_SELECTION_COLUMNS
        ? count
        : parent_columns;

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        view_rows,
        view_columns,
        selection_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_free(owned_indices);
        return status;
    }

    view->selection_indices = owned_indices;
    view->selection_axis = axis;
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_permutation_view(
    numerus_matrix *parent,
    const size_t *permutation,
    size_t count,
    numerus_matrix_selection_axis axis,
    numerus_matrix **matrix
)
{
    bool *seen;
    size_t seen_bytes;
    size_t expected_count;
    size_t index;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || permutation == NULL || count == 0 ||
        (axis != NUMERUS_MATRIX_SELECTION_ROWS &&
         axis != NUMERUS_MATRIX_SELECTION_COLUMNS)) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    expected_count = axis == NUMERUS_MATRIX_SELECTION_ROWS
        ? numerus_matrix_rows(parent)
        : numerus_matrix_columns(parent);
    if (count != expected_count) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    if (!numerus_size_multiply(count, sizeof(*seen), &seen_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    seen = numerus_matrix_alloc(seen_bytes);
    if (seen == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    for (index = 0; index < count; index++) {
        seen[index] = false;
    }

    for (index = 0; index < count; index++) {
        size_t selected = permutation[index];

        if (selected >= count) {
            numerus_matrix_free(seen);
            return NUMERUS_MATRIX_OUT_OF_BOUNDS;
        }
        if (seen[selected]) {
            numerus_matrix_free(seen);
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }
        seen[selected] = true;
    }

    numerus_matrix_free(seen);
    status = numerus_matrix_create_selection_view(
        parent, permutation, count, axis, matrix
    );
    return status;
}

int numerus_matrix_create_reshape_view(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t parent_count;
    size_t view_count;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || rows == 0 || columns == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(
            numerus_matrix_rows(parent),
            numerus_matrix_columns(parent),
            &parent_count
        ) ||
        !numerus_size_multiply(rows, columns, &view_count)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    if (parent_count != view_count) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        rows,
        columns,
        reshape_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_parent_columns = numerus_matrix_columns(parent);
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_padding_view(
    numerus_matrix *parent,
    size_t top,
    size_t bottom,
    size_t left,
    size_t right,
    double value,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t rows_with_top;
    size_t rows;
    size_t columns_with_left;
    size_t columns;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_add(numerus_matrix_rows(parent), top, &rows_with_top) ||
        !numerus_size_add(rows_with_top, bottom, &rows) ||
        !numerus_size_add(numerus_matrix_columns(parent), left, &columns_with_left) ||
        !numerus_size_add(columns_with_left, right, &columns)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent, rows, columns, NULL, NULL, NULL, &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->padding_view = true;
    view->padding_top = top;
    view->padding_left = left;
    view->padding_value = value;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_repeat_view(
    numerus_matrix *parent,
    size_t row_repetitions,
    size_t column_repetitions,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t rows;
    size_t columns;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || row_repetitions == 0 || column_repetitions == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(
            numerus_matrix_rows(parent), row_repetitions, &rows
        ) ||
        !numerus_size_multiply(
            numerus_matrix_columns(parent), column_repetitions, &columns
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent, rows, columns, NULL, NULL, NULL, &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->repeat_view = true;
    *matrix = view;
    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_block_diagonal_view(
    numerus_matrix *parent,
    size_t repetitions,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t rows;
    size_t columns;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || repetitions == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(
            numerus_matrix_rows(parent), repetitions, &rows
        ) ||
        !numerus_size_multiply(
            numerus_matrix_columns(parent), repetitions, &columns
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent, rows, columns, NULL, NULL, NULL, &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->block_diagonal_view = true;
    *matrix = view;
    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_block_grid_view(
    numerus_matrix *const *blocks,
    size_t block_row_count,
    size_t block_column_count,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t block_count;
    size_t block_bytes;
    size_t row_offset_count;
    size_t row_offset_bytes;
    size_t column_offset_count;
    size_t column_offset_bytes;
    size_t total_rows = 0;
    size_t total_columns = 0;
    size_t row;
    size_t column;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (blocks == NULL || block_row_count == 0 || block_column_count == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (!numerus_size_multiply(
            block_row_count, block_column_count, &block_count
        ) ||
        !numerus_size_multiply(
            block_count, sizeof(*view->block_matrices), &block_bytes
        ) ||
        !numerus_size_add(block_row_count, 1, &row_offset_count) ||
        !numerus_size_multiply(
            row_offset_count, sizeof(*view->block_row_offsets), &row_offset_bytes
        ) ||
        !numerus_size_add(block_column_count, 1, &column_offset_count) ||
        !numerus_size_multiply(
            column_offset_count, sizeof(*view->block_column_offsets), &column_offset_bytes
        )) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    for (row = 0; row < block_row_count; row++) {
        size_t block_height;
        size_t next_total;

        if (blocks[row * block_column_count] == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        block_height = numerus_matrix_rows(blocks[row * block_column_count]);
        for (column = 1; column < block_column_count; column++) {
            numerus_matrix *block = blocks[row * block_column_count + column];

            if (block == NULL) {
                return NUMERUS_MATRIX_INVALID_ARGUMENT;
            }
            if (numerus_matrix_rows(block) != block_height) {
                return NUMERUS_MATRIX_DIMENSION_MISMATCH;
            }
        }

        if (!numerus_size_add(total_rows, block_height, &next_total)) {
            return NUMERUS_MATRIX_OVERFLOW;
        }
        total_rows = next_total;
    }

    for (column = 0; column < block_column_count; column++) {
        size_t block_width;
        size_t next_total;

        if (blocks[column] == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        block_width = numerus_matrix_columns(blocks[column]);
        for (row = 1; row < block_row_count; row++) {
            numerus_matrix *block = blocks[row * block_column_count + column];

            if (block == NULL) {
                return NUMERUS_MATRIX_INVALID_ARGUMENT;
            }
            if (numerus_matrix_columns(block) != block_width) {
                return NUMERUS_MATRIX_DIMENSION_MISMATCH;
            }
        }

        if (!numerus_size_add(total_columns, block_width, &next_total)) {
            return NUMERUS_MATRIX_OVERFLOW;
        }
        total_columns = next_total;
    }

    status = numerus_matrix_allocate(&view);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->block_matrices = numerus_matrix_alloc(block_bytes);
    view->block_row_offsets = numerus_matrix_alloc(row_offset_bytes);
    view->block_column_offsets = numerus_matrix_alloc(column_offset_bytes);

    if (view->block_matrices == NULL || view->block_row_offsets == NULL ||
        view->block_column_offsets == NULL) {
        numerus_matrix_free(view->block_matrices);
        numerus_matrix_free(view->block_row_offsets);
        numerus_matrix_free(view->block_column_offsets);
        numerus_matrix_free(view);
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    for (row = 0; row < block_count; row++) {
        view->block_matrices[row] = blocks[row];
    }

    view->block_row_offsets[0] = 0;
    for (row = 0; row < block_row_count; row++) {
        view->block_row_offsets[row + 1] =
            view->block_row_offsets[row] +
            numerus_matrix_rows(blocks[row * block_column_count]);
    }

    view->block_column_offsets[0] = 0;
    for (column = 0; column < block_column_count; column++) {
        view->block_column_offsets[column + 1] =
            view->block_column_offsets[column] +
            numerus_matrix_columns(blocks[column]);
    }

    view->rows = total_rows;
    view->columns = total_columns;
    view->block_grid_view = true;
    view->block_row_count = block_row_count;
    view->block_column_count = block_column_count;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_diagonal_extract_view(
    numerus_matrix *parent,
    ptrdiff_t offset,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t parent_rows;
    size_t parent_columns;
    size_t row_offset = 0;
    size_t column_offset = 0;
    size_t length;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    parent_rows = numerus_matrix_rows(parent);
    parent_columns = numerus_matrix_columns(parent);

    if (offset >= 0) {
        column_offset = (size_t) offset;
        if (column_offset >= parent_columns) {
            return NUMERUS_MATRIX_OUT_OF_BOUNDS;
        }
        length = parent_columns - column_offset;
        if (length > parent_rows) {
            length = parent_rows;
        }
    } else {
        row_offset = (size_t) (-(offset + 1)) + 1;
        if (row_offset >= parent_rows) {
            return NUMERUS_MATRIX_OUT_OF_BOUNDS;
        }
        length = parent_rows - row_offset;
        if (length > parent_columns) {
            length = parent_columns;
        }
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent, 1, length, diagonal_extract_coordinate_transform,
        NULL, NULL, &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_row_offset = row_offset;
    view->transform_column_offset = column_offset;
    view->transform_context = view;
    *matrix = view;
    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_diagonal_from_vector_view(
    numerus_matrix *vector,
    ptrdiff_t offset,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t vector_length;
    size_t offset_magnitude;
    size_t size;
    size_t element_count;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (vector == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (numerus_matrix_rows(vector) == 1) {
        vector_length = numerus_matrix_columns(vector);
    } else if (numerus_matrix_columns(vector) == 1) {
        vector_length = numerus_matrix_rows(vector);
    } else {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    offset_magnitude = offset >= 0
        ? (size_t) offset
        : (size_t) (-(offset + 1)) + 1;

    if (!numerus_size_add(vector_length, offset_magnitude, &size) ||
        !numerus_size_multiply(size, size, &element_count)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        vector, size, size, NULL, NULL, NULL, &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->diagonal_matrix_view = true;
    view->diagonal_offset_positive = offset >= 0;
    view->diagonal_offset_magnitude = offset_magnitude;
    view->diagonal_vector_length = vector_length;
    *matrix = view;
    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_slice_view(
    numerus_matrix *parent,
    size_t row_start,
    size_t row_count,
    size_t column_start,
    size_t column_count,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    size_t parent_rows;
    size_t parent_columns;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || row_count == 0 || column_count == 0) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    parent_rows = numerus_matrix_rows(parent);
    parent_columns = numerus_matrix_columns(parent);

    if (row_start > parent_rows || column_start > parent_columns) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }
    if (row_count > parent_rows - row_start ||
        column_count > parent_columns - column_start) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        row_count,
        column_count,
        slice_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_row_offset = row_start;
    view->transform_column_offset = column_start;
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_transpose(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    if (parent == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_columns(parent),
        numerus_matrix_rows(parent),
        transpose_coordinate_transform,
        NULL,
        NULL,
        matrix
    );
}

int numerus_matrix_create_flip_rows(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    if (parent == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
        flip_rows_coordinate_transform,
        NULL,
        parent,
        matrix
    );
}

int numerus_matrix_create_flip_columns(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    if (parent == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
        flip_columns_coordinate_transform,
        NULL,
        parent,
        matrix
    );
}

int numerus_matrix_create_remove_row(
    numerus_matrix *parent,
    size_t row,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (row >= numerus_matrix_rows(parent)) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }
    if (numerus_matrix_rows(parent) == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent) - 1,
        numerus_matrix_columns(parent),
        remove_row_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_index1 = row;
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_remove_column(
    numerus_matrix *parent,
    size_t column,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (column >= numerus_matrix_columns(parent)) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }
    if (numerus_matrix_columns(parent) == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent) - 1,
        remove_column_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_index1 = column;
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_swap_rows(
    numerus_matrix *parent,
    size_t row1,
    size_t row2,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (row1 >= numerus_matrix_rows(parent) ||
        row2 >= numerus_matrix_rows(parent)) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
        swap_rows_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_index1 = row1;
    view->transform_index2 = row2;
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_swap_columns(
    numerus_matrix *parent,
    size_t column1,
    size_t column2,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (column1 >= numerus_matrix_columns(parent) ||
        column2 >= numerus_matrix_columns(parent)) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
        swap_columns_coordinate_transform,
        NULL,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_index1 = column1;
    view->transform_index2 = column2;
    view->transform_context = view;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_rotate_90_clockwise(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    if (parent == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_columns(parent),
        numerus_matrix_rows(parent),
        rotate_90_clockwise_coordinate_transform,
        NULL,
        parent,
        matrix
    );
}

int numerus_matrix_create_rotate_180(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    if (parent == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
        rotate_180_coordinate_transform,
        NULL,
        parent,
        matrix
    );
}

int numerus_matrix_create_rotate_90_counterclockwise(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    if (parent == NULL || matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_columns(parent),
        numerus_matrix_rows(parent),
        rotate_90_counterclockwise_coordinate_transform,
        NULL,
        parent,
        matrix
    );
}

int numerus_matrix_create_binary_view(
    numerus_matrix *left,
    numerus_matrix *right,
    numerus_matrix_binary_operation operation,
    numerus_matrix **matrix
)
{
    numerus_matrix *result = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (left == NULL || right == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (operation != NUMERUS_MATRIX_BINARY_ADD &&
        operation != NUMERUS_MATRIX_BINARY_SUBTRACT &&
        operation != NUMERUS_MATRIX_BINARY_HADAMARD &&
        operation != NUMERUS_MATRIX_BINARY_DIVIDE) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(left) != numerus_matrix_rows(right) ||
        numerus_matrix_columns(left) != numerus_matrix_columns(right)) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }

    status = numerus_matrix_allocate(&result);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    result->rows = numerus_matrix_rows(left);
    result->columns = numerus_matrix_columns(left);
    result->storage = NULL;
    result->parent = left;
    result->parent2 = right;
    result->binary_operation = operation;
    result->coordinate_transform = NULL;
    result->value_transform = NULL;
    result->transform_context = NULL;
    *matrix = result;

    return NUMERUS_MATRIX_SUCCESS;
}

static int numerus_matrix_create_join(
    numerus_matrix *parent,
    numerus_matrix *parent2,
    numerus_matrix_join_type join_type,
    numerus_matrix **matrix
)
{
    numerus_matrix *result = NULL;
    size_t rows;
    size_t columns;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL || parent2 == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (join_type == NUMERUS_MATRIX_JOIN_HORIZONTAL) {
        if (numerus_matrix_rows(parent) != numerus_matrix_rows(parent2)) {
            return NUMERUS_MATRIX_DIMENSION_MISMATCH;
        }

        rows = numerus_matrix_rows(parent);
        if (numerus_matrix_columns(parent) > SIZE_MAX - numerus_matrix_columns(parent2)) {
            return NUMERUS_MATRIX_OVERFLOW;
        }
        columns = numerus_matrix_columns(parent) + numerus_matrix_columns(parent2);
    } else if (join_type == NUMERUS_MATRIX_JOIN_VERTICAL) {
        if (numerus_matrix_columns(parent) != numerus_matrix_columns(parent2)) {
            return NUMERUS_MATRIX_DIMENSION_MISMATCH;
        }

        columns = numerus_matrix_columns(parent);
        if (numerus_matrix_rows(parent) > SIZE_MAX - numerus_matrix_rows(parent2)) {
            return NUMERUS_MATRIX_OVERFLOW;
        }
        rows = numerus_matrix_rows(parent) + numerus_matrix_rows(parent2);
    } else {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_allocate(&result);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    result->rows = rows;
    result->columns = columns;
    result->storage = NULL;
    result->parent = parent;
    result->parent2 = parent2;
    result->join_type = join_type;
    result->binary_operation = NUMERUS_MATRIX_BINARY_NONE;
    result->coordinate_transform = NULL;
    result->value_transform = NULL;
    result->transform_context = NULL;
    *matrix = result;

    return NUMERUS_MATRIX_SUCCESS;
}

int numerus_matrix_create_join_horizontal(
    numerus_matrix *parent,
    numerus_matrix *parent2,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_join(
        parent, parent2, NUMERUS_MATRIX_JOIN_HORIZONTAL, matrix
    );
}

int numerus_matrix_create_join_vertical(
    numerus_matrix *parent,
    numerus_matrix *parent2,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_join(
        parent, parent2, NUMERUS_MATRIX_JOIN_VERTICAL, matrix
    );
}


























