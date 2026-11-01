/**
 * @file numelyx_core_matrix.c
 * @brief Core Matrix lifecycle, dimensions, element access, and shared operations.
 *
 * @details This implementation follows the focused subsystem headers and
 * preserves their ownership, validation, and status-reporting contracts.
 */

#include "numelyx_core_matrix_internal.h"
#include "numelyx_core_size.h"
#include "numelyx_core_numeric.h"
#include "numelyx_core_matrix_lu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

static numelyx_core_matrix_status identity_coordinate_transform(
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

    *parent_row = row;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status passthrough_value_transform(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    (void) row;
    (void) column;
    (void) context;

    if (result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *result = parent_value;

    return NUMERUS_MATRIX_SUCCESS;
}

#define NUMERUS_MATRIX_CACHE_MAX_BYTES (64U * 1024U)

enum {
    NUMERUS_MATRIX_FLAG_SQUARE = UINT32_C(1) << 0,
    NUMERUS_MATRIX_FLAG_ZERO = UINT32_C(1) << 1,
    NUMERUS_MATRIX_FLAG_DIAGONAL = UINT32_C(1) << 2,
    NUMERUS_MATRIX_FLAG_UPPER_TRIANGULAR = UINT32_C(1) << 3,
    NUMERUS_MATRIX_FLAG_LOWER_TRIANGULAR = UINT32_C(1) << 4,
    NUMERUS_MATRIX_FLAG_SYMMETRIC = UINT32_C(1) << 5,
    NUMERUS_MATRIX_FLAG_IDENTITY = UINT32_C(1) << 6,
    NUMERUS_MATRIX_FLAGS_ALL = (UINT32_C(1) << 7) - 1
};









static numelyx_core_matrix_status scale_value_transform(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    const double *scalar = context;

    (void) row;
    (void) column;

    if (scalar == NULL || result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *result = parent_value * *scalar;
    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status add_scalar_value_transform(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    const double *scalar = context;

    (void) row;
    (void) column;
    if (scalar == NULL || result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = parent_value + *scalar;
    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status subtract_scalar_value_transform(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    const double *scalar = context;

    (void) row;
    (void) column;
    if (scalar == NULL || result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *result = parent_value - *scalar;
    return NUMERUS_MATRIX_SUCCESS;
}









static numelyx_core_matrix_status divide_scalar_value_transform(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    const double *divisor = context;

    (void) row;
    (void) column;

    if (divisor == NULL || result == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *result = parent_value / *divisor;
    return NUMERUS_MATRIX_SUCCESS;
}





















static int dimensions_valid(size_t rows, size_t columns)
{
    return rows != 0 && columns != 0;
}

static int dimensions_are_square(size_t rows, size_t columns)
{
    return rows == columns;
}

static numelyx_core_matrix_status storage_status_to_matrix_status(int status)
{
    switch (status) {
        case NUMERUS_STORAGE_SUCCESS:
            return NUMERUS_MATRIX_SUCCESS;
        case NUMERUS_STORAGE_INVALID_ARGUMENT:
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        case NUMERUS_STORAGE_OVERFLOW:
            return NUMERUS_MATRIX_OVERFLOW;
        case NUMERUS_STORAGE_OUT_OF_MEMORY:
            return NUMERUS_MATRIX_OUT_OF_MEMORY;
        case NUMERUS_STORAGE_OUT_OF_BOUNDS:
            return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static void numelyx_core_matrix_destroy_final(void *object);

int numelyx_core_matrix_allocate(numelyx_core_matrix **matrix)
{
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *matrix = numelyx_core_matrix_alloc(sizeof(**matrix));
    if (*matrix == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    if (!numelyx_core_mutex_init(&(*matrix)->cache_mutex)) {
        numelyx_core_matrix_free(*matrix);
        *matrix = NULL;
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    if (numelyx_core_lifetime_init(
            &(*matrix)->lifetime, numelyx_core_matrix_destroy_final
        ) != NUMELYX_CORE_LIFETIME_SUCCESS) {
        numelyx_core_mutex_destroy(&(*matrix)->cache_mutex);
        numelyx_core_matrix_free(*matrix);
        *matrix = NULL;
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    (*matrix)->rows = 0;
    (*matrix)->columns = 0;
    (*matrix)->storage = NULL;
    (*matrix)->parent = NULL;
    (*matrix)->parent2 = NULL;
    (*matrix)->join_type = NUMERUS_MATRIX_JOIN_HORIZONTAL;
    (*matrix)->binary_operation = NUMERUS_MATRIX_BINARY_NONE;
    (*matrix)->coordinate_transform = NULL;
    (*matrix)->value_transform = NULL;
    (*matrix)->transform_context = NULL;
    (*matrix)->transform_index1 = 0;
    (*matrix)->transform_index2 = 0;
    (*matrix)->cached_flags = (numelyx_core_matrix_flags) {0};
    (*matrix)->flags_computed = 0;
    (*matrix)->determinant_state = 0;
    (*matrix)->cached_determinant = 0.0;
    (*matrix)->cached_inverse_values = NULL;
    (*matrix)->cached_lu_factorization = NULL;
    (*matrix)->transform_scalar = 0.0;
    (*matrix)->transform_row_offset = 0;
    (*matrix)->transform_column_offset = 0;
    (*matrix)->transform_parent_columns = 0;
    (*matrix)->selection_indices = NULL;
    (*matrix)->selection_axis = NUMERUS_MATRIX_SELECTION_NONE;
    (*matrix)->padding_view = false;
    (*matrix)->padding_top = 0;
    (*matrix)->padding_left = 0;
    (*matrix)->padding_value = 0.0;
    (*matrix)->repeat_view = false;
    (*matrix)->block_diagonal_view = false;
    (*matrix)->block_grid_view = false;
    (*matrix)->block_matrices = NULL;
    (*matrix)->block_row_offsets = NULL;
    (*matrix)->block_column_offsets = NULL;
    (*matrix)->block_row_count = 0;
    (*matrix)->block_column_count = 0;
    (*matrix)->block_count = 0;
    (*matrix)->diagonal_matrix_view = false;
    (*matrix)->diagonal_offset_positive = true;
    (*matrix)->diagonal_offset_magnitude = 0;
    (*matrix)->diagonal_vector_length = 0;

    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status create_storage(
    numelyx_core_storage_kind kind,
    size_t rows,
    size_t columns,
    const numelyx_core_matrix_data *data,
    numelyx_core_storage **storage
)
{
    if (storage == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    switch (kind) {
        case NUMERUS_STORAGE_DENSE:
            return storage_status_to_matrix_status(numelyx_core_storage_create_dense(
                rows,
                columns,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_UPPER_TRIANGULAR:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numelyx_core_storage_create_upper_triangular(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_LOWER_TRIANGULAR:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numelyx_core_storage_create_lower_triangular(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_DIAGONAL:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numelyx_core_storage_create_diagonal(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_IDENTITY:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numelyx_core_storage_create_identity(rows, storage));

        case NUMERUS_STORAGE_CONSTANT:
            return storage_status_to_matrix_status(numelyx_core_storage_create_constant(
                rows,
                columns,
                data == NULL ? 0.0 : data->value,
                storage
            ));

        case NUMERUS_STORAGE_ZERO:
            return storage_status_to_matrix_status(numelyx_core_storage_create_zero(rows, columns, storage));

        case NUMERUS_STORAGE_SCALED_IDENTITY:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numelyx_core_storage_create_scaled_identity(
                rows,
                data == NULL ? 0.0 : data->value,
                storage
            ));

        case NUMERUS_STORAGE_SPARSE:
            return storage_status_to_matrix_status(numelyx_core_storage_create_sparse(
                rows,
                columns,
                data == NULL ? 0.0 : data->default_value,
                data == NULL ? NULL : data->entries,
                data == NULL ? 0 : data->count,
                storage
            ));

        case NUMERUS_STORAGE_SYMMETRIC:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numelyx_core_storage_create_symmetric(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_BANDED:
            return storage_status_to_matrix_status(numelyx_core_storage_create_banded(
                rows,
                columns,
                data == NULL ? 0 : data->lower_bandwidth,
                data == NULL ? 0 : data->upper_bandwidth,
                data == NULL ? NULL : data->values,
                storage
            ));
    }

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

/**
 * Create a root Matrix by constructing and owning its immutable Storage.
 */
int numelyx_core_matrix_create(
    numelyx_core_storage_kind kind,
    size_t rows,
    size_t columns,
    const numelyx_core_matrix_data *data,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_storage *storage = NULL;
    numelyx_core_matrix *result = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *matrix = NULL;

    if (!dimensions_valid(rows, columns)) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = create_storage(kind, rows, columns, data, &storage);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    status = numelyx_core_matrix_allocate(&result);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numelyx_core_storage_destroy(storage);
        return status;
    }

    result->rows = rows;
    result->columns = columns;
    result->storage = storage;
    result->parent = NULL;
    result->parent2 = NULL;
    result->coordinate_transform = NULL;
    result->value_transform = NULL;
    result->transform_context = NULL;
    *matrix = result;

    return NUMERUS_MATRIX_SUCCESS;
}

/**
 * Create a derived Matrix that delegates all reads to its parent.
 */
int numelyx_core_matrix_create_from_parent(
    numelyx_core_matrix *parent,
    size_t rows,
    size_t columns,
    numelyx_core_matrix **matrix
)
{
    return numelyx_core_matrix_create_from_parent_with_transforms(
        parent,
        rows,
        columns,
        NULL,
        NULL,
        NULL,
        matrix
    );
}

int numelyx_core_matrix_create_from_parent_with_transforms(
    numelyx_core_matrix *parent,
    size_t rows,
    size_t columns,
    numelyx_core_coordinate_transform_fn coordinate_transform,
    numelyx_core_value_transform_fn value_transform,
    const void *context,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_matrix *result;
    int status;

    if (parent == NULL || matrix == NULL || !dimensions_valid(rows, columns)) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *matrix = NULL;

    status = numelyx_core_matrix_allocate(&result);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    status = numelyx_core_matrix_retain(parent);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numelyx_core_matrix_free(result);
        return status;
    }

    result->rows = rows;
    result->columns = columns;
    result->storage = NULL;
    result->parent = parent;
    result->parent2 = NULL;
    result->coordinate_transform = coordinate_transform == NULL
        ? identity_coordinate_transform
        : coordinate_transform;
    result->value_transform = value_transform == NULL
        ? passthrough_value_transform
        : value_transform;
    result->transform_context = context;
    *matrix = result;

    return NUMERUS_MATRIX_SUCCESS;
}

int numelyx_core_matrix_create_scale(
    numelyx_core_matrix *parent,
    double scalar,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_matrix *view = NULL;
    numelyx_core_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numelyx_core_matrix_create_from_parent_with_transforms(
        parent,
        numelyx_core_matrix_rows(parent),
        numelyx_core_matrix_columns(parent),
        NULL,
        scale_value_transform,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    /*
     * The callback context points into the stable, heap-allocated view.
     * This avoids borrowing a stack scalar or allocating a separate context.
     */
    view->transform_scalar = scalar;
    view->transform_context = &view->transform_scalar;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}

int numelyx_core_matrix_create_negate(
    numelyx_core_matrix *parent,
    numelyx_core_matrix **matrix
)
{
    return numelyx_core_matrix_create_scale(parent, -1.0, matrix);
}

static numelyx_core_matrix_status create_scalar_transform_view(
    numelyx_core_matrix *parent,
    double scalar,
    numelyx_core_value_transform_fn transform,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_matrix *view = NULL;
    numelyx_core_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;
    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numelyx_core_matrix_create_from_parent_with_transforms(
        parent,
        numelyx_core_matrix_rows(parent),
        numelyx_core_matrix_columns(parent),
        NULL,
        transform,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_scalar = scalar;
    view->transform_context = &view->transform_scalar;
    *matrix = view;
    return NUMERUS_MATRIX_SUCCESS;
}

int numelyx_core_matrix_create_scalar_add(
    numelyx_core_matrix *parent,
    double scalar,
    numelyx_core_matrix **matrix
)
{
    return create_scalar_transform_view(
        parent, scalar, add_scalar_value_transform, matrix
    );
}

int numelyx_core_matrix_create_scalar_subtract(
    numelyx_core_matrix *parent,
    double scalar,
    numelyx_core_matrix **matrix
)
{
    return create_scalar_transform_view(
        parent, scalar, subtract_scalar_value_transform, matrix
    );
}





















int numelyx_core_matrix_create_divide_scalar(
    numelyx_core_matrix *parent,
    double divisor,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_matrix *view = NULL;
    numelyx_core_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (divisor == 0.0) {
        return NUMERUS_MATRIX_DIVISION_BY_ZERO;
    }

    status = numelyx_core_matrix_create_from_parent_with_transforms(
        parent,
        numelyx_core_matrix_rows(parent),
        numelyx_core_matrix_columns(parent),
        NULL,
        divide_scalar_value_transform,
        NULL,
        &view
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    view->transform_scalar = divisor;
    view->transform_context = &view->transform_scalar;
    *matrix = view;

    return NUMERUS_MATRIX_SUCCESS;
}





















































/**
 * Read a Matrix element without validating this Matrix's own coordinates.
 *
 * Parent access remains checked because a coordinate transform can map a
 * valid child coordinate outside the parent's logical dimensions.
 */
numelyx_core_matrix_status numelyx_core_matrix_get_unchecked(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    double *value
)
{
    if (matrix == NULL || value == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (matrix->padding_view) {
        size_t parent_row;
        size_t parent_column;

        if (matrix->parent == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        if (row < matrix->padding_top || column < matrix->padding_left) {
            *value = matrix->padding_value;
            return NUMERUS_MATRIX_SUCCESS;
        }

        parent_row = row - matrix->padding_top;
        parent_column = column - matrix->padding_left;

        if (parent_row >= numelyx_core_matrix_rows(matrix->parent) ||
            parent_column >= numelyx_core_matrix_columns(matrix->parent)) {
            *value = matrix->padding_value;
            return NUMERUS_MATRIX_SUCCESS;
        }

        return numelyx_core_matrix_get(
            matrix->parent, parent_row, parent_column, value
        );
    }

    if (matrix->repeat_view) {
        size_t parent_rows;
        size_t parent_columns;

        if (matrix->parent == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        parent_rows = numelyx_core_matrix_rows(matrix->parent);
        parent_columns = numelyx_core_matrix_columns(matrix->parent);
        return numelyx_core_matrix_get(
            matrix->parent, row % parent_rows, column % parent_columns, value
        );
    }

    if (matrix->block_diagonal_view) {
        size_t parent_rows;
        size_t parent_columns;
        size_t block_row;
        size_t block_column;

        if (matrix->parent == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        parent_rows = numelyx_core_matrix_rows(matrix->parent);
        parent_columns = numelyx_core_matrix_columns(matrix->parent);
        block_row = row / parent_rows;
        block_column = column / parent_columns;

        if (block_row != block_column) {
            *value = 0.0;
            return NUMERUS_MATRIX_SUCCESS;
        }

        return numelyx_core_matrix_get(
            matrix->parent, row % parent_rows, column % parent_columns, value
        );
    }

    if (matrix->diagonal_matrix_view) {
        size_t diagonal_index;

        if (matrix->parent == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        if (matrix->diagonal_offset_positive) {
            if (column < row ||
                column - row != matrix->diagonal_offset_magnitude) {
                *value = 0.0;
                return NUMERUS_MATRIX_SUCCESS;
            }
            diagonal_index = row;
        } else {
            if (row < column ||
                row - column != matrix->diagonal_offset_magnitude) {
                *value = 0.0;
                return NUMERUS_MATRIX_SUCCESS;
            }
            diagonal_index = column;
        }

        if (diagonal_index >= matrix->diagonal_vector_length) {
            *value = 0.0;
            return NUMERUS_MATRIX_SUCCESS;
        }

        if (numelyx_core_matrix_rows(matrix->parent) == 1) {
            return numelyx_core_matrix_get(
                matrix->parent, 0, diagonal_index, value
            );
        }
        return numelyx_core_matrix_get(
            matrix->parent, diagonal_index, 0, value
        );
    }

    if (matrix->block_grid_view) {
        size_t block_row = 0;
        size_t block_column = 0;
        size_t local_row;
        size_t local_column;
        numelyx_core_matrix *block;

        if (matrix->block_matrices == NULL ||
            matrix->block_row_offsets == NULL ||
            matrix->block_column_offsets == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        while (block_row + 1 < matrix->block_row_count &&
            row >= matrix->block_row_offsets[block_row + 1]) {
            block_row++;
        }
        while (block_column + 1 < matrix->block_column_count &&
            column >= matrix->block_column_offsets[block_column + 1]) {
            block_column++;
        }

        local_row = row - matrix->block_row_offsets[block_row];
        local_column = column - matrix->block_column_offsets[block_column];
        block = matrix->block_matrices[
            block_row * matrix->block_column_count + block_column
        ];

        return numelyx_core_matrix_get(block, local_row, local_column, value);
    }

    if (matrix->parent2 != NULL &&
        matrix->binary_operation != NUMERUS_MATRIX_BINARY_NONE) {
        double left_value;
        double right_value;
        double result_value;
        numelyx_core_matrix_status status;

        status = numelyx_core_matrix_get(
            matrix->parent, row, column, &left_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        status = numelyx_core_matrix_get(
            matrix->parent2, row, column, &right_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        if (matrix->binary_operation == NUMERUS_MATRIX_BINARY_ADD) {
            result_value = left_value + right_value;
        } else if (matrix->binary_operation == NUMERUS_MATRIX_BINARY_SUBTRACT) {
            result_value = left_value - right_value;
        } else if (matrix->binary_operation == NUMERUS_MATRIX_BINARY_HADAMARD) {
            result_value = left_value * right_value;
        } else if (matrix->binary_operation == NUMERUS_MATRIX_BINARY_DIVIDE) {
            result_value = left_value / right_value;
        } else {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        *value = result_value;
        return NUMERUS_MATRIX_SUCCESS;
    }

    if (matrix->parent2 != NULL) {
        if (matrix->join_type == NUMERUS_MATRIX_JOIN_HORIZONTAL) {
            if (column < numelyx_core_matrix_columns(matrix->parent)) {
                return numelyx_core_matrix_get(
                    matrix->parent, row, column, value
                );
            }

            return numelyx_core_matrix_get(
                matrix->parent2,
                row,
                column - numelyx_core_matrix_columns(matrix->parent),
                value
            );
        }

        if (matrix->join_type == NUMERUS_MATRIX_JOIN_VERTICAL) {
            if (row < numelyx_core_matrix_rows(matrix->parent)) {
                return numelyx_core_matrix_get(
                    matrix->parent, row, column, value
                );
            }

            return numelyx_core_matrix_get(
                matrix->parent2,
                row - numelyx_core_matrix_rows(matrix->parent),
                column,
                value
            );
        }

        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (matrix->parent != NULL) {
        size_t parent_row;
        size_t parent_column;
        double parent_value;
        double transformed_value;
        numelyx_core_matrix_status status;

        status = matrix->coordinate_transform(
            row,
            column,
            &parent_row,
            &parent_column,
            matrix->transform_context
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        status = numelyx_core_matrix_get(
            matrix->parent,
            parent_row,
            parent_column,
            &parent_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        status = matrix->value_transform(
            row,
            column,
            parent_value,
            &transformed_value,
            matrix->transform_context
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        *value = transformed_value;
        return NUMERUS_MATRIX_SUCCESS;
    }

    *value = numelyx_core_storage_get_unchecked(matrix->storage, row, column);

    return NUMERUS_MATRIX_SUCCESS;
}

/**
 * Validate Matrix coordinates and read the logical value.
 */
numelyx_core_matrix_status numelyx_core_matrix_get(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    double *value
)
{
    if (matrix == NULL || value == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    if (row >= matrix->rows || column >= matrix->columns) {
        return NUMERUS_MATRIX_OUT_OF_BOUNDS;
    }

    return numelyx_core_matrix_get_unchecked(matrix, row, column, value);
}


numelyx_core_matrix_status numelyx_core_matrix_get_flags(
    numelyx_core_matrix *matrix,
    numelyx_core_matrix_flags *flags
)
{
    numelyx_core_matrix_flags result = {0};
    bool is_zero = true;
    bool is_diagonal = true;
    bool is_upper = true;
    bool is_lower = true;
    bool is_symmetric = true;
    bool is_identity = true;
    size_t row, column;

    if (matrix == NULL || flags == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if ((matrix->flags_computed & NUMERUS_MATRIX_FLAGS_ALL) ==
        NUMERUS_MATRIX_FLAGS_ALL) {
        *flags = matrix->cached_flags;
        return NUMERUS_MATRIX_SUCCESS;
    }

    result.square = matrix->rows == matrix->columns;

    for (row = 0; row < matrix->rows; row++) {
        for (column = 0; column < matrix->columns; column++) {
            double value;
            numelyx_core_matrix_status status = numelyx_core_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (!numelyx_core_double_is_zero(value)) {
                is_zero = false;
            }
            if (row != column && !numelyx_core_double_is_zero(value)) {
                is_diagonal = false;
            }
            if (row > column && !numelyx_core_double_is_zero(value)) {
                is_upper = false;
            }
            if (row < column && !numelyx_core_double_is_zero(value)) {
                is_lower = false;
            }
            if ((row == column && !numelyx_core_double_is_one(value)) ||
                (row != column && !numelyx_core_double_is_zero(value))) {
                is_identity = false;
            }
            /*
             * Compare each off-diagonal pair once during the shared scan.
             * The previous second pass fetched both values for every pair.
             */
            if (result.square && row < column && is_symmetric) {
                double transposed_value;
                status = numelyx_core_matrix_get(
                    matrix, column, row, &transposed_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    return status;
                }
                if (!numelyx_core_double_equals(value, transposed_value)) {
                    is_symmetric = false;
                }
            }
        }
    }

    result.zero = is_zero;
    if (result.square) {
        result.diagonal = is_diagonal;
        result.upper_triangular = is_upper;
        result.lower_triangular = is_lower;
        result.identity = is_identity;
        result.symmetric = is_symmetric;
    }

    matrix->cached_flags = result;
    matrix->flags_computed = NUMERUS_MATRIX_FLAGS_ALL;
    *flags = result;
    return NUMERUS_MATRIX_SUCCESS;
}

static numelyx_core_matrix_status matrix_determinant_from_triangular(
    numelyx_core_matrix *matrix,
    double *determinant
)
{
    double result = 1.0;
    size_t index;
    for (index = 0; index < matrix->rows; index++) {
        double diagonal_value;
        numelyx_core_matrix_status status = numelyx_core_matrix_get(
            matrix, index, index, &diagonal_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        result *= diagonal_value;
    }
    *determinant = result;
    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_determinant(
    numelyx_core_matrix *matrix,
    double *determinant
)
{
    size_t size;
    size_t index;
    double result = 1.0;
    numelyx_core_matrix_lu_factorization *factorization = NULL;
    numelyx_core_matrix_status status;

    if (matrix == NULL || determinant == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (matrix->rows != matrix->columns) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }
    if (matrix->determinant_state == 2) {
        *determinant = matrix->cached_determinant;
        return NUMERUS_MATRIX_SUCCESS;
    }
    if (matrix->determinant_state == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    matrix->determinant_state = 1;
    size = matrix->rows;
    if ((matrix->flags_computed &
        (NUMERUS_MATRIX_FLAG_ZERO | NUMERUS_MATRIX_FLAG_IDENTITY |
         NUMERUS_MATRIX_FLAG_UPPER_TRIANGULAR |
         NUMERUS_MATRIX_FLAG_LOWER_TRIANGULAR)) ==
        (NUMERUS_MATRIX_FLAG_ZERO | NUMERUS_MATRIX_FLAG_IDENTITY |
         NUMERUS_MATRIX_FLAG_UPPER_TRIANGULAR |
         NUMERUS_MATRIX_FLAG_LOWER_TRIANGULAR)) {
        if (matrix->cached_flags.zero) {
            result = 0.0;
            goto cache_result;
        }
        if (matrix->cached_flags.identity) {
            result = 1.0;
            goto cache_result;
        }
        if (matrix->cached_flags.upper_triangular ||
            matrix->cached_flags.lower_triangular) {
            status = matrix_determinant_from_triangular(matrix, &result);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                matrix->determinant_state = 0;
                return status;
            }
            goto cache_result;
        }
    }

    status = numelyx_core_matrix_get_or_factorize_lu(matrix, &factorization);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        matrix->determinant_state = 0;
        return status;
    }

    if (numelyx_core_matrix_lu_rank(factorization) < size) {
        result = 0.0;
    } else {
        result = (double) numelyx_core_matrix_lu_permutation_sign(factorization);
        for (index = 0; index < size; index++) {
            double diagonal_value;

            status = numelyx_core_matrix_lu_get_upper(
                factorization, index, index, &diagonal_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numelyx_core_matrix_lu_destroy(factorization);
                matrix->determinant_state = 0;
                return status;
            }

            result *= diagonal_value;
        }
    }

    numelyx_core_matrix_lu_destroy(factorization);

cache_result:
    matrix->cached_determinant = result;
    matrix->determinant_state = 2;
    *determinant = result;
    return NUMERUS_MATRIX_SUCCESS;
}

size_t numelyx_core_matrix_rows(const numelyx_core_matrix *matrix)
{
    return matrix == NULL ? 0 : matrix->rows;
}

size_t numelyx_core_matrix_columns(const numelyx_core_matrix *matrix)
{
    return matrix == NULL ? 0 : matrix->columns;
}

numelyx_core_storage_kind numelyx_core_matrix_storage_kind(
    const numelyx_core_matrix *matrix
)
{
    if (matrix == NULL) {
        return NUMERUS_STORAGE_ZERO;
    }

    if (matrix->parent != NULL) {
        return numelyx_core_matrix_storage_kind(matrix->parent);
    }

    return numelyx_core_storage_kind_of(matrix->storage);
}

/**
 * Release one Matrix reference. Final reclamation releases retained dependencies.
 */

/*
 * Numerical caches own bounded values/factorizations, contain no parent
 * references, and are destroyed with their owner.
 * Cache population is not thread-safe; concurrent mutation of one Matrix's
 * lazy caches is outside the current API contract.
 */
numelyx_core_matrix_status numelyx_core_matrix_get_cached_inverse(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **inverse
)
{
    if (inverse == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *inverse = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (matrix->cached_inverse_values == NULL) {
        return NUMERUS_MATRIX_SUCCESS;
    }
    return (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        matrix->rows, matrix->columns, matrix->cached_inverse_values, inverse
    );
}

void numelyx_core_matrix_store_inverse_cache(
    const numelyx_core_matrix *matrix,
    const double *values
)
{
    numelyx_core_matrix *mutable_matrix;
    size_t element_count;
    size_t value_bytes;
    size_t index;
    double *copy;

    if (matrix == NULL || values == NULL ||
        !numelyx_core_size_multiply(matrix->rows, matrix->columns, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(*values), &value_bytes) ||
        value_bytes > NUMERUS_MATRIX_CACHE_MAX_BYTES) {
        return;
    }

    mutable_matrix = (numelyx_core_matrix *) matrix;
    if (mutable_matrix->cached_inverse_values != NULL) {
        return;
    }

    copy = numelyx_core_matrix_alloc(value_bytes);
    if (copy == NULL) {
        return;
    }
    for (index = 0; index < element_count; index++) {
        copy[index] = values[index];
    }
    mutable_matrix->cached_inverse_values = copy;
}

numelyx_core_matrix_status numelyx_core_matrix_get_or_factorize_lu(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix_lu_factorization **factorization
)
{
    numelyx_core_matrix_status status;

    if (factorization == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *factorization = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numelyx_core_matrix_get_cached_lu(matrix, factorization);
    if (status != NUMERUS_MATRIX_SUCCESS || *factorization != NULL) {
        return status;
    }

    status = numelyx_core_matrix_lu_factorize(matrix, factorization);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        numelyx_core_matrix_store_lu_cache(matrix, *factorization);
    }
    return status;
}

numelyx_core_matrix_status numelyx_core_matrix_get_cached_lu(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix_lu_factorization **factorization
)
{
    numelyx_core_matrix_status status;

    if (factorization == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *factorization = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (matrix->cached_lu_factorization == NULL) {
        return NUMERUS_MATRIX_SUCCESS;
    }

    status = numelyx_core_matrix_lu_retain(matrix->cached_lu_factorization);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *factorization = matrix->cached_lu_factorization;
    }
    return status;
}

void numelyx_core_matrix_store_lu_cache(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix_lu_factorization *factorization
)
{
    numelyx_core_matrix *mutable_matrix;
    size_t size;
    size_t element_count;
    size_t values_bytes;
    size_t permutation_bytes;
    size_t total_bytes;

    if (matrix == NULL || factorization == NULL ||
        matrix->rows != matrix->columns ||
        !numelyx_core_size_multiply(matrix->rows, matrix->columns, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(double), &values_bytes) ||
        !numelyx_core_size_multiply(matrix->rows, sizeof(size_t), &permutation_bytes) ||
        !numelyx_core_size_add(values_bytes, permutation_bytes, &total_bytes) ||
        total_bytes > NUMERUS_MATRIX_CACHE_MAX_BYTES) {
        return;
    }

    mutable_matrix = (numelyx_core_matrix *) matrix;
    if (mutable_matrix->cached_lu_factorization != NULL) {
        return;
    }
    size = numelyx_core_matrix_lu_size(factorization);
    if (size != matrix->rows ||
        numelyx_core_matrix_lu_retain(factorization) != NUMERUS_MATRIX_SUCCESS) {
        return;
    }
    mutable_matrix->cached_lu_factorization = factorization;
}

numelyx_core_matrix_status numelyx_core_matrix_retain(
    numelyx_core_matrix *matrix
)
{
    numelyx_core_lifetime_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numelyx_core_lifetime_retain(&matrix->lifetime);
    switch (status) {
        case NUMELYX_CORE_LIFETIME_SUCCESS:
            return NUMERUS_MATRIX_SUCCESS;
        case NUMELYX_CORE_LIFETIME_OVERFLOW:
            return NUMERUS_MATRIX_OVERFLOW;
        default:
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
}

static void numelyx_core_matrix_destroy_final(void *object)
{
    numelyx_core_matrix *matrix = (numelyx_core_matrix *) object;
    numelyx_core_matrix *parent;
    numelyx_core_matrix *parent2;
    numelyx_core_matrix **block_matrices;
    size_t block_count;
    size_t index;

    parent = matrix->parent;
    parent2 = matrix->parent2;
    block_matrices = matrix->block_matrices;
    block_count = matrix->block_count;

    if (parent == NULL) {
        numelyx_core_storage_destroy(matrix->storage);
    }

    numelyx_core_matrix_lu_destroy(matrix->cached_lu_factorization);
    numelyx_core_matrix_free(matrix->cached_inverse_values);
    numelyx_core_matrix_free(matrix->selection_indices);
    numelyx_core_matrix_free(matrix->block_row_offsets);
    numelyx_core_matrix_free(matrix->block_column_offsets);
    numelyx_core_mutex_destroy(&matrix->cache_mutex);
    numelyx_core_matrix_free(matrix);

    numelyx_core_matrix_destroy(parent);
    numelyx_core_matrix_destroy(parent2);
    for (index = 0; index < block_count; index++) {
        numelyx_core_matrix_destroy(block_matrices[index]);
    }
    numelyx_core_matrix_free(block_matrices);
}

void numelyx_core_matrix_destroy(numelyx_core_matrix *matrix)
{
    if (matrix != NULL) {
        (void) numelyx_core_lifetime_release(
            &matrix->lifetime, matrix
        );
    }
}
