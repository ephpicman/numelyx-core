/**
 * @file numerus_matrix.c
 * @brief Core Matrix lifecycle, dimensions, element access, and shared operations.
 *
 * @details This implementation follows the focused subsystem headers and
 * preserves their ownership, validation, and status-reporting contracts.
 */

#include "numerus_matrix_internal.h"
#include "numerus_size.h"
#include "numerus_numeric.h"
#include "numerus_matrix_lu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

static numerus_matrix_status identity_coordinate_transform(
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

static numerus_matrix_status passthrough_value_transform(
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









static numerus_matrix_status scale_value_transform(
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

static numerus_matrix_status add_scalar_value_transform(
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

static numerus_matrix_status subtract_scalar_value_transform(
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









static numerus_matrix_status divide_scalar_value_transform(
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

static numerus_matrix_status storage_status_to_matrix_status(int status)
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

int numerus_matrix_allocate(numerus_matrix **matrix)
{
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *matrix = numerus_matrix_alloc(sizeof(**matrix));
    if (*matrix == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
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
    (*matrix)->cached_flags = (numerus_matrix_flags) {0};
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
    (*matrix)->diagonal_matrix_view = false;
    (*matrix)->diagonal_offset_positive = true;
    (*matrix)->diagonal_offset_magnitude = 0;
    (*matrix)->diagonal_vector_length = 0;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status create_storage(
    numerus_storage_kind kind,
    size_t rows,
    size_t columns,
    const numerus_matrix_data *data,
    numerus_storage **storage
)
{
    if (storage == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    switch (kind) {
        case NUMERUS_STORAGE_DENSE:
            return storage_status_to_matrix_status(numerus_storage_create_dense(
                rows,
                columns,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_UPPER_TRIANGULAR:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numerus_storage_create_upper_triangular(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_LOWER_TRIANGULAR:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numerus_storage_create_lower_triangular(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_DIAGONAL:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numerus_storage_create_diagonal(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_IDENTITY:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numerus_storage_create_identity(rows, storage));

        case NUMERUS_STORAGE_CONSTANT:
            return storage_status_to_matrix_status(numerus_storage_create_constant(
                rows,
                columns,
                data == NULL ? 0.0 : data->value,
                storage
            ));

        case NUMERUS_STORAGE_ZERO:
            return storage_status_to_matrix_status(numerus_storage_create_zero(rows, columns, storage));

        case NUMERUS_STORAGE_SCALED_IDENTITY:
            if (!dimensions_are_square(rows, columns)) {
                return NUMERUS_MATRIX_NOT_SQUARE;
            }
            return storage_status_to_matrix_status(numerus_storage_create_scaled_identity(
                rows,
                data == NULL ? 0.0 : data->value,
                storage
            ));

        case NUMERUS_STORAGE_SPARSE:
            return storage_status_to_matrix_status(numerus_storage_create_sparse(
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
            return storage_status_to_matrix_status(numerus_storage_create_symmetric(
                rows,
                data == NULL ? NULL : data->values,
                storage
            ));

        case NUMERUS_STORAGE_BANDED:
            return storage_status_to_matrix_status(numerus_storage_create_banded(
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
int numerus_matrix_create(
    numerus_storage_kind kind,
    size_t rows,
    size_t columns,
    const numerus_matrix_data *data,
    numerus_matrix **matrix
)
{
    numerus_storage *storage = NULL;
    numerus_matrix *result = NULL;
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

    status = numerus_matrix_allocate(&result);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        numerus_storage_destroy(storage);
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
int numerus_matrix_create_from_parent(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_from_parent_with_transforms(
        parent,
        rows,
        columns,
        NULL,
        NULL,
        NULL,
        matrix
    );
}

int numerus_matrix_create_from_parent_with_transforms(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_coordinate_transform_fn coordinate_transform,
    numerus_value_transform_fn value_transform,
    const void *context,
    numerus_matrix **matrix
)
{
    numerus_matrix *result;
    int status;

    if (parent == NULL || matrix == NULL || !dimensions_valid(rows, columns)) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *matrix = NULL;

    status = numerus_matrix_allocate(&result);
    if (status != NUMERUS_MATRIX_SUCCESS) {
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

int numerus_matrix_create_scale(
    numerus_matrix *parent,
    double scalar,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
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

int numerus_matrix_create_negate(
    numerus_matrix *parent,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_scale(parent, -1.0, matrix);
}

static numerus_matrix_status create_scalar_transform_view(
    numerus_matrix *parent,
    double scalar,
    numerus_value_transform_fn transform,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    numerus_matrix_status status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;
    if (parent == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
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

int numerus_matrix_create_scalar_add(
    numerus_matrix *parent,
    double scalar,
    numerus_matrix **matrix
)
{
    return create_scalar_transform_view(
        parent, scalar, add_scalar_value_transform, matrix
    );
}

int numerus_matrix_create_scalar_subtract(
    numerus_matrix *parent,
    double scalar,
    numerus_matrix **matrix
)
{
    return create_scalar_transform_view(
        parent, scalar, subtract_scalar_value_transform, matrix
    );
}





















int numerus_matrix_create_divide_scalar(
    numerus_matrix *parent,
    double divisor,
    numerus_matrix **matrix
)
{
    numerus_matrix *view = NULL;
    numerus_matrix_status status;

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

    status = numerus_matrix_create_from_parent_with_transforms(
        parent,
        numerus_matrix_rows(parent),
        numerus_matrix_columns(parent),
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
numerus_matrix_status numerus_matrix_get_unchecked(
    const numerus_matrix *matrix,
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

        if (parent_row >= numerus_matrix_rows(matrix->parent) ||
            parent_column >= numerus_matrix_columns(matrix->parent)) {
            *value = matrix->padding_value;
            return NUMERUS_MATRIX_SUCCESS;
        }

        return numerus_matrix_get(
            matrix->parent, parent_row, parent_column, value
        );
    }

    if (matrix->repeat_view) {
        size_t parent_rows;
        size_t parent_columns;

        if (matrix->parent == NULL) {
            return NUMERUS_MATRIX_INVALID_ARGUMENT;
        }

        parent_rows = numerus_matrix_rows(matrix->parent);
        parent_columns = numerus_matrix_columns(matrix->parent);
        return numerus_matrix_get(
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

        parent_rows = numerus_matrix_rows(matrix->parent);
        parent_columns = numerus_matrix_columns(matrix->parent);
        block_row = row / parent_rows;
        block_column = column / parent_columns;

        if (block_row != block_column) {
            *value = 0.0;
            return NUMERUS_MATRIX_SUCCESS;
        }

        return numerus_matrix_get(
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

        if (numerus_matrix_rows(matrix->parent) == 1) {
            return numerus_matrix_get(
                matrix->parent, 0, diagonal_index, value
            );
        }
        return numerus_matrix_get(
            matrix->parent, diagonal_index, 0, value
        );
    }

    if (matrix->block_grid_view) {
        size_t block_row = 0;
        size_t block_column = 0;
        size_t local_row;
        size_t local_column;
        numerus_matrix *block;

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

        return numerus_matrix_get(block, local_row, local_column, value);
    }

    if (matrix->parent2 != NULL &&
        matrix->binary_operation != NUMERUS_MATRIX_BINARY_NONE) {
        double left_value;
        double right_value;
        double result_value;
        numerus_matrix_status status;

        status = numerus_matrix_get(
            matrix->parent, row, column, &left_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) {
            return status;
        }

        status = numerus_matrix_get(
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
            if (column < numerus_matrix_columns(matrix->parent)) {
                return numerus_matrix_get(
                    matrix->parent, row, column, value
                );
            }

            return numerus_matrix_get(
                matrix->parent2,
                row,
                column - numerus_matrix_columns(matrix->parent),
                value
            );
        }

        if (matrix->join_type == NUMERUS_MATRIX_JOIN_VERTICAL) {
            if (row < numerus_matrix_rows(matrix->parent)) {
                return numerus_matrix_get(
                    matrix->parent, row, column, value
                );
            }

            return numerus_matrix_get(
                matrix->parent2,
                row - numerus_matrix_rows(matrix->parent),
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
        numerus_matrix_status status;

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

        status = numerus_matrix_get(
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

    *value = numerus_storage_get_unchecked(matrix->storage, row, column);

    return NUMERUS_MATRIX_SUCCESS;
}

/**
 * Validate Matrix coordinates and read the logical value.
 */
numerus_matrix_status numerus_matrix_get(
    const numerus_matrix *matrix,
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

    return numerus_matrix_get_unchecked(matrix, row, column, value);
}


numerus_matrix_status numerus_matrix_get_flags(
    numerus_matrix *matrix,
    numerus_matrix_flags *flags
)
{
    numerus_matrix_flags result = {0};
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
            numerus_matrix_status status = numerus_matrix_get(
                matrix, row, column, &value
            );

            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }
            if (!numerus_double_is_zero(value)) {
                is_zero = false;
            }
            if (row != column && !numerus_double_is_zero(value)) {
                is_diagonal = false;
            }
            if (row > column && !numerus_double_is_zero(value)) {
                is_upper = false;
            }
            if (row < column && !numerus_double_is_zero(value)) {
                is_lower = false;
            }
            if ((row == column && !numerus_double_is_one(value)) ||
                (row != column && !numerus_double_is_zero(value))) {
                is_identity = false;
            }
            /*
             * Compare each off-diagonal pair once during the shared scan.
             * The previous second pass fetched both values for every pair.
             */
            if (result.square && row < column && is_symmetric) {
                double transposed_value;
                status = numerus_matrix_get(
                    matrix, column, row, &transposed_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    return status;
                }
                if (!numerus_double_equals(value, transposed_value)) {
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

static numerus_matrix_status matrix_determinant_from_triangular(
    numerus_matrix *matrix,
    double *determinant
)
{
    double result = 1.0;
    size_t index;
    for (index = 0; index < matrix->rows; index++) {
        double diagonal_value;
        numerus_matrix_status status = numerus_matrix_get(
            matrix, index, index, &diagonal_value
        );
        if (status != NUMERUS_MATRIX_SUCCESS) return status;
        result *= diagonal_value;
    }
    *determinant = result;
    return NUMERUS_MATRIX_SUCCESS;
}

numerus_matrix_status numerus_matrix_determinant(
    numerus_matrix *matrix,
    double *determinant
)
{
    size_t size;
    size_t index;
    double result = 1.0;
    numerus_matrix_lu_factorization *factorization = NULL;
    numerus_matrix_status status;

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

    status = numerus_matrix_get_or_factorize_lu(matrix, &factorization);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        matrix->determinant_state = 0;
        return status;
    }

    if (numerus_matrix_lu_rank(factorization) < size) {
        result = 0.0;
    } else {
        result = (double) numerus_matrix_lu_permutation_sign(factorization);
        for (index = 0; index < size; index++) {
            double diagonal_value;

            status = numerus_matrix_lu_get_upper(
                factorization, index, index, &diagonal_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                numerus_matrix_lu_destroy(factorization);
                matrix->determinant_state = 0;
                return status;
            }

            result *= diagonal_value;
        }
    }

    numerus_matrix_lu_destroy(factorization);

cache_result:
    matrix->cached_determinant = result;
    matrix->determinant_state = 2;
    *determinant = result;
    return NUMERUS_MATRIX_SUCCESS;
}

size_t numerus_matrix_rows(const numerus_matrix *matrix)
{
    return matrix == NULL ? 0 : matrix->rows;
}

size_t numerus_matrix_columns(const numerus_matrix *matrix)
{
    return matrix == NULL ? 0 : matrix->columns;
}

numerus_storage_kind numerus_matrix_storage_kind(
    const numerus_matrix *matrix
)
{
    if (matrix == NULL) {
        return NUMERUS_STORAGE_ZERO;
    }

    if (matrix->parent != NULL) {
        return numerus_matrix_storage_kind(matrix->parent);
    }

    return numerus_storage_kind_of(matrix->storage);
}

/**
 * Release a Matrix. Parent relationships are non-owning.
 */

/*
 * Numerical caches own bounded values/factorizations, contain no parent
 * references, and are destroyed with their owner.
 * Cache population is not thread-safe; concurrent mutation of one Matrix's
 * lazy caches is outside the current API contract.
 */
numerus_matrix_status numerus_matrix_get_cached_inverse(
    const numerus_matrix *matrix,
    numerus_matrix **inverse
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
    return (numerus_matrix_status) numerus_matrix_create_dense(
        matrix->rows, matrix->columns, matrix->cached_inverse_values, inverse
    );
}

void numerus_matrix_store_inverse_cache(
    const numerus_matrix *matrix,
    const double *values
)
{
    numerus_matrix *mutable_matrix;
    size_t element_count;
    size_t value_bytes;
    size_t index;
    double *copy;

    if (matrix == NULL || values == NULL ||
        !numerus_size_multiply(matrix->rows, matrix->columns, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(*values), &value_bytes) ||
        value_bytes > NUMERUS_MATRIX_CACHE_MAX_BYTES) {
        return;
    }

    mutable_matrix = (numerus_matrix *) matrix;
    if (mutable_matrix->cached_inverse_values != NULL) {
        return;
    }

    copy = numerus_matrix_alloc(value_bytes);
    if (copy == NULL) {
        return;
    }
    for (index = 0; index < element_count; index++) {
        copy[index] = values[index];
    }
    mutable_matrix->cached_inverse_values = copy;
}

numerus_matrix_status numerus_matrix_get_or_factorize_lu(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization **factorization
)
{
    numerus_matrix_status status;

    if (factorization == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *factorization = NULL;
    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    status = numerus_matrix_get_cached_lu(matrix, factorization);
    if (status != NUMERUS_MATRIX_SUCCESS || *factorization != NULL) {
        return status;
    }

    status = numerus_matrix_lu_factorize(matrix, factorization);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_store_lu_cache(matrix, *factorization);
    }
    return status;
}

numerus_matrix_status numerus_matrix_get_cached_lu(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization **factorization
)
{
    numerus_matrix_status status;

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

    status = numerus_matrix_lu_retain(matrix->cached_lu_factorization);
    if (status == NUMERUS_MATRIX_SUCCESS) {
        *factorization = matrix->cached_lu_factorization;
    }
    return status;
}

void numerus_matrix_store_lu_cache(
    const numerus_matrix *matrix,
    numerus_matrix_lu_factorization *factorization
)
{
    numerus_matrix *mutable_matrix;
    size_t size;
    size_t element_count;
    size_t values_bytes;
    size_t permutation_bytes;
    size_t total_bytes;

    if (matrix == NULL || factorization == NULL ||
        matrix->rows != matrix->columns ||
        !numerus_size_multiply(matrix->rows, matrix->columns, &element_count) ||
        !numerus_size_multiply(element_count, sizeof(double), &values_bytes) ||
        !numerus_size_multiply(matrix->rows, sizeof(size_t), &permutation_bytes) ||
        !numerus_size_add(values_bytes, permutation_bytes, &total_bytes) ||
        total_bytes > NUMERUS_MATRIX_CACHE_MAX_BYTES) {
        return;
    }

    mutable_matrix = (numerus_matrix *) matrix;
    if (mutable_matrix->cached_lu_factorization != NULL) {
        return;
    }
    size = numerus_matrix_lu_size(factorization);
    if (size != matrix->rows ||
        numerus_matrix_lu_retain(factorization) != NUMERUS_MATRIX_SUCCESS) {
        return;
    }
    mutable_matrix->cached_lu_factorization = factorization;
}

void numerus_matrix_destroy(numerus_matrix *matrix)
{
    if (matrix == NULL) {
        return;
    }

    if (matrix->parent == NULL) {
        numerus_storage_destroy(matrix->storage);
    }

    numerus_matrix_lu_destroy(matrix->cached_lu_factorization);
    numerus_matrix_free(matrix->cached_inverse_values);
    numerus_matrix_free(matrix->selection_indices);
    numerus_matrix_free(matrix->block_matrices);
    numerus_matrix_free(matrix->block_row_offsets);
    numerus_matrix_free(matrix->block_column_offsets);
    numerus_matrix_free(matrix);
}
