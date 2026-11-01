/**
 * @file matrix_test.c
 * @brief Core Matrix lifecycle, construction, metadata, and element-access regression tests.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"
#include "../numerus_numeric.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static int matrix_value_equals(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double expected
)
{
    double actual = 0.0;

    return numerus_matrix_get_unchecked(
        matrix, row, column, &actual
    ) == NUMERUS_MATRIX_SUCCESS && actual == expected;
}

static void assert_matrix_values(
    const numerus_matrix *matrix,
    size_t rows,
    size_t columns,
    const double *expected
);

static void test_factory_and_dimensions(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix_data data = {0};
    numerus_matrix *matrix = NULL;

    data.values = values;

    assert(numerus_matrix_create(
        NUMERUS_STORAGE_DENSE, 2, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(matrix) == 2);
    assert(numerus_matrix_columns(matrix) == 3);
    assert(numerus_matrix_storage_kind(matrix) == NUMERUS_STORAGE_DENSE);

    {
        double value = 0;
        assert(numerus_matrix_get(matrix, 1, 2, &value) == NUMERUS_MATRIX_SUCCESS);
        assert(value == 6.0);
    }

    numerus_matrix_destroy(matrix);
}

static void test_convenience_constructors(void)
{
    numerus_matrix *matrix = NULL;

    assert(numerus_matrix_create_identity(3, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_storage_kind(matrix) == NUMERUS_STORAGE_IDENTITY);
    assert(matrix_value_equals(matrix, 1, 1, 1.0));
    assert(matrix_value_equals(matrix, 1, 2, 0.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_zero(2, 3, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 1, 2, 0.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_zero_square(4, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(matrix) == 4);
    assert(numerus_matrix_columns(matrix) == 4);
    numerus_matrix_destroy(matrix);

    {
        const double diagonal[] = {10, 20, 30};

        assert(numerus_matrix_create_diagonal(
            3, diagonal, &matrix
        ) == NUMERUS_MATRIX_SUCCESS);
        assert(matrix_value_equals(matrix, 2, 2, 30.0));
        assert(matrix_value_equals(matrix, 0, 2, 0.0));
        numerus_matrix_destroy(matrix);
    }

    assert(numerus_matrix_create_constant(
        2, 3, 42.0, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 1, 2, 42.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_scaled_identity(
        3, 7.0, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 2, 2, 7.0));
    assert(matrix_value_equals(matrix, 2, 1, 0.0));
    numerus_matrix_destroy(matrix);
}

static void test_all_storage_kinds(void)
{
    const double packed[] = {1, 2, 3, 4, 5, 6};
    const double diagonal[] = {10, 20, 30};
    const double banded[] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 0
    };
    const numerus_storage_sparse_entry entries[] = {
        {5, 20},
        {1, 10}
    };
    numerus_matrix_data data = {0};
    numerus_matrix *matrix = NULL;

    assert(numerus_matrix_create_upper_triangular(
        3, packed, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 2, 3.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_lower_triangular(
        3, packed, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 2, 0, 4.0));
    numerus_matrix_destroy(matrix);

    data.values = packed;
    assert(numerus_matrix_create(
        NUMERUS_STORAGE_UPPER_TRIANGULAR, 3, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 2, 3.0));
    assert(matrix_value_equals(matrix, 2, 0, 0.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create(
        NUMERUS_STORAGE_LOWER_TRIANGULAR, 3, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 2, 0, 4.0));
    assert(matrix_value_equals(matrix, 0, 2, 0.0));
    numerus_matrix_destroy(matrix);

    data.values = diagonal;
    assert(numerus_matrix_create(
        NUMERUS_STORAGE_DIAGONAL, 3, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 2, 2, 30.0));
    numerus_matrix_destroy(matrix);

    data.value = 42.0;
    assert(numerus_matrix_create(
        NUMERUS_STORAGE_CONSTANT, 2, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 1, 42.0));
    numerus_matrix_destroy(matrix);

    data.default_value = -1.0;
    data.entries = entries;
    data.count = 2;
    assert(numerus_matrix_create(
        NUMERUS_STORAGE_SPARSE, 2, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 1, 10.0));
    assert(matrix_value_equals(matrix, 0, 0, -1.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_sparse(
        2, 3, -1.0, entries, 2, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 1, 10.0));
    numerus_matrix_destroy(matrix);

    data.values = banded;
    data.entries = NULL;
    data.count = 0;
    data.lower_bandwidth = 1;
    data.upper_bandwidth = 1;
    assert(numerus_matrix_create(
        NUMERUS_STORAGE_BANDED, 3, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 1, 2, 5.0));
    assert(matrix_value_equals(matrix, 0, 2, 0.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_symmetric(
        3, packed, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 2, 4.0));
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_banded(
        3, 3, 1, 1, banded, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 1, 2, 5.0));
    numerus_matrix_destroy(matrix);

    data.values = packed;
    assert(numerus_matrix_create(
        NUMERUS_STORAGE_SYMMETRIC, 3, 3, &data, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(matrix, 0, 2, 4.0));
    assert(matrix_value_equals(matrix, 2, 0, 4.0));
    numerus_matrix_destroy(matrix);
}

static void test_parent_chain(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *middle = NULL;
    numerus_matrix *leaf = NULL;

    assert(numerus_matrix_create_dense(
        2, 2, values, &root
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_from_parent(
        root, 2, 2, &middle
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_from_parent(
        middle, 2, 2, &leaf
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_storage_kind(leaf) == NUMERUS_STORAGE_DENSE);
    assert(numerus_matrix_rows(leaf) == 2);
    assert(numerus_matrix_columns(leaf) == 2);

    {
        double value = 0;

        assert(numerus_matrix_get(leaf, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
        assert(value == 2.0);

        assert(numerus_matrix_get(leaf, 1, 0, &value) == NUMERUS_MATRIX_SUCCESS);
        assert(value == 3.0);

        assert(matrix_value_equals(leaf, 1, 1, 4.0));
    }

    /* Child destruction must not destroy or mutate its parent. */
    numerus_matrix_destroy(leaf);
    assert(matrix_value_equals(middle, 1, 1, 4.0));

    numerus_matrix_destroy(middle);
    assert(matrix_value_equals(root, 0, 0, 1.0));

    numerus_matrix_destroy(root);

    /* A derived Matrix can expose different dimensions from its parent. */
    assert(numerus_matrix_create_dense(2, 3, (const double[]){
        1, 2, 3, 4, 5, 6
    }, &root) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent(
        root, 3, 2, &leaf
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(leaf) == 3);
    assert(numerus_matrix_columns(leaf) == 2);
    numerus_matrix_destroy(leaf);
    numerus_matrix_destroy(root);
}

static numerus_matrix_status transpose_coordinates(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) context;

    *parent_row = column;
    *parent_column = row;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status diagonal_index_values(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    (void) context;

    *result = row == column ? (double) (row + 1) : parent_value;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status invalid_coordinates(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) row;
    (void) column;
    (void) parent_row;
    (void) parent_column;
    (void) context;

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static numerus_matrix_status counting_identity_coordinates(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    size_t *calls = (size_t *) context;

    assert(calls != NULL);
    assert(parent_row != NULL);
    assert(parent_column != NULL);

    (*calls)++;
    *parent_row = row;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status out_of_bounds_coordinates(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) context;

    *parent_row = row + 2;
    *parent_column = column;

    return NUMERUS_MATRIX_SUCCESS;
}

static numerus_matrix_status invalid_value(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
)
{
    (void) row;
    (void) column;
    (void) parent_value;
    (void) result;
    (void) context;

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static void test_transpose_transform(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *parent = NULL;
    numerus_matrix *child = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_dense(2, 3, values, &parent) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        parent, 3, 2, transpose_coordinates, NULL, NULL, &child
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(child) == 3);
    assert(numerus_matrix_columns(child) == 2);
    assert(matrix_value_equals(child, 0, 0, 1.0));
    assert(matrix_value_equals(child, 0, 1, 4.0));
    assert(matrix_value_equals(child, 1, 0, 2.0));
    assert(matrix_value_equals(child, 1, 1, 5.0));
    assert(matrix_value_equals(child, 2, 0, 3.0));
    assert(matrix_value_equals(child, 2, 1, 6.0));

    assert(numerus_matrix_get(child, 3, 0, &value) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(value == 123.0);

    numerus_matrix_destroy(child);
    numerus_matrix_destroy(parent);
}

static void test_transpose_constructor(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *parent = NULL;
    numerus_matrix *transpose = NULL;
    numerus_matrix *double_transpose = NULL;
    numerus_matrix *invalid_output = NULL;

    assert(numerus_matrix_create_dense(2, 3, values, &parent) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_transpose(parent, &transpose) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(transpose) == 3);
    assert(numerus_matrix_columns(transpose) == 2);
    assert(numerus_matrix_storage_kind(transpose) == NUMERUS_STORAGE_DENSE);

    assert(matrix_value_equals(transpose, 0, 0, 1.0));
    assert(matrix_value_equals(transpose, 0, 1, 4.0));
    assert(matrix_value_equals(transpose, 1, 0, 2.0));
    assert(matrix_value_equals(transpose, 1, 1, 5.0));
    assert(matrix_value_equals(transpose, 2, 0, 3.0));
    assert(matrix_value_equals(transpose, 2, 1, 6.0));

    assert(numerus_matrix_create_transpose(transpose, &double_transpose) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(double_transpose) == 2);
    assert(numerus_matrix_columns(double_transpose) == 3);

    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 3; column++) {
            assert(matrix_value_equals(
                double_transpose,
                row,
                column,
                values[row * 3 + column]
            ));
        }
    }

    assert(numerus_matrix_create_transpose(NULL, &invalid_output) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_transpose(parent, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(double_transpose);
    numerus_matrix_destroy(transpose);
    numerus_matrix_destroy(parent);
}

static void test_orientation_transforms(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *parent = NULL;
    numerus_matrix *view = NULL;

    assert(numerus_matrix_create_dense(2, 3, values, &parent) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_flip_rows(parent, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(view) == 2);
    assert(numerus_matrix_columns(view) == 3);
    assert(matrix_value_equals(view, 0, 0, 4.0));
    assert(matrix_value_equals(view, 0, 2, 6.0));
    assert(matrix_value_equals(view, 1, 0, 1.0));
    assert(matrix_value_equals(view, 1, 2, 3.0));
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_flip_columns(parent, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(view, 0, 0, 3.0));
    assert(matrix_value_equals(view, 0, 2, 1.0));
    assert(matrix_value_equals(view, 1, 0, 6.0));
    assert(matrix_value_equals(view, 1, 2, 4.0));
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_rotate_90_clockwise(parent, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(view) == 3);
    assert(numerus_matrix_columns(view) == 2);
    assert(matrix_value_equals(view, 0, 0, 4.0));
    assert(matrix_value_equals(view, 0, 1, 1.0));
    assert(matrix_value_equals(view, 1, 0, 5.0));
    assert(matrix_value_equals(view, 1, 1, 2.0));
    assert(matrix_value_equals(view, 2, 0, 6.0));
    assert(matrix_value_equals(view, 2, 1, 3.0));
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_rotate_180(parent, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(view) == 2);
    assert(numerus_matrix_columns(view) == 3);
    assert(matrix_value_equals(view, 0, 0, 6.0));
    assert(matrix_value_equals(view, 0, 2, 4.0));
    assert(matrix_value_equals(view, 1, 0, 3.0));
    assert(matrix_value_equals(view, 1, 2, 1.0));
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_rotate_90_counterclockwise(parent, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(view) == 3);
    assert(numerus_matrix_columns(view) == 2);
    assert(matrix_value_equals(view, 0, 0, 3.0));
    assert(matrix_value_equals(view, 0, 1, 6.0));
    assert(matrix_value_equals(view, 1, 0, 2.0));
    assert(matrix_value_equals(view, 1, 1, 5.0));
    assert(matrix_value_equals(view, 2, 0, 1.0));
    assert(matrix_value_equals(view, 2, 1, 4.0));
    numerus_matrix_destroy(view);

    /* Rotation of a transpose composes as a horizontal flip. */
    assert(numerus_matrix_create_transpose(parent, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    {
        numerus_matrix *rotated = NULL;

        assert(numerus_matrix_create_rotate_90_clockwise(view, &rotated) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_rows(rotated) == 2);
        assert(numerus_matrix_columns(rotated) == 3);
        for (size_t row = 0; row < 2; row++) {
            for (size_t column = 0; column < 3; column++) {
                assert(matrix_value_equals(
                    rotated,
                    row,
                    column,
                    values[row * 3 + (2 - column)]
                ));
            }
        }
        numerus_matrix_destroy(rotated);
    }
    numerus_matrix_destroy(view);
    numerus_matrix_destroy(parent);

    assert(numerus_matrix_create_flip_rows(NULL, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_flip_columns(NULL, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_rotate_90_clockwise(NULL, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_rotate_180(NULL, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_rotate_90_counterclockwise(NULL, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
}

static void test_value_transform(void)
{
    numerus_matrix *parent = NULL;
    numerus_matrix *child = NULL;

    assert(numerus_matrix_create_identity(4, &parent) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        parent, 4, 4, NULL, diagonal_index_values, NULL, &child
    ) == NUMERUS_MATRIX_SUCCESS);

    for (size_t i = 0; i < 4; i++) {
        assert(matrix_value_equals(child, i, i, (double) (i + 1)));
    }

    assert(matrix_value_equals(child, 0, 1, 0.0));
    assert(matrix_value_equals(child, 1, 3, 0.0));
    assert(matrix_value_equals(child, 3, 0, 0.0));

    numerus_matrix_destroy(child);
    numerus_matrix_destroy(parent);
}

static void test_transform_error_propagation(void)
{
    numerus_matrix *parent = NULL;
    numerus_matrix *child = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_zero(2, 2, &parent) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_from_parent_with_transforms(
        parent, 2, 2, invalid_coordinates, NULL, NULL, &child
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(child, 0, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 123.0);
    numerus_matrix_destroy(child);

    assert(numerus_matrix_create_from_parent_with_transforms(
        parent, 2, 2, out_of_bounds_coordinates, NULL, NULL, &child
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(child, 0, 0, &value) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(value == 123.0);
    numerus_matrix_destroy(child);

    assert(numerus_matrix_create_from_parent_with_transforms(
        parent, 2, 2, NULL, invalid_value, NULL, &child
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(child, 0, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 123.0);
    numerus_matrix_destroy(child);

    numerus_matrix_destroy(parent);
}

static void test_matrix_joins(void)
{
    const double left_values[] = {1, 2, 3, 4};
    const double right_values[] = {5, 6};
    const double bottom_values[] = {7, 8};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    numerus_matrix *bottom = NULL;
    numerus_matrix *joined = NULL;
    numerus_matrix *derived = NULL;
    numerus_matrix *nested = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_dense(2, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 2, bottom_values, &bottom) ==
        NUMERUS_MATRIX_SUCCESS);

    /* Horizontal join places the second parent to the right. */
    assert(numerus_matrix_create_join_horizontal(left, right, &joined) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(joined) == 2);
    assert(numerus_matrix_columns(joined) == 3);
    assert(matrix_value_equals(joined, 0, 0, 1.0));
    assert(matrix_value_equals(joined, 0, 1, 2.0));
    assert(matrix_value_equals(joined, 0, 2, 5.0));
    assert(matrix_value_equals(joined, 1, 0, 3.0));
    assert(matrix_value_equals(joined, 1, 1, 4.0));
    assert(matrix_value_equals(joined, 1, 2, 6.0));
    assert(numerus_matrix_get(joined, 0, 3, &value) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(value == 123.0);

    /* A derived view can use a joined Matrix as its parent. */
    assert(numerus_matrix_create_transpose(joined, &derived) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(derived) == 3);
    assert(numerus_matrix_columns(derived) == 2);
    assert(matrix_value_equals(derived, 2, 0, 5.0));
    assert(matrix_value_equals(derived, 2, 1, 6.0));
    numerus_matrix_destroy(derived);
    derived = NULL;

    /* Nested joins remain lazy and preserve the left-to-right layout. */
    assert(numerus_matrix_create_dense(2, 1, (const double[]){9, 10}, &derived) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_join_horizontal(joined, derived, &nested) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(nested) == 2);
    assert(numerus_matrix_columns(nested) == 4);
    assert(matrix_value_equals(nested, 0, 3, 9.0));
    assert(matrix_value_equals(nested, 1, 3, 10.0));
    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(derived);
    numerus_matrix_destroy(joined);
    joined = NULL;
    derived = NULL;
    nested = NULL;

    /* Vertical join places the second parent below the first. */
    assert(numerus_matrix_create_join_vertical(left, bottom, &joined) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(joined) == 3);
    assert(numerus_matrix_columns(joined) == 2);
    assert(matrix_value_equals(joined, 0, 0, 1.0));
    assert(matrix_value_equals(joined, 1, 1, 4.0));
    assert(matrix_value_equals(joined, 2, 0, 7.0));
    assert(matrix_value_equals(joined, 2, 1, 8.0));
    numerus_matrix_destroy(joined);

    /* Incompatible dimensions are rejected without returning an object. */
    assert(numerus_matrix_create_join_horizontal(left, bottom, &joined) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(joined == NULL);
    assert(numerus_matrix_create_join_vertical(left, right, &joined) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(joined == NULL);
    assert(numerus_matrix_create_join_horizontal(NULL, right, &joined) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(joined == NULL);
    assert(numerus_matrix_create_join_vertical(left, right, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    /* Logical dimensions let us exercise addition overflow without huge data. */
    {
        numerus_matrix *huge = NULL;

        assert(numerus_matrix_create_from_parent(
            left, (size_t) -1, 2, &huge
        ) == NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_join_vertical(huge, left, &joined) ==
            NUMERUS_MATRIX_OVERFLOW);
        assert(joined == NULL);
        numerus_matrix_destroy(huge);
    }

    numerus_matrix_destroy(bottom);
    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}

static void test_remove_row_and_column_transforms(void)
{
    const double source[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    const double remove_middle_row[] = {1, 2, 3, 7, 8, 9};
    const double remove_first_row[] = {4, 5, 6, 7, 8, 9};
    const double remove_last_row[] = {1, 2, 3, 4, 5, 6};
    const double remove_middle_column[] = {1, 3, 4, 6, 7, 9};
    const double remove_first_column[] = {2, 3, 5, 6, 8, 9};
    const double remove_last_column[] = {1, 2, 4, 5, 7, 8};
    numerus_matrix *root = NULL;
    numerus_matrix *view = NULL;

    assert(numerus_matrix_create_dense(3, 3, source, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_remove_row(root, 1, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 2, 3, remove_middle_row);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_remove_row(root, 0, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 2, 3, remove_first_row);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_remove_row(root, 2, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 2, 3, remove_last_row);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_remove_column(root, 1, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 2, remove_middle_column);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_remove_column(root, 0, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 2, remove_first_column);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_remove_column(root, 2, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 2, remove_last_column);
    numerus_matrix_destroy(view);

    /* The views must not change the immutable parent. */
    assert_matrix_values(root, 3, 3, source);

    view = root;
    assert(numerus_matrix_create_remove_row(root, 3, &view) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(view == NULL);
    view = root;
    assert(numerus_matrix_create_remove_column(root, 3, &view) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(view == NULL);

    {
        const double one_row_values[] = {1, 2};
        const double one_column_values[] = {1, 2};
        numerus_matrix *one_row = NULL;
        numerus_matrix *one_column = NULL;

        assert(numerus_matrix_create_dense(1, 2, one_row_values, &one_row) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_remove_row(one_row, 0, &view) ==
            NUMERUS_MATRIX_INVALID_ARGUMENT);
        assert(view == NULL);

        assert(numerus_matrix_create_dense(2, 1, one_column_values, &one_column) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_remove_column(one_column, 0, &view) ==
            NUMERUS_MATRIX_INVALID_ARGUMENT);
        assert(view == NULL);

        numerus_matrix_destroy(one_column);
        numerus_matrix_destroy(one_row);
    }

    assert(numerus_matrix_create_remove_row(NULL, 0, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_remove_column(root, 0, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(root);
}

static void test_swap_row_and_column_transforms(void)
{
    const double source[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    const double swap_rows[] = {
        7, 8, 9,
        4, 5, 6,
        1, 2, 3
    };
    const double swap_columns[] = {
        3, 2, 1,
        6, 5, 4,
        9, 8, 7
    };
    numerus_matrix *root = NULL;
    numerus_matrix *view = NULL;

    assert(numerus_matrix_create_dense(3, 3, source, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_swap_rows(root, 0, 2, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 3, swap_rows);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_swap_columns(root, 0, 2, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 3, swap_columns);
    numerus_matrix_destroy(view);

    /* Swapping an index with itself is a valid identity view. */
    assert(numerus_matrix_create_swap_rows(root, 1, 1, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 3, source);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_swap_columns(root, 2, 2, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 3, source);
    numerus_matrix_destroy(view);

    /* Index validation and output initialization are part of the API contract. */
    view = root;
    assert(numerus_matrix_create_swap_rows(root, 0, 3, &view) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(view == NULL);
    view = root;
    assert(numerus_matrix_create_swap_columns(root, 3, 0, &view) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(view == NULL);
    assert(numerus_matrix_create_swap_rows(NULL, 0, 1, &view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_swap_columns(root, 0, 1, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    /* Swapping views are lazy and leave their parent unchanged. */
    assert_matrix_values(root, 3, 3, source);
    numerus_matrix_destroy(root);
}

static void test_validation(void)
{
    numerus_matrix *matrix = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_zero(0, 3, &matrix) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_create_zero(3, 0, &matrix) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create(
        NUMERUS_STORAGE_IDENTITY, 2, 3, NULL, &matrix
    ) == NUMERUS_MATRIX_NOT_SQUARE);

    assert(numerus_matrix_create_from_parent(NULL, 2, 2, &matrix) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_from_parent(
        NULL, 2, 2, NULL
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_zero(2, 2, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(matrix, 2, 0, &value) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(value == 123.0);

    assert(numerus_matrix_get(matrix, 0, 2, &value) ==
        NUMERUS_MATRIX_OUT_OF_BOUNDS);
    assert(numerus_matrix_get(matrix, 0, 0, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numerus_matrix_get(NULL, 0, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(matrix);
}


static void assert_matrix_values(
    const numerus_matrix *matrix,
    size_t rows,
    size_t columns,
    const double *expected
)
{
    assert(numerus_matrix_rows(matrix) == rows);
    assert(numerus_matrix_columns(matrix) == columns);

    for (size_t row = 0; row < rows; row++) {
        for (size_t column = 0; column < columns; column++) {
            assert(matrix_value_equals(
                matrix,
                row,
                column,
                expected[row * columns + column]
            ));
        }
    }
}

/*
 * Check every coordinate, not just representative corners. Rectangular
 * input is important because swapped dimensions expose mapping mistakes.
 */
static void test_orientation_transforms_exhaustively(void)
{
    const double source[] = {1, 2, 3, 4, 5, 6};
    const double transpose[] = {1, 4, 2, 5, 3, 6};
    const double flip_rows[] = {4, 5, 6, 1, 2, 3};
    const double flip_columns[] = {3, 2, 1, 6, 5, 4};
    const double rotate_clockwise[] = {4, 1, 5, 2, 6, 3};
    const double rotate_180[] = {6, 5, 4, 3, 2, 1};
    const double rotate_counterclockwise[] = {3, 6, 2, 5, 1, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *view = NULL;

    assert(numerus_matrix_create_dense(2, 3, source, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_transpose(root, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 2, transpose);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_flip_rows(root, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 2, 3, flip_rows);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_flip_columns(root, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 2, 3, flip_columns);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_rotate_90_clockwise(root, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 2, rotate_clockwise);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_rotate_180(root, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 2, 3, rotate_180);
    numerus_matrix_destroy(view);

    assert(numerus_matrix_create_rotate_90_counterclockwise(root, &view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_matrix_values(view, 3, 2, rotate_counterclockwise);
    numerus_matrix_destroy(view);
    numerus_matrix_destroy(root);

    /* Degenerate rectangular dimensions exercise zero-offset boundaries. */
    {
        const double row_values[] = {7, 8, 9};
        const double row_rotated[] = {7, 8, 9};
        const double column_values[] = {7, 8, 9};
        const double column_rotated[] = {9, 8, 7};
        numerus_matrix *row = NULL;
        numerus_matrix *column = NULL;

        assert(numerus_matrix_create_dense(1, 3, row_values, &row) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_rotate_90_clockwise(row, &view) ==
            NUMERUS_MATRIX_SUCCESS);
        assert_matrix_values(view, 3, 1, row_rotated);
        numerus_matrix_destroy(view);

        assert(numerus_matrix_create_dense(3, 1, column_values, &column) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_rotate_90_clockwise(column, &view) ==
            NUMERUS_MATRIX_SUCCESS);
        assert_matrix_values(view, 1, 3, column_rotated);
        numerus_matrix_destroy(view);

        numerus_matrix_destroy(column);
        numerus_matrix_destroy(row);
    }
}

static void test_join_error_propagation_and_horizontal_overflow(void)
{
    const double values[] = {1, 2};
    numerus_matrix *root = NULL;
    numerus_matrix *valid = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *joined = NULL;
    numerus_matrix *huge = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_dense(2, 1, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent(
        root, 2, 1, &valid
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 1, invalid_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    /* Errors from either side of a join must propagate without writing value. */
    assert(numerus_matrix_create_join_horizontal(valid, failing, &joined) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(joined, 0, 1, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 123.0);
    assert(numerus_matrix_get(joined, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.0);
    numerus_matrix_destroy(joined);
    joined = NULL;

    assert(numerus_matrix_create_join_horizontal(failing, valid, &joined) ==
        NUMERUS_MATRIX_SUCCESS);
    value = 123.0;
    assert(numerus_matrix_get(joined, 0, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 123.0);
    assert(numerus_matrix_get(joined, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 1.0);
    numerus_matrix_destroy(joined);

    /* The logical dimensions allow overflow tests without huge allocations. */
    assert(numerus_matrix_create_from_parent(
        root, 2, (size_t) -1, &huge
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_join_horizontal(huge, valid, &joined) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(joined == NULL);

    numerus_matrix_destroy(huge);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(valid);
    numerus_matrix_destroy(root);
}


static void test_epsilon_comparisons(void)
{
    assert(numerus_double_equals(1.0, 1.0 + 5e-10));
    assert(numerus_double_equals(1e12, 1e12 + 500.0));
    assert(!numerus_double_equals(1.0, 1.0 + 2e-9));
    assert(numerus_double_is_zero(5e-10));
    assert(!numerus_double_is_zero(2e-9));
    assert(numerus_double_is_one(1.0 + 5e-10));
    assert(!numerus_double_is_one(1.0 + 2e-9));
    assert(!numerus_double_equals(NAN, NAN));
    assert(numerus_double_equals(INFINITY, INFINITY));
    assert(!numerus_double_equals(INFINITY, -INFINITY));
    assert(!numerus_double_is_zero(INFINITY));
    assert(!numerus_double_is_one(NAN));
}

static void test_epsilon_matrix_flags(void)
{
    const double values[] = {
        1.0 + 5e-10, 2.0,
        2.0 + 5e-10, 5e-10
    };
    const double non_symmetric[] = {
        1.0, 2.0,
        2.0 + 2e-9, 1.0
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix_flags flags = {0};

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(flags.symmetric);
    assert(!flags.zero);
    assert(!flags.diagonal);
    assert(!flags.identity);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, non_symmetric, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!flags.symmetric);
    numerus_matrix_destroy(matrix);
}

static void test_flag_cache_and_shared_symmetry_scan(void)
{
    const double values[] = {
        1, 2, 3,
        2, 4, 5,
        3, 5, 6
    };
    numerus_matrix *root = NULL;
    numerus_matrix *matrix = NULL;
    numerus_matrix_flags flags = {0};
    size_t calls = 0;

    assert(numerus_matrix_create_dense(3, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 3, 3, counting_identity_coordinates, NULL, &calls, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get_flags(matrix, &flags) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(flags.square);
    assert(flags.symmetric);
    assert(!flags.zero);
    assert(!flags.diagonal);
    assert(!flags.upper_triangular);
    assert(!flags.lower_triangular);
    assert(!flags.identity);

    /*
     * Nine reads for the full scan plus one reverse-coordinate read for
     * each of the three off-diagonal pairs. The old separate symmetry pass
     * needed six additional reads.
     */
    assert(calls == 12);

    assert(numerus_matrix_get_flags(matrix, &flags) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(calls == 12);

    numerus_matrix_destroy(matrix);
    numerus_matrix_destroy(root);
}

static void test_cached_analysis(void)
{
    const double general[] = {1, 2, 3, 4};
    const double symmetric[] = {1, 2, 2, 1};
    const double upper[] = {2, 3, 4};
    const double singular[] = {1, 2, 2, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix_flags flags = {0};
    double determinant = 123.0;

    assert(numerus_matrix_create_dense(2, 2, general, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) == NUMERUS_MATRIX_SUCCESS);
    assert(flags.square);
    assert(!flags.zero);
    assert(!flags.diagonal);
    assert(!flags.symmetric);
    assert(!flags.upper_triangular);
    assert(!flags.lower_triangular);
    assert(!flags.identity);
    /* A second read must return the same cached property set. */
    {
        numerus_matrix_flags cached = {0};
        assert(numerus_matrix_get_flags(matrix, &cached) == NUMERUS_MATRIX_SUCCESS);
        assert(cached.square == flags.square);
        assert(cached.zero == flags.zero);
        assert(cached.diagonal == flags.diagonal);
        assert(cached.upper_triangular == flags.upper_triangular);
        assert(cached.lower_triangular == flags.lower_triangular);
        assert(cached.symmetric == flags.symmetric);
        assert(cached.identity == flags.identity);
    }
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == -2.0);
    determinant = 0.0;
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == -2.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, symmetric, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) == NUMERUS_MATRIX_SUCCESS);
    assert(flags.symmetric);
    assert(!flags.zero);
    assert(!flags.diagonal);
    assert(!flags.upper_triangular);
    assert(!flags.lower_triangular);
    assert(!flags.identity);
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == -3.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_upper_triangular(2, upper, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) == NUMERUS_MATRIX_SUCCESS);
    assert(flags.upper_triangular);
    assert(!flags.lower_triangular);
    assert(!flags.zero);
    assert(!flags.diagonal);
    assert(!flags.symmetric);
    assert(!flags.identity);
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == 8.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_zero(2, 2, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) == NUMERUS_MATRIX_SUCCESS);
    assert(flags.zero);
    assert(flags.diagonal);
    assert(flags.upper_triangular);
    assert(flags.lower_triangular);
    assert(flags.symmetric);
    assert(!flags.identity);
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == 0.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_identity(3, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) == NUMERUS_MATRIX_SUCCESS);
    assert(flags.identity);
    assert(flags.square);
    assert(flags.diagonal);
    assert(flags.upper_triangular);
    assert(flags.lower_triangular);
    assert(flags.symmetric);
    assert(!flags.zero);
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == 1.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, singular, &matrix) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_SUCCESS);
    assert(determinant == 0.0);
    numerus_matrix_destroy(matrix);

    {
        const double pivot_values[] = {0, 1, 1, 0};
        assert(numerus_matrix_create_dense(2, 2, pivot_values, &matrix) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_determinant(matrix, &determinant) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(determinant == -1.0);
        numerus_matrix_destroy(matrix);
    }

    {
        const double one_value[] = {7};
        numerus_matrix *root = NULL;
        numerus_matrix *failing = NULL;
        numerus_matrix_flags original_flags = {.square = true};

        assert(numerus_matrix_create_dense(1, 1, one_value, &root) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_from_parent_with_transforms(
            root, 1, 1, invalid_coordinates, NULL, NULL, &failing
        ) == NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_get_flags(failing, &original_flags) ==
            NUMERUS_MATRIX_INVALID_ARGUMENT);
        assert(original_flags.square);
        determinant = 123.0;
        assert(numerus_matrix_determinant(failing, &determinant) ==
            NUMERUS_MATRIX_INVALID_ARGUMENT);
        assert(determinant == 123.0);
        numerus_matrix_destroy(failing);
        numerus_matrix_destroy(root);
    }

    {
        const double one_value[] = {7};
        numerus_matrix *root = NULL;
        numerus_matrix *huge = NULL;

        assert(numerus_matrix_create_dense(1, 1, one_value, &root) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_create_from_parent(
            root, (size_t) -1, (size_t) -1, &huge
        ) == NUMERUS_MATRIX_SUCCESS);
        determinant = 123.0;
        assert(numerus_matrix_determinant(huge, &determinant) ==
            NUMERUS_MATRIX_OVERFLOW);
        assert(determinant == 123.0);
        numerus_matrix_destroy(huge);
        numerus_matrix_destroy(root);
    }

    {
        const double rectangular[] = {1, 2, 3, 4, 5, 6};
        assert(numerus_matrix_create_dense(2, 3, rectangular, &matrix) == NUMERUS_MATRIX_SUCCESS);
        determinant = 123.0;
        assert(numerus_matrix_determinant(matrix, &determinant) == NUMERUS_MATRIX_NOT_SQUARE);
        assert(determinant == 123.0);
        numerus_matrix_destroy(matrix);
    }

    {
        numerus_matrix_flags rectangular_flags = {0};

        assert(numerus_matrix_create_zero(2, 3, &matrix) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_get_flags(matrix, &rectangular_flags) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(!rectangular_flags.square);
        assert(rectangular_flags.zero);
        assert(!rectangular_flags.diagonal);
        assert(!rectangular_flags.upper_triangular);
        assert(!rectangular_flags.lower_triangular);
        assert(!rectangular_flags.symmetric);
        assert(!rectangular_flags.identity);
        numerus_matrix_destroy(matrix);
    }

    flags = (numerus_matrix_flags) {.square = true};
    assert(numerus_matrix_get_flags(NULL, &flags) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(flags.square);
    assert(numerus_matrix_determinant(NULL, &determinant) == NUMERUS_MATRIX_INVALID_ARGUMENT);
}

int main(void)
{
    test_factory_and_dimensions();
    test_convenience_constructors();
    test_all_storage_kinds();
    test_parent_chain();
    test_transpose_transform();
    test_transpose_constructor();
    test_orientation_transforms();
    test_orientation_transforms_exhaustively();
    test_remove_row_and_column_transforms();
    test_swap_row_and_column_transforms();
    test_value_transform();
    test_transform_error_propagation();
    test_matrix_joins();
    test_join_error_propagation_and_horizontal_overflow();
    test_validation();
    test_cached_analysis();
    test_flag_cache_and_shared_symmetry_scan();
    test_epsilon_comparisons();
    test_epsilon_matrix_flags();

    puts("Matrix tests passed.");
    return 0;
}
