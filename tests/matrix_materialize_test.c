/**
 * @file matrix_materialize_test.c
 * @brief Native tests for materializing lazy Matrix views into independent dense storage.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static int matrix_value_equals(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double expected
)
{
    double actual = 0.0;

    return numerus_matrix_get(
        matrix, row, column, &actual
    ) == NUMERUS_MATRIX_SUCCESS && actual == expected;
}

static numerus_matrix_status fail_on_second_row(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) context;

    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_materialize_nested_views_and_joins(void)
{
    const double values[] = {1, 2, 3, 4};
    const double expected[] = {
        1, 3, 1, 2,
        2, 4, 3, 4
    };
    numerus_matrix *root = NULL;
    numerus_matrix *transpose = NULL;
    numerus_matrix *joined = NULL;
    numerus_matrix *copy = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_transpose(root, &transpose) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_join_horizontal(transpose, root, &joined) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_materialize(joined, &copy) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_rows(copy) == 2);
    assert(numerus_matrix_columns(copy) == 4);
    assert(numerus_matrix_storage_kind(copy) == NUMERUS_STORAGE_DENSE);
    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 4; column++) {
            assert(matrix_value_equals(
                copy, row, column, expected[row * 4 + column]
            ));
        }
    }

    /* The materialized copy no longer depends on the source or its parents. */
    numerus_matrix_destroy(joined);
    numerus_matrix_destroy(transpose);
    numerus_matrix_destroy(root);
    assert(matrix_value_equals(copy, 0, 1, 3.0));
    assert(matrix_value_equals(copy, 1, 3, 4.0));
    numerus_matrix_destroy(copy);
}

static void test_materialize_binary_node(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *sum = NULL;
    numerus_matrix *copy = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_add(root, root, &sum) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_materialize(sum, &copy) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_storage_kind(copy) == NUMERUS_STORAGE_DENSE);
    assert(matrix_value_equals(copy, 0, 0, 2.0));
    assert(matrix_value_equals(copy, 0, 1, 4.0));
    assert(matrix_value_equals(copy, 1, 0, 6.0));
    assert(matrix_value_equals(copy, 1, 1, 8.0));

    numerus_matrix_destroy(copy);
    numerus_matrix_destroy(sum);
    numerus_matrix_destroy(root);
}

static void test_materialize_failures(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *huge = NULL;
    numerus_matrix *copy = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_second_row, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    copy = root;
    assert(numerus_matrix_materialize(failing, &copy) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(copy == NULL);

    assert(numerus_matrix_create_from_parent(
        root, SIZE_MAX, 2, &huge
    ) == NUMERUS_MATRIX_SUCCESS);
    copy = root;
    assert(numerus_matrix_materialize(huge, &copy) ==
        NUMERUS_MATRIX_OVERFLOW);
    assert(copy == NULL);

    assert(numerus_matrix_materialize(NULL, &copy) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(copy == NULL);
    assert(numerus_matrix_materialize(root, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(huge);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}


static void test_materialize_rectangular_and_degenerate_view_shapes(void)
{
    const double values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *root = NULL;
    numerus_matrix *transpose = NULL;
    numerus_matrix *nested = NULL;
    numerus_matrix *row_view = NULL;
    numerus_matrix *column_view = NULL;
    numerus_matrix *materialized = NULL;
    size_t row;
    size_t column;

    assert(numerus_matrix_create_dense(2, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_transpose(root, &transpose) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_slice(
        transpose, 1, 2, 0, 2, &nested
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_materialize(nested, &materialized) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(materialized) == 2);
    assert(numerus_matrix_columns(materialized) == 2);

    for (row = 0; row < 2; row++) {
        for (column = 0; column < 2; column++) {
            double view_value;
            double copy_value;

            assert(numerus_matrix_get(nested, row, column, &view_value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(numerus_matrix_get(materialized, row, column, &copy_value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(copy_value == view_value);
        }
    }

    assert(numerus_matrix_create_slice(
        root, 1, 1, 0, 3, &row_view
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_materialize(row_view, &materialized) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(materialized) == 1);
    assert(numerus_matrix_columns(materialized) == 3);
    numerus_matrix_destroy(materialized);
    materialized = NULL;

    assert(numerus_matrix_create_slice(
        transpose, 0, 3, 1, 1, &column_view
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_materialize(column_view, &materialized) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(materialized) == 3);
    assert(numerus_matrix_columns(materialized) == 1);
    assert(matrix_value_equals(materialized, 0, 0, 4.0));
    assert(matrix_value_equals(materialized, 1, 0, 5.0));
    assert(matrix_value_equals(materialized, 2, 0, 6.0));

    numerus_matrix_destroy(materialized);
    numerus_matrix_destroy(column_view);
    numerus_matrix_destroy(row_view);
    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(transpose);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_materialize_nested_views_and_joins();
    test_materialize_binary_node();
    test_materialize_failures();
    test_materialize_rectangular_and_degenerate_view_shapes();
    puts("Matrix materialization tests passed.");
    return 0;
}
