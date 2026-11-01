/**
 * @file matrix_vector_constructor_test.c
 * @brief Native tests for row-vector and column-vector constructors.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <stdio.h>

static void test_row_vector(void)
{
    double values[] = {2.5, -1.0, 7.0};
    numerus_matrix *matrix = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_row_vector(3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(matrix) == 1);
    assert(numerus_matrix_columns(matrix) == 3);
    assert(numerus_matrix_storage_kind(matrix) == NUMERUS_STORAGE_DENSE);

    assert(numerus_matrix_get(matrix, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.5);
    assert(numerus_matrix_get(matrix, 0, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == -1.0);
    assert(numerus_matrix_get(matrix, 0, 2, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == 7.0);

    /* The constructor copies input values into Storage. */
    values[0] = 100.0;
    assert(numerus_matrix_get(matrix, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.5);
    numerus_matrix_destroy(matrix);
}

static void test_column_vector(void)
{
    const double values[] = {4.0, 0.0, -9.5};
    numerus_matrix *matrix = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_column_vector(3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(matrix) == 3);
    assert(numerus_matrix_columns(matrix) == 1);
    assert(numerus_matrix_storage_kind(matrix) == NUMERUS_STORAGE_DENSE);

    assert(numerus_matrix_get(matrix, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == 4.0);
    assert(numerus_matrix_get(matrix, 1, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == 0.0);
    assert(numerus_matrix_get(matrix, 2, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(value == -9.5);

    numerus_matrix_destroy(matrix);
}

static void test_invalid_vector_inputs(void)
{
    const double values[] = {1.0};
    numerus_matrix *matrix = NULL;

    assert(numerus_matrix_create_row_vector(0, values, &matrix) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(matrix == NULL);
    assert(numerus_matrix_create_column_vector(0, values, &matrix) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(matrix == NULL);
    assert(numerus_matrix_create_row_vector(1, NULL, &matrix) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(matrix == NULL);
    assert(numerus_matrix_create_column_vector(1, values, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
}

int main(void)
{
    test_row_vector();
    test_column_vector();
    test_invalid_vector_inputs();
    puts("Matrix vector constructor tests passed.");
    return 0;
}
