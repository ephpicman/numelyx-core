/**
 * @file matrix_elementwise_test.c
 * @brief Native tests for elementwise Matrix operations, scalar broadcasting, and value propagation.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
static void check(const numerus_matrix *m, size_t r, size_t c, double expected) {
    double actual = 0.0;
    assert(numerus_matrix_get(m, r, c, &actual) == NUMERUS_MATRIX_SUCCESS);
    if (isnan(expected)) assert(isnan(actual)); else assert(actual == expected);
}
static void test_min_max(void) {
    const double a[] = {1, NAN, INFINITY, -INFINITY, 4};
    const double b[] = {2, 3, 5, INFINITY, 4};
    numerus_matrix *left = NULL, *right = NULL, *minimum = NULL, *maximum = NULL;
    assert(numerus_matrix_create_dense(1, 5, a, &left) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 5, b, &right) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_elementwise_min(left, right, &minimum) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_elementwise_max(left, right, &maximum) == NUMERUS_MATRIX_SUCCESS);
    check(minimum, 0, 0, 1); check(maximum, 0, 0, 2);
    check(minimum, 0, 1, NAN); check(maximum, 0, 1, NAN);
    check(minimum, 0, 2, 5); check(maximum, 0, 2, INFINITY);
    check(minimum, 0, 3, -INFINITY); check(maximum, 0, 3, INFINITY);
    check(minimum, 0, 4, 4); check(maximum, 0, 4, 4);
    check(left, 0, 0, 1);
    numerus_matrix_destroy(maximum); numerus_matrix_destroy(minimum);
    numerus_matrix_destroy(right); numerus_matrix_destroy(left);
}
static void test_clamp_and_validation(void) {
    const double values[] = {-INFINITY, -2, -0.0, 1.5, 9, INFINITY, NAN};
    numerus_matrix *source = NULL, *result = (numerus_matrix *) 1;
    assert(numerus_matrix_create_dense(1, 7, values, &source) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_clamp(source, -1, 2, &result) == NUMERUS_MATRIX_SUCCESS);
    check(result, 0, 0, -1); check(result, 0, 1, -1); check(result, 0, 2, -0.0);
    check(result, 0, 3, 1.5); check(result, 0, 4, 2); check(result, 0, 5, 2);
    check(result, 0, 6, NAN);
    numerus_matrix_destroy(result); result = (numerus_matrix *) 1;
    assert(numerus_matrix_create_clamp(source, 2, 1, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_clamp(source, NAN, 1, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_clamp(NULL, 0, 1, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_clamp(source, 0, 1, NULL) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    numerus_matrix_destroy(source);
}
static numerus_matrix_status reject_read(size_t row, size_t column,
    size_t *parent_row, size_t *parent_column, const void *context) {
    (void) row; (void) column; (void) parent_row; (void) parent_column; (void) context;
    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}
static void test_read_failures_and_shapes(void) {
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *source = NULL, *different = NULL, *failing = NULL, *result = NULL;
    assert(numerus_matrix_create_dense(2, 2, values, &source) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 4, values, &different) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_elementwise_min(source, different, &result) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(result == NULL);
    assert(numerus_matrix_create_from_parent_with_transforms(source, 2, 2,
        reject_read, NULL, NULL, &failing) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_elementwise_max(source, failing, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_clamp(failing, 0, 1, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_elementwise_min(NULL, source, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_elementwise_max(source, NULL, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_create_elementwise_min(source, source, NULL) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    numerus_matrix_destroy(failing); numerus_matrix_destroy(different); numerus_matrix_destroy(source);
}
int main(void) {
    test_min_max(); test_clamp_and_validation(); test_read_failures_and_shapes();
    puts("Matrix element-wise min/max/clamp tests passed."); return 0;
}