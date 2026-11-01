/**
 * @file matrix_map_test.c
 * @brief Native tests for lazy Matrix mapping and eager callback application.
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
static numerus_matrix_status scale_value(size_t row, size_t column,
    double value, double *result, const void *context) {
    const double factor = *(const double *) context;
    (void) row; (void) column;
    *result = value * factor;
    return NUMERUS_MATRIX_SUCCESS;
}
static numerus_matrix_status coordinate_offset(size_t row, size_t column,
    double value, double *result, const void *context) {
    const double offset = *(const double *) context;
    *result = value + offset + (double) row + (double) column;
    return NUMERUS_MATRIX_SUCCESS;
}
static numerus_matrix_status fail_on_second_row(size_t row, size_t column,
    double value, double *result, const void *context) {
    (void) column; (void) context;
    if (row == 1) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *result = value;
    return NUMERUS_MATRIX_SUCCESS;
}
static void check(const numerus_matrix *m, size_t row, size_t column, double expected) {
    double actual = 0.0;
    assert(numerus_matrix_get(m, row, column, &actual) == NUMERUS_MATRIX_SUCCESS);
    assert(actual == expected);
}
static void test_lazy_map_and_eager_apply(void) {
    const double values[] = {1, 2, 3, 4};
    const double factor = 3.0, offset = 10.0;
    numerus_matrix *source = NULL, *mapped = NULL, *applied = NULL;
    assert(numerus_matrix_create_dense(2, 2, values, &source) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_map(source, scale_value, &factor, &mapped) == NUMERUS_MATRIX_SUCCESS);
    check(mapped, 0, 0, 3); check(mapped, 1, 1, 12);
    assert(numerus_matrix_apply(source, coordinate_offset, &offset, &applied) == NUMERUS_MATRIX_SUCCESS);
    check(applied, 0, 0, 11); check(applied, 0, 1, 13);
    check(applied, 1, 0, 14); check(applied, 1, 1, 16);
    numerus_matrix_destroy(mapped);
    numerus_matrix_destroy(source);
    /* apply() materializes; its result is independent of the source lifetime. */
    check(applied, 1, 1, 16);
    numerus_matrix_destroy(applied);
}
static void test_failures(void) {
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *source = NULL, *result = (numerus_matrix *) 1;
    assert(numerus_matrix_create_dense(2, 2, values, &source) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_map(source, NULL, NULL, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    result = (numerus_matrix *) 1;
    assert(numerus_matrix_create_map(NULL, scale_value, NULL, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_apply(source, NULL, NULL, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_apply(source, fail_on_second_row, NULL, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);
    assert(numerus_matrix_apply(source, scale_value, NULL, NULL) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    numerus_matrix_destroy(source);
}

static void test_lazy_map_callback_failure_preserves_output(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *source = NULL;
    numerus_matrix *mapped = NULL;
    double value = 987.0;

    assert(numerus_matrix_create_dense(2, 2, values, &source) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_map(
        source, fail_on_second_row, NULL, &mapped
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(mapped, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.0);

    value = 987.0;
    assert(numerus_matrix_get(mapped, 1, 0, &value) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(value == 987.0);

    /* The lazy view remains usable at coordinates whose callback succeeds. */
    assert(numerus_matrix_get(mapped, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 1.0);

    numerus_matrix_destroy(mapped);
    numerus_matrix_destroy(source);
}

int main(void) {
    test_lazy_map_and_eager_apply(); test_failures();
    test_lazy_map_callback_failure_preserves_output();
    puts("Matrix map/apply tests passed."); return 0;
}