#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <stdio.h>

static void test_binary_view_retains_both_parents(void)
{
    const double left_values[] = {1.0, 2.0};
    const double right_values[] = {10.0, 20.0};
    numelyx_core_matrix *left = NULL;
    numelyx_core_matrix *right = NULL;
    numelyx_core_matrix *sum = NULL;
    double value = 0.0;

    assert(numelyx_core_matrix_create_dense(1, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_add(left, right, &sum) ==
        NUMERUS_MATRIX_SUCCESS);

    numelyx_core_matrix_destroy(left);
    numelyx_core_matrix_destroy(right);

    assert(numelyx_core_matrix_get(sum, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 22.0);
    numelyx_core_matrix_destroy(sum);
}

static void test_join_retains_both_parents(void)
{
    const double left_values[] = {3.0, 4.0};
    const double right_values[] = {5.0, 6.0};
    numelyx_core_matrix *left = NULL;
    numelyx_core_matrix *right = NULL;
    numelyx_core_matrix *joined = NULL;
    double value = 0.0;

    assert(numelyx_core_matrix_create_dense(2, 1, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(2, 1, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_join_horizontal(left, right, &joined) ==
        NUMERUS_MATRIX_SUCCESS);

    numelyx_core_matrix_destroy(left);
    numelyx_core_matrix_destroy(right);

    assert(numelyx_core_matrix_get(joined, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 6.0);
    numelyx_core_matrix_destroy(joined);
}

static void test_nested_views_retain_the_full_parent_chain(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *first = NULL;
    numelyx_core_matrix *second = NULL;
    double value = 0.0;

    assert(numelyx_core_matrix_create_dense(2, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_transpose(root, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_transpose(first, &second) ==
        NUMERUS_MATRIX_SUCCESS);

    numelyx_core_matrix_destroy(root);
    numelyx_core_matrix_destroy(first);

    assert(numelyx_core_matrix_get(second, 1, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 6.0);
    numelyx_core_matrix_destroy(second);
}

static void test_same_parent_can_be_retained_twice(void)
{
    const double values[] = {2.0, 3.0};
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *sum = NULL;
    double value = 0.0;

    assert(numelyx_core_matrix_create_dense(1, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_add(root, root, &sum) ==
        NUMERUS_MATRIX_SUCCESS);

    numelyx_core_matrix_destroy(root);
    assert(numelyx_core_matrix_get(sum, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 6.0);
    numelyx_core_matrix_destroy(sum);
}

int main(void)
{
    test_binary_view_retains_both_parents();
    test_join_retains_both_parents();
    test_nested_views_retain_the_full_parent_chain();
    test_same_parent_can_be_retained_twice();
    puts("Matrix parent-lifetime tests passed.");
    return 0;
}
