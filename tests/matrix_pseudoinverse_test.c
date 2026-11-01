/**
 * @file matrix_pseudoinverse_test.c
 * @brief Native tests for Moore–Penrose pseudoinverse identities and rank-deficient inputs.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void assert_close(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    double tolerance
)
{
    size_t row;
    size_t column;
    assert(numelyx_core_matrix_rows(left) == numelyx_core_matrix_rows(right));
    assert(numelyx_core_matrix_columns(left) == numelyx_core_matrix_columns(right));
    for (row = 0; row < numelyx_core_matrix_rows(left); row++) {
        for (column = 0; column < numelyx_core_matrix_columns(left); column++) {
            double a;
            double b;
            assert(numelyx_core_matrix_get(left, row, column, &a) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(numelyx_core_matrix_get(right, row, column, &b) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(fabs(a - b) <= tolerance);
        }
    }
}

static numelyx_core_matrix *multiply(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right
)
{
    numelyx_core_matrix *result = NULL;
    assert(numelyx_core_matrix_multiply(left, right, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    return result;
}

static numelyx_core_matrix *transpose(const numelyx_core_matrix *matrix)
{
    numelyx_core_matrix *result = NULL;
    assert(numelyx_core_matrix_create_transpose(
        (numelyx_core_matrix *) matrix, &result
    ) == NUMERUS_MATRIX_SUCCESS);
    return result;
}

static void test_rank_deficient_square_and_penrose_conditions(void)
{
    const double input[] = {1, 2, 2, 4};
    const double expected[] = {0.04, 0.08, 0.08, 0.16};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *inverse = NULL;
    numelyx_core_matrix *expected_matrix = NULL;
    numelyx_core_matrix *aa_plus = NULL;
    numelyx_core_matrix *a_plus_a = NULL;
    numelyx_core_matrix *aa_plus_a = NULL;
    numelyx_core_matrix *a_plus_a_a_plus = NULL;
    numelyx_core_matrix *aa_plus_t = NULL;
    numelyx_core_matrix *a_plus_a_t = NULL;

    assert(numelyx_core_matrix_create_dense(2, 2, input, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(2, 2, expected, &expected_matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_pseudoinverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_close(inverse, expected_matrix, 1e-8);

    aa_plus = multiply(matrix, inverse);
    a_plus_a = multiply(inverse, matrix);
    aa_plus_a = multiply(aa_plus, matrix);
    a_plus_a_a_plus = multiply(a_plus_a, inverse);
    aa_plus_t = transpose(aa_plus);
    a_plus_a_t = transpose(a_plus_a);

    /* A A+ A = A; A+ A A+ = A+; both projectors are symmetric. */
    assert_close(aa_plus_a, matrix, 1e-8);
    assert_close(a_plus_a_a_plus, inverse, 1e-8);
    assert_close(aa_plus, aa_plus_t, 1e-8);
    assert_close(a_plus_a, a_plus_a_t, 1e-8);

    numelyx_core_matrix_destroy(a_plus_a_t);
    numelyx_core_matrix_destroy(aa_plus_t);
    numelyx_core_matrix_destroy(a_plus_a_a_plus);
    numelyx_core_matrix_destroy(aa_plus_a);
    numelyx_core_matrix_destroy(a_plus_a);
    numelyx_core_matrix_destroy(aa_plus);
    numelyx_core_matrix_destroy(expected_matrix);
    numelyx_core_matrix_destroy(inverse);
    numelyx_core_matrix_destroy(matrix);
}

static void test_rectangular_and_zero_inputs(void)
{
    const double tall_values[] = {1, 0, 0, 1, 1, 1};
    const double zero_values[] = {0, 0, 0, 0, 0, 0};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *inverse = NULL;
    size_t row;
    size_t column;
    double value;

    assert(numelyx_core_matrix_create_dense(3, 2, tall_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_pseudoinverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(inverse) == 2);
    assert(numelyx_core_matrix_columns(inverse) == 3);
    {
        numelyx_core_matrix *aa_plus = multiply(matrix, inverse);
        numelyx_core_matrix *a_plus_a = multiply(inverse, matrix);
        numelyx_core_matrix *aa_plus_a = multiply(aa_plus, matrix);
        numelyx_core_matrix *a_plus_a_a_plus = multiply(a_plus_a, inverse);
        assert_close(aa_plus_a, matrix, 1e-8);
        assert_close(a_plus_a_a_plus, inverse, 1e-8);
        numelyx_core_matrix_destroy(a_plus_a_a_plus);
        numelyx_core_matrix_destroy(aa_plus_a);
        numelyx_core_matrix_destroy(a_plus_a);
        numelyx_core_matrix_destroy(aa_plus);
    }
    numelyx_core_matrix_destroy(inverse);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 3, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_pseudoinverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(inverse) == 3);
    assert(numelyx_core_matrix_columns(inverse) == 2);
    for (row = 0; row < 3; row++) {
        for (column = 0; column < 2; column++) {
            assert(numelyx_core_matrix_get(inverse, row, column, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(value == 0.0);
        }
    }
    numelyx_core_matrix_destroy(inverse);
    numelyx_core_matrix_destroy(matrix);
}

static void test_failure_contracts(void)
{
    const double nonfinite_values[] = {1, NAN, 0, 1};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *inverse = (void *) 1;

    assert(numelyx_core_matrix_pseudoinverse(NULL, &inverse) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(inverse == NULL);
    assert(numelyx_core_matrix_pseudoinverse(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numelyx_core_matrix_create_dense(2, 2, nonfinite_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_pseudoinverse(matrix, &inverse) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(inverse == NULL);
    numelyx_core_matrix_destroy(matrix);
}


static void test_wide_rank_deficient_penrose_conditions(void)
{
    const double input[] = {
        1, 2, 3,
        2, 4, 6
    };
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *inverse = NULL;
    numelyx_core_matrix *aa_plus = NULL;
    numelyx_core_matrix *a_plus_a = NULL;
    numelyx_core_matrix *aa_plus_a = NULL;
    numelyx_core_matrix *a_plus_a_a_plus = NULL;
    numelyx_core_matrix *aa_plus_t = NULL;
    numelyx_core_matrix *a_plus_a_t = NULL;

    assert(numelyx_core_matrix_create_dense(2, 3, input, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_pseudoinverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(inverse) == 3);
    assert(numelyx_core_matrix_columns(inverse) == 2);

    aa_plus = multiply(matrix, inverse);
    a_plus_a = multiply(inverse, matrix);
    aa_plus_a = multiply(aa_plus, matrix);
    a_plus_a_a_plus = multiply(a_plus_a, inverse);
    aa_plus_t = transpose(aa_plus);
    a_plus_a_t = transpose(a_plus_a);

    /* Verify all four Moore-Penrose conditions on a wide, rank-deficient A. */
    assert_close(aa_plus_a, matrix, 1e-8);
    assert_close(a_plus_a_a_plus, inverse, 1e-8);
    assert_close(aa_plus, aa_plus_t, 1e-8);
    assert_close(a_plus_a, a_plus_a_t, 1e-8);

    numelyx_core_matrix_destroy(a_plus_a_t);
    numelyx_core_matrix_destroy(aa_plus_t);
    numelyx_core_matrix_destroy(a_plus_a_a_plus);
    numelyx_core_matrix_destroy(aa_plus_a);
    numelyx_core_matrix_destroy(a_plus_a);
    numelyx_core_matrix_destroy(aa_plus);
    numelyx_core_matrix_destroy(inverse);
    numelyx_core_matrix_destroy(matrix);
}


static void test_tall_rank_deficient_penrose_conditions(void)
{
    const double input[] = {
        1, 2,
        2, 4,
        3, 6
    };
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *inverse = NULL;
    numelyx_core_matrix *aa_plus = NULL;
    numelyx_core_matrix *a_plus_a = NULL;
    numelyx_core_matrix *aa_plus_a = NULL;
    numelyx_core_matrix *a_plus_a_a_plus = NULL;
    numelyx_core_matrix *aa_plus_t = NULL;
    numelyx_core_matrix *a_plus_a_t = NULL;

    assert(numelyx_core_matrix_create_dense(3, 2, input, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_pseudoinverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rows(inverse) == 2);
    assert(numelyx_core_matrix_columns(inverse) == 3);

    aa_plus = multiply(matrix, inverse);
    a_plus_a = multiply(inverse, matrix);
    aa_plus_a = multiply(aa_plus, matrix);
    a_plus_a_a_plus = multiply(a_plus_a, inverse);
    aa_plus_t = transpose(aa_plus);
    a_plus_a_t = transpose(a_plus_a);

    /* Verify all four Moore-Penrose conditions for tall, rank-deficient A. */
    assert_close(aa_plus_a, matrix, 1e-8);
    assert_close(a_plus_a_a_plus, inverse, 1e-8);
    assert_close(aa_plus, aa_plus_t, 1e-8);
    assert_close(a_plus_a, a_plus_a_t, 1e-8);

    numelyx_core_matrix_destroy(a_plus_a_t);
    numelyx_core_matrix_destroy(aa_plus_t);
    numelyx_core_matrix_destroy(a_plus_a_a_plus);
    numelyx_core_matrix_destroy(aa_plus_a);
    numelyx_core_matrix_destroy(a_plus_a);
    numelyx_core_matrix_destroy(aa_plus);
    numelyx_core_matrix_destroy(inverse);
    numelyx_core_matrix_destroy(matrix);
}

int main(void)
{
    test_rank_deficient_square_and_penrose_conditions();
    test_rectangular_and_zero_inputs();
    test_wide_rank_deficient_penrose_conditions();
    test_tall_rank_deficient_penrose_conditions();
    test_failure_contracts();
    puts("Matrix pseudoinverse tests passed.");
    return 0;
}
