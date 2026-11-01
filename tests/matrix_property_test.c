/**
 * @file matrix_property_test.c
 * @brief Property-based Matrix algebra tests checking identities across generated inputs.
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

static double at(const numelyx_core_matrix *matrix, size_t row, size_t column)
{
    double value = NAN;
    assert(numelyx_core_matrix_get(matrix, row, column, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    return value;
}

static void assert_matrix_close(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    double tolerance
)
{
    assert(numelyx_core_matrix_rows(left) == numelyx_core_matrix_rows(right));
    assert(numelyx_core_matrix_columns(left) == numelyx_core_matrix_columns(right));

    for (size_t row = 0; row < numelyx_core_matrix_rows(left); row++) {
        for (size_t column = 0; column < numelyx_core_matrix_columns(left); column++) {
            double a = at(left, row, column);
            double b = at(right, row, column);
            double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
            assert(fabs(a - b) <= tolerance * scale);
        }
    }
}

static void test_multiplication_transpose_identity(void)
{
    const double a_values[] = {1.0, 2.0, -1.0, 3.0, 0.5, 4.0};
    const double b_values[] = {2.0, 1.0, 0.0, -3.0, 4.0, 2.0};
    numelyx_core_matrix *a = NULL, *b = NULL;
    numelyx_core_matrix *ab = NULL, *ab_t = NULL;
    numelyx_core_matrix *a_t = NULL, *b_t = NULL, *bt_at = NULL;

    assert(numelyx_core_matrix_create_dense(2, 3, a_values, &a) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(3, 2, b_values, &b) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_multiply(a, b, &ab) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_transpose(ab, &ab_t) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_transpose(a, &a_t) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_transpose(b, &b_t) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_multiply(b_t, a_t, &bt_at) == NUMERUS_MATRIX_SUCCESS);

    assert_matrix_close(ab_t, bt_at, 1e-12);

    numelyx_core_matrix_destroy(bt_at);
    numelyx_core_matrix_destroy(b_t);
    numelyx_core_matrix_destroy(a_t);
    numelyx_core_matrix_destroy(ab_t);
    numelyx_core_matrix_destroy(ab);
    numelyx_core_matrix_destroy(b);
    numelyx_core_matrix_destroy(a);
}

static void test_identity_and_zero_laws(void)
{
    const double values[] = {1.0, -2.0, 3.5, 4.0, 0.0, -6.0};
    numelyx_core_matrix *a = NULL, *identity = NULL, *zero = NULL;
    numelyx_core_matrix *ai = NULL, *az = NULL, *sum = NULL;

    assert(numelyx_core_matrix_create_dense(2, 3, values, &a) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_identity(3, &identity) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_zero(3, 3, &zero) == NUMERUS_MATRIX_SUCCESS);

    assert(numelyx_core_matrix_multiply(a, identity, &ai) == NUMERUS_MATRIX_SUCCESS);
    assert_matrix_close(a, ai, 0.0);

    assert(numelyx_core_matrix_multiply(a, zero, &az) == NUMERUS_MATRIX_SUCCESS);
    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 3; column++) {
            assert(at(az, row, column) == 0.0);
        }
    }

    numelyx_core_matrix_destroy(az);
    numelyx_core_matrix_destroy(zero);
    numelyx_core_matrix_destroy(identity);

    assert(numelyx_core_matrix_create_zero(2, 3, &zero) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_add(a, zero, &sum) == NUMERUS_MATRIX_SUCCESS);
    assert_matrix_close(a, sum, 0.0);

    numelyx_core_matrix_destroy(sum);
    numelyx_core_matrix_destroy(zero);
    numelyx_core_matrix_destroy(ai);
    numelyx_core_matrix_destroy(a);
}

static void test_cached_analysis_lifetime(void)
{
    const double values[] = {4.0, 1.0, 1.0, 3.0};

    for (size_t iteration = 0; iteration < 32; iteration++) {
        numelyx_core_matrix *matrix = NULL;
        numelyx_core_matrix *transpose = NULL;
        numelyx_core_matrix_flags first_flags, second_flags;
        double first_determinant = 0.0, second_determinant = 0.0;

        assert(numelyx_core_matrix_create_dense(2, 2, values, &matrix) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_get_flags(matrix, &first_flags) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_get_flags(matrix, &second_flags) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(first_flags.identity == second_flags.identity);
        assert(first_flags.symmetric == second_flags.symmetric);

        assert(numelyx_core_matrix_determinant(matrix, &first_determinant) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_determinant(matrix, &second_determinant) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(first_determinant == second_determinant);
        assert(first_determinant == 11.0);

        assert(numelyx_core_matrix_create_transpose(matrix, &transpose) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_get_flags(transpose, &first_flags) ==
            NUMERUS_MATRIX_SUCCESS);

        numelyx_core_matrix_destroy(transpose);
        numelyx_core_matrix_destroy(matrix);
    }
}

static void test_transpose_view_matches_materialized_matrix(void)
{
    const size_t shapes[][2] = {
        {1, 4},
        {4, 1},
        {2, 3},
        {3, 2}
    };
    size_t shape_index;

    for (shape_index = 0; shape_index < sizeof(shapes) / sizeof(shapes[0]); shape_index++) {
        const size_t rows = shapes[shape_index][0];
        const size_t columns = shapes[shape_index][1];
        double values[12];
        size_t index;
        numelyx_core_matrix *source = NULL;
        numelyx_core_matrix *transpose = NULL;
        numelyx_core_matrix *materialized = NULL;

        for (index = 0; index < rows * columns; index++) {
            values[index] = (double) index - 2.5;
        }

        assert(numelyx_core_matrix_create_dense(rows, columns, values, &source) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_create_transpose(source, &transpose) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_materialize(transpose, &materialized) ==
            NUMERUS_MATRIX_SUCCESS);

        assert_matrix_close(transpose, materialized, 0.0);

        numelyx_core_matrix_destroy(materialized);
        numelyx_core_matrix_destroy(transpose);
        numelyx_core_matrix_destroy(source);
    }
}

int main(void)
{
    test_multiplication_transpose_identity();
    test_identity_and_zero_laws();
    test_transpose_view_matches_materialized_matrix();
    test_cached_analysis_lifetime();
    puts("Matrix property tests passed.");
    return 0;
}
