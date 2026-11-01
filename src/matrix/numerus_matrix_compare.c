/**
 * @file numerus_matrix_compare.c
 * @brief Matrix comparison operations and exact elementwise equality.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numerus_matrix.h"
#include "numerus_numeric.h"

#include <stdbool.h>

static numerus_matrix_status compare_matrices(
    const numerus_matrix *left,
    const numerus_matrix *right,
    bool approximate,
    bool *equal
)
{
    size_t rows;
    size_t columns;
    size_t row;
    size_t column;

    if (left == NULL || right == NULL || equal == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    rows = numerus_matrix_rows(left);
    columns = numerus_matrix_columns(left);
    if (rows != numerus_matrix_rows(right) ||
        columns != numerus_matrix_columns(right)) {
        *equal = false;
        return NUMERUS_MATRIX_SUCCESS;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double left_value;
            double right_value;
            bool values_equal;
            numerus_matrix_status status;

            status = numerus_matrix_get(
                left, row, column, &left_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            status = numerus_matrix_get(
                right, row, column, &right_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            values_equal = approximate
                ? numerus_double_equals(left_value, right_value)
                : left_value == right_value;
            if (!values_equal) {
                *equal = false;
                return NUMERUS_MATRIX_SUCCESS;
            }
        }
    }

    *equal = true;
    return NUMERUS_MATRIX_SUCCESS;
}

numerus_matrix_status numerus_matrix_is_equal(
    const numerus_matrix *left,
    const numerus_matrix *right,
    bool *equal
)
{
    return compare_matrices(left, right, false, equal);
}

numerus_matrix_status numerus_matrix_is_close(
    const numerus_matrix *left,
    const numerus_matrix *right,
    bool *close
)
{
    return compare_matrices(left, right, true, close);
}
