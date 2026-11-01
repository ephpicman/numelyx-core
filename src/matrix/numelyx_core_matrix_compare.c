/**
 * @file numelyx_core_matrix_compare.c
 * @brief Matrix comparison operations and exact elementwise equality.
 *
 * @details This translation unit implements focused Matrix functionality.
 * It relies on the common Matrix access/status contracts and checked-size
 * helpers rather than exposing storage representation details to callers.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_numeric.h"

#include <stdbool.h>

static numelyx_core_matrix_status compare_matrices(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
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

    rows = numelyx_core_matrix_rows(left);
    columns = numelyx_core_matrix_columns(left);
    if (rows != numelyx_core_matrix_rows(right) ||
        columns != numelyx_core_matrix_columns(right)) {
        *equal = false;
        return NUMERUS_MATRIX_SUCCESS;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double left_value;
            double right_value;
            bool values_equal;
            numelyx_core_matrix_status status;

            status = numelyx_core_matrix_get(
                left, row, column, &left_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            status = numelyx_core_matrix_get(
                right, row, column, &right_value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                return status;
            }

            values_equal = approximate
                ? numelyx_core_double_equals(left_value, right_value)
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

numelyx_core_matrix_status numelyx_core_matrix_is_equal(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    bool *equal
)
{
    return compare_matrices(left, right, false, equal);
}

numelyx_core_matrix_status numelyx_core_matrix_is_close(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    bool *close
)
{
    return compare_matrices(left, right, true, close);
}
