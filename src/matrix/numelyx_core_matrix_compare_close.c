/**
 * @file numelyx_core_matrix_compare_close.c
 * @brief Tolerance-based approximate Matrix comparison.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_numeric.h"

numelyx_core_matrix_status numelyx_core_matrix_all_close(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    bool *close
)
{
    return numelyx_core_matrix_is_close(left, right, close);
}

numelyx_core_matrix_status numelyx_core_matrix_any_close(
    const numelyx_core_matrix *left,
    const numelyx_core_matrix *right,
    bool *close
)
{
    size_t rows, columns, row, column;
    bool result = false;

    if (left == NULL || right == NULL || close == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    rows = numelyx_core_matrix_rows(left);
    columns = numelyx_core_matrix_columns(left);
    if (rows != numelyx_core_matrix_rows(right) ||
        columns != numelyx_core_matrix_columns(right)) {
        *close = false;
        return NUMERUS_MATRIX_SUCCESS;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            double a, b;
            numelyx_core_matrix_status status =
                numelyx_core_matrix_get(left, row, column, &a);
            if (status != NUMERUS_MATRIX_SUCCESS) return status;
            status = numelyx_core_matrix_get(right, row, column, &b);
            if (status != NUMERUS_MATRIX_SUCCESS) return status;
            if (numelyx_core_double_equals(a, b)) {
                result = true;
                break;
            }
        }
        if (result) break;
    }
    *close = result;
    return NUMERUS_MATRIX_SUCCESS;
}
