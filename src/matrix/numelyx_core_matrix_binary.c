/**
 * @file numelyx_core_matrix_binary.c
 * @brief Binary Matrix operations and result construction.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numelyx_core_matrix_internal.h"

/**
 * Create a lazy element-wise sum.
 *
 * Both parents are borrowed and must outlive the result.
 */
int numelyx_core_matrix_create_add(
    numelyx_core_matrix *left,
    numelyx_core_matrix *right,
    numelyx_core_matrix **matrix
)
{
    return numelyx_core_matrix_create_binary_view(
        left,
        right,
        NUMERUS_MATRIX_BINARY_ADD,
        matrix
    );
}

/**
 * Create a lazy element-wise difference.
 *
 * Both parents are borrowed and must outlive the result.
 */
int numelyx_core_matrix_create_subtract(
    numelyx_core_matrix *left,
    numelyx_core_matrix *right,
    numelyx_core_matrix **matrix
)
{
    return numelyx_core_matrix_create_binary_view(
        left,
        right,
        NUMERUS_MATRIX_BINARY_SUBTRACT,
        matrix
    );
}

/**
 * Create a lazy Hadamard (element-wise) product.
 *
 * Both parents are borrowed and must outlive the result.
 */
int numelyx_core_matrix_create_hadamard_product(
    numelyx_core_matrix *left,
    numelyx_core_matrix *right,
    numelyx_core_matrix **matrix
)
{
    return numelyx_core_matrix_create_binary_view(
        left,
        right,
        NUMERUS_MATRIX_BINARY_HADAMARD,
        matrix
    );
}

/**
 * Create a lazy element-wise quotient.
 *
 * Division follows IEEE-754 double semantics, including infinities and NaN
 * for zero divisors. Both parents are borrowed and must outlive the result.
 */
int numelyx_core_matrix_create_divide(
    numelyx_core_matrix *left,
    numelyx_core_matrix *right,
    numelyx_core_matrix **matrix
)
{
    return numelyx_core_matrix_create_binary_view(
        left,
        right,
        NUMERUS_MATRIX_BINARY_DIVIDE,
        matrix
    );
}
