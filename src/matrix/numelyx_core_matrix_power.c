/**
 * @file numelyx_core_matrix_power.c
 * @brief Integer matrix powers and exponentiation helpers.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numelyx_core_matrix.h"

#include <stdint.h>
#include <stddef.h>

/**
 * Compute a non-negative integer matrix power by exponentiation by squaring.
 *
 * Every computed result is materialized. Exponent zero returns identity;
 * exponent one returns an independent copy of the input.
 */
int numelyx_core_matrix_power(
    const numelyx_core_matrix *base,
    size_t exponent,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_matrix *power = NULL;
    numelyx_core_matrix *result = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (base == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numelyx_core_matrix_rows(base) != numelyx_core_matrix_columns(base)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    if (exponent == 0) {
        return numelyx_core_matrix_create_identity(
            numelyx_core_matrix_rows(base), matrix
        );
    }
    if (exponent == 1) {
        return numelyx_core_matrix_materialize(base, matrix);
    }

    status = numelyx_core_matrix_materialize(base, &power);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    while (exponent > 0) {
        if ((exponent & 1U) != 0) {
            if (result == NULL) {
                if (exponent == 1) {
                    result = power;
                    power = NULL;
                    break;
                }

                status = numelyx_core_matrix_materialize(power, &result);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
            } else {
                numelyx_core_matrix *next_result = NULL;

                status = numelyx_core_matrix_multiply(
                    result, power, &next_result
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }

                numelyx_core_matrix_destroy(result);
                result = next_result;
            }
        }

        exponent >>= 1;
        if (exponent > 0) {
            numelyx_core_matrix *squared = NULL;

            status = numelyx_core_matrix_multiply(power, power, &squared);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }

            numelyx_core_matrix_destroy(power);
            power = squared;
        }
    }

    numelyx_core_matrix_destroy(power);
    *matrix = result;
    return NUMERUS_MATRIX_SUCCESS;

failure:
    numelyx_core_matrix_destroy(result);
    numelyx_core_matrix_destroy(power);
    return status;
}

/**
 * Signed-exponent variant of matrix power. The magnitude calculation avoids
 * negating INT64_MIN directly, which would overflow a signed integer.
 */
int numelyx_core_matrix_power_signed(
    const numelyx_core_matrix *base,
    int64_t exponent,
    numelyx_core_matrix **matrix
)
{
    numelyx_core_matrix *inverse = NULL;
    uint64_t magnitude;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (base == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (exponent >= 0) {
        if ((uint64_t) exponent > (uint64_t) SIZE_MAX) {
            return NUMERUS_MATRIX_OVERFLOW;
        }
        return numelyx_core_matrix_power(base, (size_t) exponent, matrix);
    }

    /* This form is defined even for INT64_MIN. */
    magnitude = (uint64_t) (-(exponent + 1)) + UINT64_C(1);
    if (magnitude > (uint64_t) SIZE_MAX) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numelyx_core_matrix_inverse(base, &inverse);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    status = numelyx_core_matrix_power(inverse, (size_t) magnitude, matrix);
    numelyx_core_matrix_destroy(inverse);
    return status;
}
