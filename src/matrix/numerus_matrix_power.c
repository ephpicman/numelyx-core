/**
 * @file numerus_matrix_power.c
 * @brief Integer matrix powers and exponentiation helpers.
 *
 * @details This translation unit implements one focused part of the internal
 * Matrix API. Public-to-the-subsystem declarations live in the corresponding
 * Matrix headers; shared representation invariants and status semantics are
 * defined by the Matrix core and internal headers.
 */

#include "numerus_matrix.h"

#include <stdint.h>
#include <stddef.h>

/**
 * Compute a non-negative integer matrix power by exponentiation by squaring.
 *
 * Every computed result is materialized. Exponent zero returns identity;
 * exponent one returns an independent copy of the input.
 */
int numerus_matrix_power(
    const numerus_matrix *base,
    size_t exponent,
    numerus_matrix **matrix
)
{
    numerus_matrix *power = NULL;
    numerus_matrix *result = NULL;
    int status;

    if (matrix == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *matrix = NULL;

    if (base == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numerus_matrix_rows(base) != numerus_matrix_columns(base)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    if (exponent == 0) {
        return numerus_matrix_create_identity(
            numerus_matrix_rows(base), matrix
        );
    }
    if (exponent == 1) {
        return numerus_matrix_materialize(base, matrix);
    }

    status = numerus_matrix_materialize(base, &power);
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

                status = numerus_matrix_materialize(power, &result);
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
            } else {
                numerus_matrix *next_result = NULL;

                status = numerus_matrix_multiply(
                    result, power, &next_result
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }

                numerus_matrix_destroy(result);
                result = next_result;
            }
        }

        exponent >>= 1;
        if (exponent > 0) {
            numerus_matrix *squared = NULL;

            status = numerus_matrix_multiply(power, power, &squared);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }

            numerus_matrix_destroy(power);
            power = squared;
        }
    }

    numerus_matrix_destroy(power);
    *matrix = result;
    return NUMERUS_MATRIX_SUCCESS;

failure:
    numerus_matrix_destroy(result);
    numerus_matrix_destroy(power);
    return status;
}

/**
 * Signed-exponent variant of matrix power. The magnitude calculation avoids
 * negating INT64_MIN directly, which would overflow a signed integer.
 */
int numerus_matrix_power_signed(
    const numerus_matrix *base,
    int64_t exponent,
    numerus_matrix **matrix
)
{
    numerus_matrix *inverse = NULL;
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
        return numerus_matrix_power(base, (size_t) exponent, matrix);
    }

    /* This form is defined even for INT64_MIN. */
    magnitude = (uint64_t) (-(exponent + 1)) + UINT64_C(1);
    if (magnitude > (uint64_t) SIZE_MAX) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    status = numerus_matrix_inverse(base, &inverse);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    status = numerus_matrix_power(inverse, (size_t) magnitude, matrix);
    numerus_matrix_destroy(inverse);
    return status;
}
