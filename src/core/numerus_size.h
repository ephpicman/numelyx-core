/**
 * @file numerus_size.h
 * @brief Checked size_t arithmetic shared by Storage and Matrix.
 *
 * Helpers leave the output parameter unchanged when the operation would
 * overflow or the output pointer is NULL.
 */
#ifndef NUMERUS_SIZE_H
#define NUMERUS_SIZE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Add two size values if the result is representable by size_t.
 * @param left First non-negative operand.
 * @param right Second non-negative operand.
 * @param result Receives the sum on success; may be NULL only to receive failure.
 * @return true on success; false for a NULL output pointer or overflow.
 *
 * On failure, @p result is not modified.
 */
static inline bool numerus_size_add(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || right > SIZE_MAX - left) {
        return false;
    }

    *result = left + right;
    return true;
}

/**
 * @brief Multiply two size values if the result is representable by size_t.
 * @param left First operand.
 * @param right Second operand.
 * @param result Receives the product on success.
 * @return true on success; false for a NULL output pointer or overflow.
 *
 * On failure, @p result is not modified. A zero operand is handled without
 * dividing by zero in the overflow check.
 */
static inline bool numerus_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return false;
    }

    *result = left * right;
    return true;
}

/** Compute n * (n + 1) / 2 without overflowing an intermediate value. */
/**
 * @brief Compute the triangular number n(n + 1) / 2 with checked arithmetic.
 * @param n Number of entries in the corresponding triangle dimension.
 * @param result Receives the element count on success.
 * @return true on success; false if an intermediate or final value overflows,
 *         or if @p result is NULL.
 *
 * The even/odd split divides one factor before multiplication so the
 * mathematically representable result is not rejected due to an avoidable
 * overflow in the intermediate product n(n + 1).
 */
static inline bool numerus_size_triangular_count(size_t n, size_t *result)
{
    size_t left;
    size_t right;

    if (n % 2 == 0) {
        left = n / 2;
        if (!numerus_size_add(n, 1, &right)) {
            return false;
        }
    } else {
        left = n;
        right = n / 2 + 1;
    }

    return numerus_size_multiply(left, right, result);
}

#endif
