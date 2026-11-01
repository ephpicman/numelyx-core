/**
 * @file storage_test.c
 * @brief Native tests for Storage representation and access invariants.
 *
 * @details This file is part of Numerus's internal C implementation. Keep its
 * contracts aligned with the Matrix/Storage interfaces and the corresponding
 * implementation units; these declarations do not constitute a stable public
 * C ABI unless explicitly documented otherwise.
 */

#include "../numerus_storage.h"
#include "../numerus_size.h"

#include <assert.h>
#include <stdio.h>

static void test_basic_storages(void)
{
    const double dense[] = {1, 2, 3, 4, 5, 6};
    const double packed[] = {1, 2, 3, 4, 5, 6};
    const double diagonal[] = {10, 20, 30};
    numerus_storage *s = NULL;

    assert(numerus_storage_create_dense(2, 3, dense, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 1, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 6.0);
    }
    assert(numerus_storage_rows(s) == 2);
    assert(numerus_storage_columns(s) == 3);
    numerus_storage_destroy(s);

    assert(numerus_storage_create_upper_triangular(3, packed, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 0, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 3.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 1, 1, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 4.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 2, 0, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);

    assert(numerus_storage_create_lower_triangular(3, packed, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 2, 0, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 4.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 0, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);

    assert(numerus_storage_create_diagonal(3, diagonal, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 2, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 30.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 0, 1, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);
}

static void test_implicit_storages(void)
{
    numerus_storage *s = NULL;

    assert(numerus_storage_create_identity(3, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 0, 0, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 1.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 0, 1, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);

    assert(numerus_storage_create_constant(2, 3, 42, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 1, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 42.0);
        assert(sizeof(value) == sizeof(double));
    }
    numerus_storage_destroy(s);

    assert(numerus_storage_create_zero(2, 3, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 1, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);

    assert(numerus_storage_create_scaled_identity(3, 7, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 1, 1, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 7.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 1, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);
}

static void test_sparse(void)
{
    const numerus_storage_sparse_entry entries[] = {
        {5, 20},
        {1, 10}
    };
    numerus_storage *s = NULL;

    assert(numerus_storage_create_sparse(2, 3, -1, entries, 2, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 0, 1, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 10.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 1, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 20.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 0, 0, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == -1.0);
    }
    numerus_storage_destroy(s);
}

static void test_symmetric_and_banded(void)
{
    const double symmetric[] = {1, 2, 3, 4, 5, 6};
    const double banded[] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 0
    };
    numerus_storage *s = NULL;

    assert(numerus_storage_create_symmetric(3, symmetric, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 0, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 4.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 2, 0, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 4.0);
    }
    numerus_storage_destroy(s);

    assert(numerus_storage_create_banded(3, 3, 1, 1, banded, &s) == 0);
    {
        double value;
        assert(numerus_storage_get(s, 0, 1, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 2.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 1, 0, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 3.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 1, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 5.0);
    }
    {
        double value;
        assert(numerus_storage_get(s, 0, 2, &value) == NUMERUS_STORAGE_SUCCESS);
        assert(value == 0.0);
    }
    numerus_storage_destroy(s);
}

static void test_bounds(void)
{
    numerus_storage *s = NULL;
    double value = 123.0;

    assert(numerus_storage_create_zero(2, 3, &s) == NUMERUS_STORAGE_SUCCESS);
    assert(numerus_storage_get(s, 2, 0, &value) == NUMERUS_STORAGE_OUT_OF_BOUNDS);
    assert(value == 123.0);
    assert(numerus_storage_get(s, 0, 3, &value) == NUMERUS_STORAGE_OUT_OF_BOUNDS);
    assert(numerus_storage_get(s, 0, 0, NULL) == NUMERUS_STORAGE_INVALID_ARGUMENT);
    assert(numerus_storage_get(NULL, 0, 0, &value) == NUMERUS_STORAGE_INVALID_ARGUMENT);
    assert(numerus_storage_get_unchecked(s, 1, 2) == 0.0);
    numerus_storage_destroy(s);
}


static void test_validation_and_immutable_copies(void)
{
    double dense_values[] = {10, 20};
    numerus_storage *storage = NULL;
    double value = 0.0;

    assert(numerus_storage_create_dense(
        1, 2, dense_values, &storage
    ) == NUMERUS_STORAGE_SUCCESS);

    /* Storage owns a copy; callers may change their input buffer afterwards. */
    dense_values[0] = 999;
    assert(numerus_storage_get(storage, 0, 0, &value) ==
        NUMERUS_STORAGE_SUCCESS);
    assert(value == 10.0);
    numerus_storage_destroy(storage);
    storage = NULL;

    {
        numerus_storage_sparse_entry entries[] = {
            {0, 11},
            {2, 33}
        };

        assert(numerus_storage_create_sparse(
            1, 3, -1.0, entries, 2, &storage
        ) == NUMERUS_STORAGE_SUCCESS);

        entries[0].value = 999;
        entries[1].index = 0;

        assert(numerus_storage_get(storage, 0, 0, &value) ==
            NUMERUS_STORAGE_SUCCESS);
        assert(value == 11.0);
        assert(numerus_storage_get(storage, 0, 2, &value) ==
            NUMERUS_STORAGE_SUCCESS);
        assert(value == 33.0);
        numerus_storage_destroy(storage);
        storage = NULL;
    }

    {
        const numerus_storage_sparse_entry duplicate_indices[] = {
            {1, 10},
            {1, 20}
        };
        const numerus_storage_sparse_entry out_of_range[] = {
            {3, 10}
        };

        assert(numerus_storage_create_sparse(
            1, 3, 0.0, duplicate_indices, 2, &storage
        ) == NUMERUS_STORAGE_INVALID_ARGUMENT);
        assert(storage == NULL);

        assert(numerus_storage_create_sparse(
            1, 3, 0.0, out_of_range, 1, &storage
        ) == NUMERUS_STORAGE_OUT_OF_BOUNDS);
        assert(storage == NULL);

        assert(numerus_storage_create_sparse(
            1, 3, 0.0, NULL, 1, &storage
        ) == NUMERUS_STORAGE_INVALID_ARGUMENT);
    }

    assert(numerus_storage_create_dense(
        (size_t) -1, 2, NULL, &storage
    ) == NUMERUS_STORAGE_OVERFLOW);
}


static void test_checked_size_arithmetic(void)
{
    size_t result = 77;

    assert(numerus_size_add(2, 3, &result));
    assert(result == 5);
    assert(numerus_size_add(0, 0, &result));
    assert(result == 0);
    result = 77;
    assert(!numerus_size_add(SIZE_MAX, 1, &result));
    assert(result == 77);
    assert(!numerus_size_add(1, 1, NULL));

    assert(numerus_size_multiply(6, 7, &result));
    assert(result == 42);
    assert(numerus_size_multiply(0, SIZE_MAX, &result));
    assert(result == 0);
    result = 77;
    assert(!numerus_size_multiply(SIZE_MAX, 2, &result));
    assert(result == 77);
    assert(!numerus_size_multiply(1, 1, NULL));

    assert(numerus_size_triangular_count(0, &result));
    assert(result == 0);
    assert(numerus_size_triangular_count(1, &result));
    assert(result == 1);
    assert(numerus_size_triangular_count(4, &result));
    assert(result == 10);
    assert(numerus_size_triangular_count(5, &result));
    assert(result == 15);
    result = 77;
    assert(!numerus_size_triangular_count(SIZE_MAX, &result));
    assert(result == 77);
    assert(!numerus_size_triangular_count(3, NULL));
}

int main(void)
{
    test_basic_storages();
    test_implicit_storages();
    test_sparse();
    test_symmetric_and_banded();
    test_bounds();
    test_validation_and_immutable_copies();
    test_checked_size_arithmetic();

    puts("Storage tests passed.");
    return 0;
}
