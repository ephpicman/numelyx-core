/**
 * @file numelyx_core_matrix_core.h
 * @brief Matrix lifecycle and access API.
 */
#ifndef NUMERUS_MATRIX_CORE_H
#define NUMERUS_MATRIX_CORE_H

#include "numelyx_core_matrix_types.h"

/**
 * @brief Create a root Matrix from a Storage representation.
 *
 * On success, Matrix owns the newly created Storage.
 *
 * @param kind Storage representation to construct.
 * @param rows Number of rows; must be greater than zero.
 * @param columns Number of columns; must be greater than zero.
 * @param data Representation-specific construction data. It may be NULL for
 *             representations that require no data.
 * @param matrix Receives the new Matrix.
 */
int numelyx_core_matrix_create(
    numelyx_core_storage_kind kind,
    size_t rows,
    size_t columns,
    const numelyx_core_matrix_data *data,
    numelyx_core_matrix **matrix
);

/**
 * @brief Create a Matrix whose values are provided by an existing parent.
 *
 * A child may have different logical dimensions from its parent. This is
 * required for derived views such as a transpose. The child retains the
 * parent automatically and releases that dependency when the child is reclaimed.
 *
 * @param parent Existing parent Matrix.
 * @param rows Number of rows exposed by the child.
 * @param columns Number of columns exposed by the child.
 * @param matrix Receives the new child Matrix.
 */
int numelyx_core_matrix_create_from_parent(
    numelyx_core_matrix *parent,
    size_t rows,
    size_t columns,
    numelyx_core_matrix **matrix
);

/**
 * @brief Create a child Matrix with coordinate and value transforms.
 *
 * A NULL callback selects its identity/pass-through default. The context is
 * borrowed and is never modified or freed by Matrix. It must remain valid and
 * logically unchanged for the child's lifetime. Callbacks must be deterministic
 * and must not depend on mutable external state; this contract is required for
 * the child to remain logically immutable. The const qualifier documents and
 * enforces read-only access through this API, but cannot prevent a caller from
 * mutating the same object through another alias.
 */
int numelyx_core_matrix_create_from_parent_with_transforms(
    numelyx_core_matrix *parent,
    size_t rows,
    size_t columns,
    numelyx_core_coordinate_transform_fn coordinate_transform,
    numelyx_core_value_transform_fn value_transform,
    const void *context,
    numelyx_core_matrix **matrix
);

/** Read one Matrix element with validation. */
numelyx_core_matrix_status numelyx_core_matrix_get(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    double *value
);

/**
 * @brief Read one Matrix element without validation.
 *
 * This is the hot-path accessor for callers that already validated the
 * Matrix and coordinates.
 */
numelyx_core_matrix_status numelyx_core_matrix_get_unchecked(
    const numelyx_core_matrix *matrix,
    size_t row,
    size_t column,
    double *value
);

/** Return the number of rows, or zero for NULL. */
size_t numelyx_core_matrix_rows(const numelyx_core_matrix *matrix);

/** Return the number of columns, or zero for NULL. */
size_t numelyx_core_matrix_columns(const numelyx_core_matrix *matrix);

/** Return the Storage kind of the root Matrix, or ZERO for NULL. */
numelyx_core_storage_kind numelyx_core_matrix_storage_kind(
    const numelyx_core_matrix *matrix
);

/**
 * @brief Retain a Matrix reference.
 *
 * The caller must already hold a valid reference. The operation fails rather
 * than overflowing the reference counter. Reference counting is not currently
 * thread-safe; callers must externally synchronize concurrent retain/release.
 */
numelyx_core_matrix_status numelyx_core_matrix_retain(
    numelyx_core_matrix *matrix
);

/**
 * @brief Release one caller-owned Matrix reference.
 *
 * A Matrix is reclaimed only after its final reference is released. Parent
 * references held by derived views are released automatically during final
 * reclamation. Passing NULL is a no-op. Releasing the same reference twice is
 * invalid; callers must not use a pointer after releasing its last reference.
 */
void numelyx_core_matrix_destroy(numelyx_core_matrix *matrix);

#endif
