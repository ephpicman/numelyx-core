# Storage

Storage is Numerus's internal representation layer for matrix data. It exposes
one logical matrix interface while allowing the implementation to use a
representation suited to the matrix structure.

Storage is **not** a PHP-facing API. Higher-level components such as Matrix
should depend on the opaque Storage contract rather than on its internal
allocation layout.

## Core properties

### Immutable

A Storage instance has no mutation API. Constructor inputs are copied into
memory owned by Storage. After construction, its logical values and dimensions
do not change.

This permits higher-level objects to share immutable storage without
coordinating writes. Immutability alone does not manage object lifetime or
provide a thread-safety guarantee.

### Opaque

The public header declares:

```c
typedef struct numerus_storage numerus_storage;
```

The structure is private to `numerus_storage.c`. Consumers therefore cannot
depend on whether a matrix is dense, packed, implicit, sparse, or banded.

### Common logical interface

Every representation implements the same logical operation:

```c
numerus_storage_get(storage, row, column, &value);
```

A successful read produces a `double`, regardless of how the value is
represented internally.

The checked accessor returns an explicit status:

- `NUMERUS_STORAGE_SUCCESS`
- `NUMERUS_STORAGE_INVALID_ARGUMENT`
- `NUMERUS_STORAGE_OVERFLOW`
- `NUMERUS_STORAGE_OUT_OF_MEMORY`
- `NUMERUS_STORAGE_OUT_OF_BOUNDS`

The output value is not modified when a checked read fails.

For validated hot loops, `numerus_storage_get_unchecked()` skips argument and
bounds checks. It must only be used when the caller has already established
that the Storage pointer and coordinates are valid. Passing invalid arguments
to this function is undefined behaviour.

## Representations

| Kind | Logical representation | Stored values |
| --- | --- | --- |
| Dense | Arbitrary rectangular matrix | Every element, row-major |
| Upper triangular | Square matrix with values on/above diagonal | Packed upper triangle |
| Lower triangular | Square matrix with values on/below diagonal | Packed lower triangle |
| Diagonal | Square diagonal matrix | Main diagonal |
| Identity | Identity matrix | None |
| Constant | Every element has one value | One value |
| Zero | All-zero matrix | None |
| Scaled identity | Diagonal is one shared value, elsewhere zero | One value |
| Sparse | Default value plus explicit overrides | Sorted sparse entries |
| Symmetric | Symmetric square matrix | One triangular half |
| Banded | Values inside fixed lower/upper bandwidth | Fixed-width row bands |

All constructors require positive dimensions. Square-only representations
require equal row and column counts.

### Dense

Dense storage is row-major and stores all `rows * columns` values. The
implementation checks size calculations for overflow before allocating.

### Triangular

Upper and lower triangular storage use packed arrays containing only the
logical triangle. Reads outside the represented triangle return `0.0`.

### Diagonal

Only the main diagonal is stored. Every off-diagonal read returns `0.0`.

### Identity and scaled identity

Identity stores no values: diagonal reads are `1.0` and off-diagonal reads are
`0.0`.

Scaled identity stores one value. Diagonal reads return that value and
off-diagonal reads return `0.0`.

### Constant and zero

Constant storage stores one value and returns it for every valid coordinate.
Zero storage stores no values and returns `0.0` for every valid coordinate.

### Sparse

Sparse storage combines a default value with explicit row-major entries:

```c
typedef struct {
    size_t index;
    double value;
} numerus_storage_sparse_entry;
```

The linear index is `row * columns + column`. The constructor validates the
element count, copies the entries, sorts them by linear index, rejects
duplicate indices, rejects out-of-range indices, and stores the default value.

Lookup uses binary search, giving `O(log k)` lookup for `k` explicit entries.
An empty entry set is valid and behaves as a constant matrix whose value is
`default_value`.

### Symmetric

Symmetric storage keeps the lower triangle and mirrors coordinates from the
upper triangle during reads. A coordinate `(row, column)` is normalised to
`(max(row, column), min(row, column))` before accessing the packed lower
triangle.

### Banded

Banded storage keeps a fixed number of diagonals around the main diagonal.
`lower_bandwidth` controls how many diagonals below the main diagonal are
stored; `upper_bandwidth` controls how many are stored above it. Coordinates
outside the configured band return `0.0`.

Bandwidths larger than the dimensions are clamped to the largest meaningful
values. Since dimensions must be positive, there is no empty-matrix case.

The internal row width is:

```text
lower_bandwidth + 1 + upper_bandwidth
```

This keeps coordinate access predictable and avoids storing structural zeros
outside the configured band.

## Memory management

The extension build uses PHP's Zend Memory Manager:

- `emalloc()`
- `ecalloc()`
- `efree()`

Standalone native C tests define `NUMERUS_STORAGE_USE_LIBC_ALLOC` so the
implementation can be tested without embedding the PHP runtime. This is a
test/build convenience; the extension itself uses Zend Memory Manager.

## Error and ownership rules

Storage owns every dynamically allocated buffer created by its constructors.
Input arrays are never retained by reference, so the caller may release or
reuse input buffers after a successful constructor call.

On constructor failure, no partially constructed Storage is returned: the
output pointer is reset to `NULL` after cleanup.

The checked getter distinguishes failure from a legitimate numeric zero.
`0.0` is always a valid matrix value and is never used as an error sentinel.

## API

The authoritative function-level contracts are declared in
[`numerus_storage.h`](../numerus_storage.h).

Main operations:

- `numerus_storage_create_*()` — construct an immutable representation.
- `numerus_storage_get()` — validated element access.
- `numerus_storage_get_unchecked()` — unchecked hot-path element access.
- `numerus_storage_destroy()` — release Storage and owned memory.
- `numerus_storage_rows()` / `numerus_storage_columns()` — dimensions.
- `numerus_storage_kind_of()` — representation kind.

## Testing

The Storage native test suite covers value access for each representation,
implicit zero regions, sparse validation and lookup, symmetric mirroring, band
boundaries, dimensions, bounds handling, NULL arguments, preservation of the
output value after failed reads, and the unchecked accessor.

Run it from the repository root:

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/storage_test tests/storage_test.c numerus_storage.c
./tests/storage_test
```

## Scope boundary

Storage is deliberately limited to representation and element access. It does
not currently implement matrix arithmetic, decomposition algorithms,
BLAS/LAPACK integration, iteration APIs, expression trees, PHP object wrappers,
mutation, or high-level statistical and optimisation APIs. Those concerns
belong to higher layers and should be introduced when there is a concrete
numerical requirement for them.
