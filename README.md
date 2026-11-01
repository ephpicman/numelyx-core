# Numelyx Core

Numelyx Core is a standalone C numerical-computing library migrated from the
native computational layer of [Numerus](https://github.com/ephpicman/Numerus).

The repository contains the language-independent numerical implementation:
immutable matrix storage, composable Matrix primitives and views, numerical
linear algebra, random-number generation, probability distributions, and
optimization primitives. PHP extension glue, Zend-specific public APIs,
PHPT tests, and PHP build configuration are intentionally excluded.

## Build

Requirements: CMake 3.16+ and a C11 compiler.

```sh
cmake -S . -B build -DNUMELYX_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Standalone builds use the C runtime allocator. The source retains conditional
Zend allocator support only in the upstream integration points; the CMake
target selects the libc allocator paths and does not include PHP headers.

## Thread safety\n\nShared opaque objects use atomic reference counting. Matrix lazy caches are\nserialized per Matrix, and mutable RNG state is serialized per RNG. See\n[`docs/thread-safety-and-lifetime.md`](docs/thread-safety-and-lifetime.md) for\nownership requirements and callback restrictions.\n\n## Benchmarks

```sh
cmake -S . -B build -DNUMELYX_BUILD_BENCHMARKS=ON
cmake --build build --target matrix_benchmark
./build/matrix_benchmark
```

## Source layout

- `src/core`: checked size arithmetic, stable scalar functions, generic object lifetime, and synchronization primitives.
- `src/storage`: opaque immutable storage representations.
- `src/matrix`: composable Matrix primitives, views, and factorizations.
- `src/statistics`: RNG, probability, and optimization primitives.
- `tests`: native C regression tests.
- `docs`: numerical, subsystem, thread-safety, and lifetime contracts.

The C symbols retain their upstream `numelyx_core_*` names during the extraction;
renaming the ABI is a separate compatibility decision.
