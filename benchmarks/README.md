# Matrix Benchmarks

This directory contains a standalone benchmark for the internal C Matrix API.
It is diagnostic tooling, not a performance test with pass/fail thresholds.

## Run

From the repository root, using GCC/Clang with GNU ld on a Unix-like system:

```sh
cc -Isrc/core -Isrc/storage -Isrc/matrix -Isrc/statistics -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free \
  -o benchmarks/matrix_benchmark \
  benchmarks/matrix_benchmark.c \
 src/matrix/numerus_matrix.c src/matrix/numerus_matrix_view_nodes.c src/matrix/numerus_matrix_constructors.c src/matrix/numerus_matrix_binary.c \
  src/matrix/numerus_matrix_materialize.c src/matrix/numerus_matrix_multiply.c src/matrix/numerus_matrix_analysis.c \\
  src/matrix/numerus_matrix_lu.c src/matrix/numerus_matrix_inverse.c src/matrix/numerus_matrix_condition.c src/storage/numerus_storage.c
./benchmarks/matrix_benchmark
```

The benchmark reports compiler version, shape, iteration count, CPU time from
`clock()`, allocation calls, and total bytes requested through `malloc` and
`calloc` during the measured interval. The linker wrappers are used only in
this standalone executable; extension builds continue to use the Zend Memory
Manager.

## Workloads and methodology

- 64×64 dense, diagonal, upper-triangular, and sparse reads.
- 64×64 transpose, scalar-transform, and binary-arithmetic view reads. Nested transpose views are measured separately to expose composition overhead.
- 64×128 joined-view reads.
- 1,000 scalar-view constructions.
- Three materializations of a transpose view.
- Three 32×32 dense matrix multiplications.
- Three 64×64 generic matrix multiplications for dense×dense, dense×identity,
  identity×dense, dense×zero, dense×diagonal, dense×upper-triangular, and
  dense×sparse operands. These establish comparison points before any
  Storage-specific fast path is introduced. Each workload runs five samples
  of three multiplications and reports median, minimum, maximum, and median
  allocation metrics.\n- Three 32×32 cold LU-based inverses, three cached inverse hits, cold and cached-LU solve comparisons, and condition estimates. Cold inverse measurements use independently materialized, un-inverted matrices; cache-hit measurements warm one source before timing. The inverse input is diagonally dominant to avoid benchmarking a singular matrix.
- 128 first-use versus repeated-use reads for cached structural flags and determinants. Independent matrices are prepared outside the timed interval for cache-miss measurements; repeated calls use one already-computed Matrix for cache-hit measurements.\n- One maximum residual check, `max|A*A⁻¹-I|`, outside the timed interval.

The source matrices are constructed before counters and timers are reset for
each measured case. Multiplication workloads use five samples and report the
median and range to reduce the influence of a single noisy timing sample. Read workloads perform full checked-coordinate scans;
operation workloads create and destroy each result inside the timed interval. Inverse residual validation is deliberately outside the timed interval.
A volatile sink prevents the read results from being trivially discarded.

These numbers are intended as a reproducible starting point, not universal
performance claims. They depend on compiler, machine load, allocator, and build
configuration. Do not compare results from materially different environments
without recording those differences. No timing threshold is a CI gate; CI runs
the benchmark as a smoke test and prints its measurements for review.

Allocation metrics count requested bytes and calls to `malloc`/`calloc` in
the linked standalone program. They do not measure peak live memory, retained
bytes, allocator metadata, or allocations internal to libc that do not pass
through the wrapped symbols.
