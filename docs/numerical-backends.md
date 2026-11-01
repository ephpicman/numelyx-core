# BLAS and LAPACK backend

Numelyx Core uses BLAS and LAPACK as required numerical dependencies. The
public Matrix API and Storage representations remain owned by Numelyx Core;
vendor-specific routines are confined to the internal numerical layer.

## Required interfaces

- BLAS through the CBLAS C interface.
- LAPACK through the LAPACKE C interface.
- CMake must locate BLAS, LAPACK, `cblas.h`, `lapacke.h`, and the LAPACKE
  library, then verify that both C APIs link successfully.
- Ubuntu development packages: `libopenblas-dev` and `liblapacke-dev`.
- Windows dependencies are declared in `vcpkg.json` and installed with
  `vcpkg install --triplet x64-windows`.
- CI exercises both dependency setups.

## Current dispatch

| Operation | Dispatch |
| --- | --- |
| Dense, finite matrix multiplication | CBLAS `cblas_dgemm` (row-major) |
| Finite square LU factorization | LAPACKE `LAPACKE_dgetrf` (row-major) |
| Cholesky factorization | LAPACKE `LAPACKE_dpotrf` (row-major), with Numelyx Core symmetry and scale-relative pivot checks |
| Unpivoted reduced QR | LAPACKE `LAPACKE_dgeqrf` + `LAPACKE_dorgqr` (row-major) |
| Reduced SVD | LAPACKE `LAPACKE_dgesdd` (row-major); one-sided Jacobi remains the non-convergence fallback |
| Symmetric eigen decomposition | LAPACKE `LAPACKE_dsyevd` (row-major); cyclic Jacobi remains the non-convergence fallback |
| Triangular solve | LAPACKE `LAPACKE_dtrtrs` (row-major), after exact structural validation |
| Identity, zero, and sparse multiplication shortcuts | Existing Numelyx Core fast paths |
| Singular LU inputs, non-finite data, or dimensions beyond the conservative 32-bit LAPACK boundary | Existing native LU path (preserves exact-pivot rank semantics) |
| Pivoted QR (rank-revealing) | Existing native implementation; its permutation/rank contract remains separate |
| Non-finite dense multiplication and non-dense multiplication | Existing native multiplication path |

Dense inputs are gathered through the Matrix accessor before calling the
backend. This preserves the opacity of Storage and supports logical dense
views without assuming that every Matrix is a contiguous root allocation.
CBLAS/LAPACKE integer dimensions are range-checked before conversion.

## Numerical and ownership contracts

- Backend calls operate on temporary row-major buffers; they never receive
  pointers into immutable shared Storage.
- Output Matrix objects are published only after the backend operation
  completes successfully.
- Native implementations remain available for the documented fallback cases
  and as differential-test references during the migration.
- BLAS/LAPACK may use different reduction orders from the native kernels.
  Floating-point comparisons must therefore use numerical tolerances except
  where exact behavior is part of a structural fast-path contract.
- A backend implementation may use internal worker threads. Applications
  combining Numerus-level parallelism with multithreaded BLAS should configure
  vendor thread counts to avoid oversubscription.

## Remaining migration

The integration dispatches GEMM, LU, Cholesky, unpivoted QR, reduced SVD,
symmetric eigen decomposition, and triangular solves. Pivoted QR remains on
its existing native implementation because its rank-revealing permutation
contract is separate from unpivoted QR. Sparse and structured Storage fast
paths remain Numelyx Core responsibilities; a general sparse solver dependency
is not introduced by this change.
