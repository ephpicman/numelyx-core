# Statistical Foundation API Boundary

Status: accepted design decision. The initial ownership-safe PHP Matrix façade and OLS workflow were delivered in PRs #194–#197. This document records the boundary between the supported PHP API and internal C capabilities; it is not a statement that the façade is still pending.

## Decision

Numerus provides reusable numerical primitives rather than model-specific estimators. Internal C APIs are authoritative for native ownership, numerical failure, output preservation, and reproducibility. The PHP façade must expose only deliberate, tested contracts; it must not leak C structs, raw pointers, allocator details, or internal status integers.

A C implementation is not automatically a PHP feature. Documentation must distinguish internal capabilities from methods actually registered by the extension.

## Current PHP façade

The initial public surface is a value-owning `Numerus\\Matrix` object with:

- `Matrix::fromRows(array $rows)` and `Matrix::zeros(int $rows, int $columns)`
- `rows()`, `columns()`, and zero-based `get(int $row, int $column)`
- `transpose()`, `multiply(Matrix $other)`, and `inverse()`
- `leastSquares(Matrix $rightHandSide)`

The supported OLS workflow composes these primitives. The recommended path is `leastSquares()`, which uses pivoted QR and requires a full-column-rank design matrix. Rank-deficient and underdetermined inputs are rejected; a PHP-facing minimum-norm solution is not currently exposed.

The explicit normal-equation composition `(XᵀX)⁻¹Xᵀy` is possible but is not the recommended numerical default because forming normal equations squares the condition number and explicit inversion can amplify rounding error.

Returned PHP matrices own independent native storage. Native lazy views have non-owning parent references, but the PHP façade materializes returned results rather than exposing those borrowed views. Cloning PHP Matrix objects is unsupported to avoid ambiguous native ownership.

See the [PHP API reference](php-api.md), [user guide](user-guide.md), and [release-gate audit](php-api-release-audit.md) for the exact public contract and acceptance evidence.

## Internal capabilities are not automatically PHP APIs

The repository includes internal C capabilities for storage and matrix views, decompositions, triangular solves, log determinants, scalar/log-domain utilities, RNG and sampling, optimization, and probability calculations. Their presence in the source tree does not mean they are callable from PHP.

Any future PHP exposure should be justified by a concrete compositional need and must define:

- PHP types, method names, argument defaults, and named-argument compatibility;
- validation and PHP-visible exception behavior;
- ownership, lifetime, cloning, and any callback/context rules;
- numerical preconditions, failure semantics, and output guarantees;
- PHPT coverage for success, invalid input, numerical failure, and lifecycle;
- user documentation that distinguishes supported behavior from internal implementation details.

## Boundary by capability

| Capability | Current implementation | PHP-facing status |
|---|---|---|
| Matrix construction, access, transpose, multiplication, inverse | Internal C plus PHP façade | Exposed through `Numerus\\Matrix` |
| Pivoted-QR least-squares solve | Internal C plus PHP façade | Exposed through `Matrix::leastSquares()`; full column rank required |
| Lazy Matrix views and storage representations | Internal C | Not exposed as PHP objects |
| RNG, sampling, scalar/log-domain functions, optimizer, probability primitives | Internal C | Not currently exposed as PHP APIs |
| Ready-made OLS/WLS/GLS/MLE estimators | Not part of the product design | Deliberately not provided |
| Minimum-norm least-squares solution through pseudoinverse/SVD | Internal capabilities may exist at C level | Not exposed through the initial PHP façade |
| Complete statistical inference API | Future roadmap scope | Not currently exposed |

## Contract and naming rules

1. Keep internal C symbols prefixed `numerus_` and declarations in focused headers. Do not promise a stable external C ABI unless it is separately versioned and documented.
2. Keep owning objects opaque. Constructors must document ownership and failure output state; every retained handle needs an explicit release operation.
3. Borrowed callback contexts must document lifetime, mutability, reentrancy, and thread-safety assumptions. C callbacks are not automatically PHP callbacks.
4. Keep failure categories distinct in C. PHP wrappers must map invalid arguments, non-finite input, singularity, non-convergence, and allocation failure to documented PHP-visible failures; never rely on NaN alone to signal failure.
5. Do not return a C view with a non-owning parent reference to PHP unless the PHP object model explicitly guarantees that parent lifetime. The current façade avoids this by materializing returned results.
6. Keep PHPT coverage for every PHP-visible contract. Native tests remain necessary for C-level ownership, numerical invariants, allocation failure, and deterministic vectors.
7. Do not introduce model-specific `fit_ols()`, `fit_gls()`, `fit_mle()`, GP fitting, or Bayesian inference APIs without changing the established product decision explicitly.

## Release implication

The initial OLS-enabling PHP surface is implemented. Release readiness is determined by the current [release roadmap](release-roadmap.md), [PHP OLS release-gate audit](php-api-release-audit.md), and CI on the exact release-candidate commit—not by the broader statistical-foundation roadmap alone. Windows remains unverified until a Windows build and test job passes.
