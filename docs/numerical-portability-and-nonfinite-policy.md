# Numerical Portability and Non-Finite Policy

Status: accepted policy for the v1 statistical C APIs. Individual implementation issues must implement and test these rules; this document is not evidence that future APIs exist.

## Supported and tested build matrix

The repository's current CI workflow (`.github/workflows/tests.yml`) tests:

- PHP 8.2, 8.3, 8.4 and 8.5 on `ubuntu-latest`;
- the normal extension build and native tests;
- a separate debug-oriented build.

The workflow does not currently test Windows, macOS, alternate C compilers, or a dedicated AddressSanitizer/UndefinedBehaviorSanitizer configuration. `config.w32` exists, but that file alone is not evidence that Windows builds or tests pass.

**Release-tested support means the CI matrix above.** Treat Windows and other platforms as unverified until an actual build/test job is added and green. Do not describe an untested platform as supported merely because a build configuration file exists. Keep the README's build instructions, but explicitly note the Windows validation gap.

## C and floating-point assumptions

- New code must use fixed-width unsigned integer types for the PCG raw generator. PCG32 relies on defined modulo-(2^{64}) unsigned arithmetic; signed overflow is forbidden.
- Statistical double APIs assume finite IEEE-754 binary64 values on supported builds. Where a specific algorithm requires binary64 properties, check `DBL_MANT_DIG == 53` and `DBL_MAX_EXP == 1024` at compile time rather than silently making a different reproducibility promise.
- Do not require a newer C language mode than the repository's extension build supports unless the build configuration is deliberately updated and validated. Standalone tests currently compile with C11; that does not automatically establish the extension's language-mode contract.
- Use checked size multiplication/addition before allocating or indexing buffers. Keep arithmetic, allocation failure and dimension overflow distinct.

## Non-finite and domain policy

### Matrix APIs

Existing Matrix APIs retain their documented operation-specific behavior. Matrix constructors/storage predicates may represent or inspect non-finite values; numerical algorithms must not assume every Matrix is finite merely because it was successfully constructed.

For new statistical algorithms:

- Reject NaN and either infinity in observations, design matrices, covariance matrices, initial optimizer parameters, options, objective values and gradient values, unless a function's documented mathematical domain explicitly requires a different policy.
- Return a dedicated non-finite status/reason; never signal failure solely by returning NaN or infinity.
- Validate covariance/factorization assumptions separately. A finite covariance that is not symmetric positive definite is a covariance-domain failure, not a non-finite failure.
- On API failure, output pointers remain NULL or scalar outputs remain unchanged as defined by the function contract. Never publish partially computed results.

### Scalar utilities (#118)

- Sigmoid, logit, softplus and log-sigmoid accept finite input only.
- Logit accepts only (0 < p < 1); endpoints are domain errors, not successful infinite outputs.
- Sigmoid saturation to exactly 0 or 1 for extreme finite values is a valid floating-point result.
- Log-sum-exp requires a non-empty array of finite values. NaN and infinities are rejected; empty input is an invalid argument. If the final finite-input calculation overflows, return a numerical-failure status and preserve the output.
- Use `log1p`/`expm1` from the C math library where appropriate; do not implement naïve expressions such as `log(1 + exp(x))`.

### RNG and distribution transforms (#115–#117, #121)

- Raw PCG32 output is an integer stream and is bit-for-bit reproducible for the documented algorithm, seed, stream and call sequence.
- Uniform conversion is fixed to the top 53 bits of two raw outputs, scaled by (2^{-53}), producing ([0,1)). It must not depend on ambient/global RNG state.
- Normal sampling uses Box–Muller without caching, maps the radial uniform into ((0,1]) to avoid `log(0)), and consumes exactly four raw outputs per result. Since ordinary `libm` implementations can differ, only same supported build/runtime repeatability may be promised for floating-point normal values.
- RNG functions do not accept floating-point seeds; seed/stream are explicit fixed-width integers. RNG output is not cryptographic.

### Optimizer callbacks (#119)

- Initial parameters and options must be finite and valid before callbacks are invoked.
- Callback failure is distinct from a callback that returns success with a non-finite result.
- Non-finite objective/gradient values must not be used in line search or BFGS updates. Invalid trial points may be rejected/backtracked only where the optimizer contract says so; do not turn them into apparent convergence.
- Convergence, iteration limit, small-step termination, line-search failure, callback failure, non-finite evaluation and numerical breakdown are separate result reasons. Allocation failure remains an API failure.

### Gaussian probability (#120–#121)

- Observations, means and covariance/factor inputs must be finite.
- Log density requires the covariance contract accepted by the implementation (v1 initial path: symmetric positive definite); reject singular or non-SPD covariance instead of returning a misleading density.
- Do not form covariance inverses. Factorization, triangular solves and signed log determinant must propagate their documented statuses.
- Sampling requires a valid factor and explicit RNG state. Any failed factorization or transform leaves output unpublished.

## Status taxonomy

- Keep API-level invalid argument, domain error, non-finite input/result, dimension mismatch, singularity, numerical breakdown, size overflow and allocation failure distinct where relevant.
- Reuse existing Matrix status codes only inside Matrix operations. Scalar numeric, RNG and optimizer contracts should have focused status/result types or a documented common internal taxonomy; do not accidentally make Matrix enum values part of unrelated APIs.
- An optimizer's valid call may produce a non-converged result object; its termination reason is not an API failure. A constructor/allocation failure must not publish a partial result.
- If a C math function sets `errno` or raises floating-point exceptions, do not rely on process-global `errno` alone as the public error channel. Validate inputs and outputs explicitly.

## Portability and regression checks

- Keep the existing PHP 8.2–8.5 Linux CI matrix green for every PR.
- Add exact PCG raw-output vectors to #115. These tests should be platform-independent.
- Add boundary/non-finite tests to #118, #119, #120 and #121 as those APIs are implemented; don't add tests for APIs that do not yet exist.
- Add compiler/sanitizer jobs as a separate focused improvement when the supported build environment permits it; the current workflow is not sanitizer coverage.
- Do not claim cross-platform bitwise floating-point distribution outputs based only on a deterministic integer PRNG.

## Relationship to other contracts

- [API boundary](statistical-api-boundary.md) defines internal C versus future PHP-visible exposure.
- [RNG contract](reproducible-rng-contract.md) defines raw PCG reproducibility and explicitly limits distribution-transform guarantees.
- [Optimizer contract](optimizer-contract.md) defines callback and termination behavior.
- [Scalar utility selection](scalar-log-domain-utilities.md) defines finite domains and selected stable functions.
- [GLS/log-determinant contract](gls-whitening-and-log-determinant-contracts.md) defines covariance and factorization assumptions.


The v1 Gaussian sampler consumes an explicit RNG state, requires a finite mean vector and SPD covariance accepted by Cholesky, and produces one dense sample vector. It generates on a cloned state and advances the caller state only after output allocation succeeds. Each dimension consumes exactly four raw PCG32 outputs because normal variates use Box–Muller without caching.
