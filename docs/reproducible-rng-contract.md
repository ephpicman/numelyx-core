# Reproducible RNG Contract

Status: accepted v1 contract, implemented by #115–#117. See [current statistical foundation capabilities](statistical-foundation.md) for the integrated API and example.

## Algorithm

Use **PCG-XSH-RR 64/32** (PCG32) as the v1 deterministic simulation generator.

- State transition: (state_{n+1} = state_n 	imes 6364136223846793005 + increment pmod {2^{64}}).
- Output permutation: PCG XSH-RR 64/32 as specified by the reference algorithm (xorshifted old state, then rotate-right by the top five bits of the old state).
- Arithmetic uses `uint64_t` and `uint32_t`. Unsigned 64-bit wraparound is intentional and defined by C; no signed overflow is permitted.
- This is a simulation/numerical generator, **not cryptographic randomness**. It must never be advertised for secrets, tokens, or security decisions.
- No external RNG dependency is needed.

## Seed and stream contract

The constructor accepts two 64-bit values: `seed` (initial state material) and `stream` (sequence selector).

Use the standard PCG seeding procedure, in this order:

1. Set internal state to zero.
2. Set the odd increment to `(stream << 1) | 1` using unsigned arithmetic.
3. Advance the generator once.
4. Add `seed` to the state using unsigned 64-bit arithmetic.
5. Advance the generator once more.
6. The next output is the first value returned to the caller.

The uppermost bit of `stream` cannot affect the odd increment after the left shift; document this property rather than claiming all 64 stream bits are independently represented. Different stream values select different sequence parameters, but Numerus must not claim formal statistical independence solely because streams or seeds differ. A stream-splitting/jump facility is a separate P1 decision (#RNG-04).

## C state lifecycle

The state is opaque to callers. The implemented internal C API is declared in `src/statistics/numerus_rng.h`: 

```c
typedef struct numerus_rng numerus_rng;

numerus_rng_status numerus_rng_create(
    uint64_t seed,
    uint64_t stream,
    numerus_rng **rng
);

numerus_rng_status numerus_rng_next_u32(
    numerus_rng *rng,
    uint32_t *value
);

numerus_rng_status numerus_rng_clone(
    const numerus_rng *source,
    numerus_rng **clone
);

void numerus_rng_destroy(numerus_rng *rng);
```

The names shown above match the current implementation; treat changes to these declarations as internal C API changes and update this contract and its tests together.

- Create/clone output pointers are required and set to NULL before work; outputs are published only on success.
- A clone copies the exact generator state and increment. The next raw value from the clone must equal the next raw value from the source; advancing one must not mutate the other.
- `next_u32` consumes exactly one state transition and writes the output only on success.
- Destroy accepts NULL and is safe to call on a null pointer.
- State memory is owned by the RNG object and allocated/freed with the extension's allocator conventions; standalone tests use the repository's allocator switch pattern.
- The initial v1 contract supports replay from seed/stream and in-memory cloning. Portable serialized state is deferred; do not serialize the opaque struct bytes or promise a stable binary layout.
- A state object is mutable and must not be shared concurrently without caller synchronization. Separate instances may be used concurrently; the implementation must have no hidden global mutable RNG state.

## Raw output and reproducibility guarantees

- The first raw outputs for `seed = 42`, `stream = 54` must be:

  `a15c02b7, 7b47f409, ba1d3330, 83d2f293, bfa4784b, cbed606e, bfc6a3ad, 812fff6d`

- The vector is the reference PCG32 sequence after the seeding procedure above. Native tests must compare exact 32-bit values, not statistical approximations.
- The raw `uint32_t` sequence is bit-for-bit stable for a fixed algorithm version, seed, stream and call sequence across supported platforms.
- Any future algorithm change that alters this sequence requires an explicit algorithm/version change and a compatibility note. Do not silently change the generator behind the same name.
- Uniform conversion and distribution transforms are specified in the implementation contract below and covered by the native variate tests. Uniform mapping uses fixed bit extraction and arithmetic, not implementation-dependent random APIs.
- Normal variates may use `libm` functions. Unless Numerus ships and tests deterministic math implementations, it must **not** promise bit-for-bit identical floating-point normal outputs across different libm/compiler/platform combinations. The contract distinguishes same-platform repeatability from cross-platform raw-stream reproducibility.
- Reproducibility assumes supported platforms provide 32-bit and 64-bit unsigned integer types and the extension's supported floating-point representation; tests should make these assumptions explicit where relevant.

## Error and failure semantics

- Null required pointers return the RNG-specific invalid-argument status; invalid outputs remain unchanged except constructors/cloners, whose output handle is initialized to NULL.
- Allocation failure returns a distinct out-of-memory status.
- Raw generation itself must not allocate and must not fail after validating the state/output pointers.
- Do not use NaN or a sentinel random number to signal errors.
- RNG status types should be narrowly scoped and must not reuse Matrix status values accidentally.

## Native test coverage for #115

1. Exact reference vector above.
2. Same seed and stream produce the same sequence across independently created states.
3. Different seed or stream fixtures produce the documented distinct reference sequences, without asserting statistical independence.
4. Clone preserves next output and independent advancement.
5. Null arguments, output preservation, allocation failure, destroy-NULL and repeated lifecycle tests.
6. Compile with strict C warnings and the standalone allocator configuration; run under the repository's debug/sanitizer jobs when available.
7. No global state: interleaved calls on two instances match their independently generated sequences.

## Scope decisions

- Accepted: PCG-XSH-RR 64/32, explicit seed/stream, opaque state, clone, raw output vector, raw-stream reproducibility.
- Deferred: serialized-state format, jump-ahead/stream splitting, cryptographic RNG, broad distribution catalog.
- #116 owns uniform endpoint/mapping and normal transform details. Distribution output guarantees must not be implied by the raw PRNG guarantee.


## Uniform and normal variate contract (#116)

- `numerus_rng_uniform()` consumes exactly two raw outputs, forms 53 bits as `(high << 21) | (low >> 11)`, and divides by (2^{53}). The result is in ([0,1)); zero is allowed and one is excluded.
- `numerus_rng_normal()` uses Box–Muller without caching. The first 53-bit value is mapped to ((0,1]) by adding one before division by (2^{53}), so `log(0)` is impossible. The next 53-bit value is mapped to ([0,1)) for the angle. Each valid normal call consumes exactly four raw outputs.
- Raw uniform conversion is bit-defined for binary64; normal values are expected to replay on the same supported runtime but are not promised bit-for-bit identical across different `libm` implementations.
- No hidden cached normal value or global state exists. Invalid pointers do not advance the RNG or modify outputs. A valid transform publishes the output only after the computed value is finite.


## Index sampling contract (#117)

- `numerus_rng_sample_indices()` requires a non-empty population. A zero-sized sample succeeds with a NULL result and consumes no RNG output. A sample without replacement cannot exceed the population size.
- With replacement, each result is drawn uniformly from `[0, population_size)` using rejection sampling over 64-bit values to avoid modulo bias.
- Without replacement, Algorithm S selects a uniform subset using (O(N)) time and (O(k)) storage; Fisher–Yates then randomizes the returned order. The output contains no duplicates.
- Result arrays are allocated by Numerus and must be released with `numerus_rng_free_indices()`. Size overflow and allocation failure leave the output NULL and do not advance the RNG state.
- Determinism is defined for the fixed algorithm version, seed/stream, sampling arguments and call sequence. The exact test vectors live in `tests/rng_sampling_test.c`.
