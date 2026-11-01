# Numerus Source Architecture

## Current layout

The implementation is organized under `src/`; extension configuration and the PHP module header remain at the repository root.

```text
src/
├── numerus.c
├── core/
│   ├── numerus_numeric.c
│   └── numerus_size.h
├── storage/
│   └── numerus_storage.c
├── matrix/
│   ├── numerus_matrix.c
│   ├── numerus_matrix_view_nodes.c
│   ├── numerus_matrix_constructors.c
│   ├── focused operation and linear-algebra translation units
│   ├── numerus_matrix_internal.h
│   ├── numerus_matrix_types.h
│   ├── numerus_matrix_core.h
│   ├── numerus_matrix_constructors.h
│   ├── numerus_matrix_views.h
│   ├── numerus_matrix_operations.h
│   ├── numerus_matrix_linalg.h
│   └── numerus_matrix.h
└── statistics/
    ├── numerus_rng.c
    ├── numerus_rng_sampling.c
    ├── numerus_probability.c
    └── numerus_optimizer.c
```

Tests, benchmarks, examples, and documentation remain in their dedicated top-level directories. `config.m4` and `config.w32` enumerate the extension sources. Linux CI covers PHP 8.2–8.5 and a debug build; Windows runtime support remains unverified.

## Responsibility boundaries

- **Core:** checked size arithmetic, scalar numeric primitives, and extension bootstrap.
- **Storage:** immutable, opaque physical representations.
- **Matrix:** logical Matrix lifecycle, lazy views, representation factories, arithmetic, and numerical linear algebra.
- **Statistics:** explicit-state random generation, sampling, probability primitives, and generic optimization.
- **Focused Matrix headers:** shared types; lifecycle/access; constructors; views; operations; linear algebra/analysis. `numerus_matrix.h` is an umbrella header so existing C consumers can keep a single include.
- **Internal Matrix header:** private representation, allocator policy, and cross-translation-unit implementation helpers. It is not the supported public declaration surface.

## Matrix implementation split

The former large Matrix implementation is separated by cohesive responsibility:
- `numerus_matrix.c` retains lifecycle, generic root/parent construction, scalar-transform support, access, cached analysis and destruction.
- `numerus_matrix_view_nodes.c` owns internal lazy-view nodes, coordinate transforms, joins, and view composition.
- `numerus_matrix_constructors.c` owns representation factories and row/column vector constructors.
- Existing focused translation units continue to own arithmetic, decomposition, solving, storage and statistical composition.

The concrete `struct numerus_matrix` has one authoritative definition in `numerus_matrix_internal.h`. Allocation initialization is shared through an internal helper. Parent pointers remain non-owning; this refactor does not change their lifetime contract, cache behavior, allocator choice, or error/status semantics.

## Compatibility headers

Root-level `numerus_*.h` files are forwarding headers retained for existing native tests and C include paths during the directory migration. They contain no duplicate declarations or implementations. New internal code should include the focused headers under `src/`; the forwarding headers can be removed only after all consumers have migrated.

## Build and validation

Every structural change is intended to preserve numerical and runtime behavior. CI validates:
- Unix extension build and extension loading;
- PHPT on PHP 8.2–8.5 plus debug build;
- strict C11 native tests with `-Wall -Wextra -Wpedantic -Werror`;
- property tests and ASan/UBSan runs;
- the runnable statistical C example;
- independent compilation of each focused Matrix header and the umbrella header.

The Windows manifest is kept aligned with the source list, but a manifest update is not evidence of a successful Windows build. No Windows support claim should be made until Windows CI passes.
