# Core Thread-Safety and Object Lifetime

## Shared object ownership

Opaque heap objects that support shared ownership use the common
`numelyx_core_lifetime` primitive:

- Matrix
- Storage
- LU factorization
- RNG state
- Optimizer result

Each newly created object starts with one owned reference. A thread must hold
its own reference for the complete duration of every operation that accesses
the object. Call the object's `retain` function before transferring or sharing
an additional ownership reference, and release that reference with its
corresponding `destroy` function.

A retain operation requires the caller to already hold a valid reference (or
another external guarantee that prevents reclamation). Atomic reference
counting does not make it valid to retain a stale pointer, and it does not
protect an object from a concurrent final release when the caller owns no
reference.

The final release invokes the object's destructor exactly once. Destructors
release the references they own to dependencies. Borrowed pointers and
callback contexts are not retained and must remain valid under their
documented borrowing contracts.

## Matrix caches

Each Matrix owns one recursive mutex shared by all of its lazy numerical
caches. Cache lookup, computation, and publication are serialized under that
mutex. This intentionally serializes different cache requests for the same
Matrix, while allowing independent Matrix objects to compute concurrently.

The recursive mutex supports nested cache helpers on the same Matrix (for
example, determinant calculation requesting a cached LU factorization).
It is not a global lock and does not impose a FIFO scheduling guarantee.

Matrix values and dimensions remain immutable after construction. The mutex
protects mutable cache state; it does not make caller-owned transform contexts
or callback state thread-safe. Callbacks and borrowed contexts used by a
Matrix must be deterministic, read-only, and safe for concurrent invocation.
They must not create cross-Matrix cache-lock cycles by entering cache operations
on another Matrix while a cache computation is active.

## Other shared objects

- Storage is immutable; concurrent readers are safe while each holds a
  reference.
- LU factorizations are immutable after construction; concurrent readers are
  safe while each holds a reference.
- Optimizer results are immutable after publication; concurrent getters are
  safe while each holds a reference.
- RNG state is mutable. Its public state-changing operations and composite
  sampling operation are serialized by a per-RNG mutex. Independent RNG objects
  remain concurrent. Concurrent calls on one RNG are safe, but their order is
  determined by mutex scheduling and is not a reproducibility guarantee across
  different thread schedules.

## Scope

Thread safety is an object-level contract, not a claim that arbitrary external
state is synchronized. Applications remain responsible for synchronization of
shared callback contexts, raw buffers, and any data outside these opaque Core
objects. A caller must never destroy its only reference while another thread
is using the object without holding a separate retained reference.
