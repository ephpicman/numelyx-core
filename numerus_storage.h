/**
 * @file numerus_storage.h
 * @brief Compatibility include forwarding Storage declarations to src/storage/numerus_storage.h.
 *
 * @details Compatibility-only header retained for existing native consumers.
 * New internal code should include the focused header under src/ directly.
 */

/* Compatibility include for native consumers; implementation lives under src/. */
#ifndef NUMERUS_STORAGE_ROOT_COMPAT_H
#define NUMERUS_STORAGE_ROOT_COMPAT_H
#include "src/storage/numerus_storage.h"
#endif
