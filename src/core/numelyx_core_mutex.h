/**
 * @file numelyx_core_mutex.h
 * @brief Recursive per-object mutex abstraction for internal synchronization.
 */
#ifndef NUMELYX_CORE_MUTEX_H
#define NUMELYX_CORE_MUTEX_H

#ifdef _WIN32
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
# endif
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <windows.h>
typedef struct { CRITICAL_SECTION native; } numelyx_core_mutex;
#else
# include <pthread.h>
typedef struct { pthread_mutex_t native; } numelyx_core_mutex;
#endif

int numelyx_core_mutex_init(numelyx_core_mutex *mutex);
void numelyx_core_mutex_destroy(numelyx_core_mutex *mutex);
int numelyx_core_mutex_lock(numelyx_core_mutex *mutex);
int numelyx_core_mutex_unlock(numelyx_core_mutex *mutex);

#endif
