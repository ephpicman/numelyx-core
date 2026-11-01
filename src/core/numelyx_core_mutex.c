/**
 * @file numelyx_core_mutex.c
 * @brief Platform-backed recursive mutex implementation.
 */
#if !defined(_WIN32)
# define _POSIX_C_SOURCE 200809L
#endif
#include "numelyx_core_mutex.h"

int numelyx_core_mutex_init(numelyx_core_mutex *mutex)
{
    if (mutex == 0) return 0;
#ifdef _WIN32
    return InitializeCriticalSectionEx(&mutex->native, 0, 0) != 0;
#else
    {
        pthread_mutexattr_t attributes;
        int result;
        if (pthread_mutexattr_init(&attributes) != 0) return 0;
        result = pthread_mutexattr_settype(&attributes, PTHREAD_MUTEX_RECURSIVE);
        if (result == 0) result = pthread_mutex_init(&mutex->native, &attributes);
        (void) pthread_mutexattr_destroy(&attributes);
        return result == 0;
    }
#endif
}

void numelyx_core_mutex_destroy(numelyx_core_mutex *mutex)
{
    if (mutex == 0) return;
#ifdef _WIN32
    DeleteCriticalSection(&mutex->native);
#else
    (void) pthread_mutex_destroy(&mutex->native);
#endif
}

int numelyx_core_mutex_lock(numelyx_core_mutex *mutex)
{
    if (mutex == 0) return 0;
#ifdef _WIN32
    EnterCriticalSection(&mutex->native);
    return 1;
#else
    return pthread_mutex_lock(&mutex->native) == 0;
#endif
}

int numelyx_core_mutex_unlock(numelyx_core_mutex *mutex)
{
    if (mutex == 0) return 0;
#ifdef _WIN32
    LeaveCriticalSection(&mutex->native);
    return 1;
#else
    return pthread_mutex_unlock(&mutex->native) == 0;
#endif
}
