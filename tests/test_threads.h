/**
 * @file test_threads.h
 * @brief Small POSIX/Windows thread adapter for native concurrency tests.
 */
#ifndef NUMELYX_TEST_THREADS_H
#define NUMELYX_TEST_THREADS_H

#include <stdlib.h>

#ifdef _WIN32
# include <windows.h>
# include <process.h>
typedef struct { HANDLE handle; } test_thread_t;
typedef int (*test_thread_fn)(void *);
typedef struct { test_thread_fn function; void *argument; } test_thread_start;

static unsigned __stdcall test_thread_entry(void *pointer)
{
    test_thread_start *start = (test_thread_start *) pointer;
    int result = start->function(start->argument);
    free(start);
    return (unsigned) result;
}

static int test_thread_create(
    test_thread_t *thread,
    test_thread_fn function,
    void *argument
)
{
    test_thread_start *start;
    if (thread == NULL || function == NULL) return 0;
    start = (test_thread_start *) malloc(sizeof(*start));
    if (start == NULL) return 0;
    start->function = function;
    start->argument = argument;
    thread->handle = (HANDLE) _beginthreadex(
        NULL, 0, test_thread_entry, start, 0, NULL
    );
    if (thread->handle == NULL) {
        free(start);
        return 0;
    }
    return 1;
}

static int test_thread_join(test_thread_t *thread, int *result)
{
    DWORD exit_code;
    if (thread == NULL || thread->handle == NULL) return 0;
    if (WaitForSingleObject(thread->handle, INFINITE) != WAIT_OBJECT_0) return 0;
    if (!GetExitCodeThread(thread->handle, &exit_code)) return 0;
    (void) CloseHandle(thread->handle);
    thread->handle = NULL;
    if (result != NULL) *result = (int) exit_code;
    return 1;
}
#else
# include <pthread.h>
# include <stdint.h>
typedef pthread_t test_thread_t;
typedef int (*test_thread_fn)(void *);
typedef struct { test_thread_fn function; void *argument; } test_thread_start;

static void *test_thread_entry(void *pointer)
{
    test_thread_start *start = (test_thread_start *) pointer;
    int result = start->function(start->argument);
    free(start);
    return (void *) (intptr_t) result;
}

static int test_thread_create(
    test_thread_t *thread,
    test_thread_fn function,
    void *argument
)
{
    test_thread_start *start;
    int status;
    if (thread == NULL || function == NULL) return 0;
    start = (test_thread_start *) malloc(sizeof(*start));
    if (start == NULL) return 0;
    start->function = function;
    start->argument = argument;
    status = pthread_create(thread, NULL, test_thread_entry, start);
    if (status != 0) free(start);
    return status == 0;
}

static int test_thread_join(test_thread_t *thread, int *result)
{
    void *thread_result = NULL;
    if (thread == NULL || pthread_join(*thread, &thread_result) != 0) return 0;
    if (result != NULL) *result = (int) (intptr_t) thread_result;
    return 1;
}
#endif

#endif
