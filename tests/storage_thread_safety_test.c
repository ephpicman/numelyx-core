#include "../numelyx_core_storage.h"

#include <assert.h>
#include <stddef.h>
#include <threads.h>

typedef struct {
    numelyx_core_storage *storage;
    int failed;
} storage_worker_args;

static int storage_worker(void *argument)
{
    storage_worker_args *args = (storage_worker_args *) argument;
    size_t iteration;

    for (iteration = 0; iteration < 10000; iteration++) {
        double value = -1.0;
        if (numelyx_core_storage_get(
                args->storage, iteration % 3, (iteration / 3) % 3, &value
            ) != NUMERUS_STORAGE_SUCCESS ||
            value != (double) ((iteration % 3) * 3 + ((iteration / 3) % 3))) {
            args->failed = 1;
            break;
        }
    }

    numelyx_core_storage_destroy(args->storage);
    return 0;
}

int main(void)
{
    enum { THREAD_COUNT = 8 };
    const double values[] = {0,1,2,3,4,5,6,7,8};
    numelyx_core_storage *storage = NULL;
    storage_worker_args arguments[THREAD_COUNT];
    thrd_t threads[THREAD_COUNT];
    size_t index;

    assert(numelyx_core_storage_create_dense(3, 3, values, &storage) ==
        NUMERUS_STORAGE_SUCCESS);

    for (index = 0; index < THREAD_COUNT; index++) {
        arguments[index].storage = storage;
        arguments[index].failed = 0;
        assert(numelyx_core_storage_retain(storage) == NUMERUS_STORAGE_SUCCESS);
        assert(thrd_create(&threads[index], storage_worker, &arguments[index]) ==
            thrd_success);
    }
    for (index = 0; index < THREAD_COUNT; index++) {
        int result = -1;
        assert(thrd_join(threads[index], &result) == thrd_success);
        assert(result == 0);
        assert(arguments[index].failed == 0);
    }

    numelyx_core_storage_destroy(storage);
    return 0;
}
