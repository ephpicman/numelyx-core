#include "numelyx_core_lifetime.h"

#include <assert.h>
#include <stdatomic.h>
#include <stddef.h>
#include "test_threads.h"

typedef struct {
    numelyx_core_lifetime lifetime;
    atomic_size_t operations;
    atomic_size_t *destroy_count;
} test_object;

static void destroy_test_object(void *pointer)
{
    test_object *object = (test_object *) pointer;
    atomic_fetch_add_explicit(object->destroy_count, 1, memory_order_relaxed);
}

static int worker(void *argument)
{
    test_object *object = (test_object *) argument;
    size_t index;

    /* The worker's retained reference protects the object for this loop. */
    assert(numelyx_core_lifetime_retain(&object->lifetime) ==
        NUMELYX_CORE_LIFETIME_SUCCESS);

    for (index = 0; index < 20000; index++) {
        assert(numelyx_core_lifetime_retain(&object->lifetime) ==
            NUMELYX_CORE_LIFETIME_SUCCESS);
        atomic_fetch_add_explicit(
            &object->operations, 1, memory_order_relaxed
        );
        assert(numelyx_core_lifetime_release(
            &object->lifetime, object
        ) == NUMELYX_CORE_LIFETIME_SUCCESS);
    }

    assert(numelyx_core_lifetime_release(
        &object->lifetime, object
    ) == NUMELYX_CORE_LIFETIME_SUCCESS);
    return 0;
}

int main(void)
{
    enum { THREAD_COUNT = 8 };
    test_object object;
    atomic_size_t destroy_count;
    test_thread_t threads[THREAD_COUNT];
    size_t index;

    atomic_init(&destroy_count, 0);
    atomic_init(&object.operations, 0);
    object.destroy_count = &destroy_count;

    assert(numelyx_core_lifetime_init(
        &object.lifetime, destroy_test_object
    ) == NUMELYX_CORE_LIFETIME_SUCCESS);

    for (index = 0; index < THREAD_COUNT; index++) {
        assert(test_thread_create(&threads[index], worker, &object));
    }
    for (index = 0; index < THREAD_COUNT; index++) {
        int result = -1;
        assert(test_thread_join(&threads[index], &result));
        assert(result == 0);
    }

    assert(atomic_load_explicit(
        &object.operations, memory_order_relaxed
    ) == THREAD_COUNT * 20000);
    assert(atomic_load_explicit(
        &destroy_count, memory_order_relaxed
    ) == 0);

    assert(numelyx_core_lifetime_release(
        &object.lifetime, &object
    ) == NUMELYX_CORE_LIFETIME_SUCCESS);
    assert(atomic_load_explicit(
        &destroy_count, memory_order_relaxed
    ) == 1);

    return 0;
}
