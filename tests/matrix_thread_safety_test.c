#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include "test_threads.h"

typedef struct {
    numelyx_core_matrix *matrix;
    int failed;
} worker_args;

static int cache_worker(void *argument)
{
    worker_args *args = (worker_args *) argument;
    size_t iteration;

    for (iteration = 0; iteration < 80; iteration++) {
        numelyx_core_matrix_flags flags;
        numelyx_core_matrix *inverse = NULL;
        double determinant = 0.0;

        if (numelyx_core_matrix_get_flags(args->matrix, &flags) !=
                NUMERUS_MATRIX_SUCCESS ||
            numelyx_core_matrix_determinant(args->matrix, &determinant) !=
                NUMERUS_MATRIX_SUCCESS ||
            fabs(determinant - 18.0) > 1e-12 ||
            numelyx_core_matrix_inverse(args->matrix, &inverse) !=
                NUMERUS_MATRIX_SUCCESS ||
            inverse == NULL) {
            args->failed = 1;
            numelyx_core_matrix_destroy(inverse);
            break;
        }

        numelyx_core_matrix_destroy(inverse);
    }

    numelyx_core_matrix_destroy(args->matrix);
    return 0;
}

int main(void)
{
    enum { THREAD_COUNT = 8 };
    const double values[] = {
        4.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0
    };
    numelyx_core_matrix *matrix = NULL;
    worker_args arguments[THREAD_COUNT];
    test_thread_t threads[THREAD_COUNT];
    size_t index;

    assert(numelyx_core_matrix_create_dense(3, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);

    for (index = 0; index < THREAD_COUNT; index++) {
        arguments[index].matrix = matrix;
        arguments[index].failed = 0;
        assert(numelyx_core_matrix_retain(matrix) == NUMERUS_MATRIX_SUCCESS);
        assert(test_thread_create(
            &threads[index], cache_worker, &arguments[index]
        ));
    }

    for (index = 0; index < THREAD_COUNT; index++) {
        int result = -1;
        assert(test_thread_join(&threads[index], &result));
        assert(result == 0);
        assert(arguments[index].failed == 0);
    }

    numelyx_core_matrix_destroy(matrix);
    return 0;
}
