cmake_minimum_required(VERSION 3.16)

project(NumelyxCore VERSION 0.1.0 LANGUAGES C)

option(NUMELYX_BUILD_TESTS "Build native C tests" ON)
option(NUMELYX_BUILD_BENCHMARKS "Build diagnostic benchmarks" OFF)
option(NUMELYX_ENABLE_TSAN "Enable ThreadSanitizer (Linux, GCC/Clang)" OFF)
option(NUMELYX_ENABLE_ASAN_UBSAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer (Linux, GCC/Clang)" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

file(GLOB_RECURSE NUMELYX_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/storage/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/statistics/*.c"
)

add_library(numelyx_core STATIC ${NUMELYX_CORE_SOURCES})
add_library(Numelyx::Core ALIAS numelyx_core)

find_package(Threads REQUIRED)
target_link_libraries(numelyx_core PUBLIC Threads::Threads)

if(NUMELYX_ENABLE_TSAN)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "NUMELYX_ENABLE_TSAN is supported only on Linux")
    endif()
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "NUMELYX_ENABLE_TSAN requires GCC or Clang")
    endif()

    target_compile_options(numelyx_core PUBLIC
        -fsanitize=thread
        -fno-omit-frame-pointer
        -g
    )
    target_link_options(numelyx_core PUBLIC -fsanitize=thread)
endif()

if(NUMELYX_ENABLE_TSAN AND NUMELYX_ENABLE_ASAN_UBSAN)
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with Address/UndefinedBehaviorSanitizer")
endif()

if(NUMELYX_ENABLE_ASAN_UBSAN)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "NUMELYX_ENABLE_ASAN_UBSAN is supported only on Linux")
    endif()
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "NUMELYX_ENABLE_ASAN_UBSAN requires GCC or Clang")
    endif()

    target_compile_options(numelyx_core PUBLIC
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-sanitize-recover=all
        -g
    )
    target_link_options(numelyx_core PUBLIC
        -fsanitize=address,undefined
        -fno-sanitize-recover=all
    )
endif()

if(MSVC)
    target_compile_options(numelyx_core PUBLIC /experimental:c11atomics)
endif()

target_include_directories(numelyx_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/storage"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/statistics"
)


if(UNIX)
    target_link_libraries(numelyx_core PUBLIC m)
endif()

if(NUMELYX_BUILD_TESTS)
    enable_testing()
    file(GLOB NUMELYX_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*_test.c"
    )

    set(NUMELYX_ALLOC_WRAP_TESTS
        matrix_allocation_failure_test
        storage_allocation_failure_test
    )

    foreach(test_source IN LISTS NUMELYX_TEST_SOURCES)
        get_filename_component(test_name "${test_source}" NAME_WE)

        # Matrix and Storage allocation injection uses GNU ld --wrap.
        if(test_name IN_LIST NUMELYX_ALLOC_WRAP_TESTS
           AND NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND UNIX AND NOT APPLE))
            message(STATUS "Skipping ${test_name}: GNU ld --wrap is required")
            continue()
        endif()

        add_executable(${test_name} "${test_source}")
        target_link_libraries(${test_name} PRIVATE Numelyx::Core)

        if(test_name IN_LIST NUMELYX_ALLOC_WRAP_TESTS)
            target_link_options(${test_name} PRIVATE
                "-Wl,--wrap=malloc"
                "-Wl,--wrap=calloc"
                "-Wl,--wrap=free"
            )
        elseif(test_name MATCHES "^(gaussian_log_density_test|gaussian_sampling_test)$")
            target_sources(${test_name} PRIVATE src/statistics/numelyx_core_probability.c)
            target_compile_definitions(${test_name} PRIVATE NUMERUS_PROBABILITY_TEST_ALLOCATOR)
        elseif(test_name STREQUAL "optimizer_test")
            target_sources(${test_name} PRIVATE src/statistics/numelyx_core_optimizer.c)
            target_compile_definitions(${test_name} PRIVATE NUMERUS_OPTIMIZER_TEST_ALLOCATOR)
        elseif(test_name MATCHES "^(rng_sampling_test|rng_state_test)$")
            target_sources(${test_name} PRIVATE
                src/statistics/numelyx_core_rng.c
                src/statistics/numelyx_core_rng_sampling.c
            )
            target_compile_definitions(${test_name} PRIVATE NUMERUS_RNG_TEST_ALLOCATOR)
        endif()

        if(test_name STREQUAL "lifetime_test")
            target_link_libraries(${test_name} PRIVATE Threads::Threads)
        endif()

        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
    endforeach()
endif()

if(NUMELYX_BUILD_BENCHMARKS)
    add_executable(matrix_benchmark benchmarks/matrix_benchmark.c)
    target_link_libraries(matrix_benchmark PRIVATE Numelyx::Core)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND UNIX AND NOT APPLE)
        target_link_options(matrix_benchmark PRIVATE
            "-Wl,--wrap=malloc"
            "-Wl,--wrap=calloc"
            "-Wl,--wrap=free"
        )
    endif()
endif()
