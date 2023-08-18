#pragma once
#include <stdio.h>
#include <cstdint>
#include "types.h"

// Small controllable logging 
#ifdef DEBUG_BUILD
#define DLOG(...) printf("[DEBUG] " __VA_ARGS__, __LINE__)
#else
#define DLOG(...) do {} while (0)
#endif

#define ILOG(...) printf("[INFO] " __VA_ARGS__, __LINE__)

// #define ELOG(format, ...) printf("HNN : INFO: " format, __VA_ARGS__)
#define ELOG(...) printf("Line %d: [ERROR] " __VA_ARGS__, __LINE__, __FILE__)

#define PROFLOG(...) printf("[PROFILER] " __VA_ARGS__, __LINE__)

#define ENSURE_STATUS(ret) \
    if ((ret) != 0)        \
    ELOG("")

size_t getOffset(DTYPE data_type, int num_layers, int per_layer_size);

template <typename T>
size_t st(T value)
{
    return static_cast<size_t>(value);
}

// TODO:: Add support for float16
size_t getDtypeSize(DTYPE dtype);