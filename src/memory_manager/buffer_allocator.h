#pragma once
#include "utils.h"
#include "types.h"
/**
 * Buffer allocator is to allocate memory from the platform
 */

class BufferAllocator
{
public:
    /**
     *  Allocate mem based on platform architecture
     */
    static void *allocate_mem(size_t nitems, size_t size);
    // Keep track of total allocated mem for profiler
    static float total_allocated;
};
