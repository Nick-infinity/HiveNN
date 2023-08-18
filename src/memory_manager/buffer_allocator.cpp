#include "buffer_allocator.h"
#include "stdlib.h"
#include "utils.h"
#include "types.h"

/**
 * We use calloc for now. It can be replaced with platform specific memory (HardwareBuffer/Ion/DMABuf etc.)
 */
void *BufferAllocator::allocate_mem(size_t nitems, size_t size)
{
    void *ret = calloc(nitems, size);
    if (nullptr == ret)
    {
        ELOG("Buffer allocation failed %d\n", HNNStatus::MEM_ALLOC_ERROR);
        return nullptr;
    }
    DLOG("NITEMS %d , SIZE %d\n", nitems, size);
    total_allocated += (nitems * size);
    DLOG("Total Memory Allocated: %f MB \n", total_allocated / 1024 / 1024);
    return ret;
}

// Keeps track of total allocated memory by the application
float BufferAllocator::total_allocated = 0;
