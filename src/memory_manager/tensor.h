#pragma once
#include <cstdint>
#include "types.h"
#include "buffer_allocator.h"
#include "utils.h"

// Manuplate access pointers based on convinience
typedef union PtrUnion
{
    int32_t *i32;
    float *f;
    char *raw;
    int8_t *int8;
    qi8_256 *qi8;
    uint8_t *ui8;
    /* Only use this member if using externally to HNN */
    void *data;
} PtrUnion;

class Tensor
{
public:
    // Stores the preallocated buffer pointer
    Tensor(DTYPE data_type, char *data, size_t dimension)
    {
        this->data_type = data_type;
        this->ptr.raw = data;
        this->dimension = dimension;
    }

    // Constructor to allocate memory during object creation
    Tensor(DTYPE data_type, size_t dimension)
    {
        this->ptr.raw = static_cast<char *>(BufferAllocator::allocate_mem(dimension, getDtypeSize(data_type)));
        this->data_type = data_type;
        this->dimension = dimension;
    }

    // Custom data type tensor allocation
    Tensor(DTYPE data_type, size_t dimension, size_t single_size)
    {
        this->ptr.raw = static_cast<char *>(BufferAllocator::allocate_mem(dimension, single_size));
        this->data_type = data_type;
        this->dimension = dimension;
    }

    template <typename T>
    T *getData()
    {
        return reinterpret_cast<T *>(this->ptr.raw);
    }

    DTYPE getDType()
    {
        return this->data_type;
    }

    size_t getDim()
    {
        return this->dimension;
    }

private:
    DTYPE data_type;
    PtrUnion ptr;
    size_t dimension;
};
