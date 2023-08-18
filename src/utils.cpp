#include "utils.h"

// Do not inline , ndk r21c doesnt support inlining correctlty
size_t getOffset(DTYPE data_type, int num_layers, int per_layer_size)
{
    size_t offset = 0;
    if (data_type == DTYPE::QU8_ASSYM)
    {
        offset = num_layers * ((sizeof(float) + sizeof(float) + per_layer_size));
    }
    else if (data_type == DTYPE::QI8_256)
    {
        offset = num_layers * (per_layer_size + (per_layer_size / BLOCK_SIZE) * sizeof(float));
    }
    else if (data_type == DTYPE::FLOAT32)
    {
        offset = per_layer_size * num_layers * sizeof(float);
    }
    // DLOG("Dtype %d, num_layers %d, per_layer_size %d , offset %d \n", data_type, num_layers, per_layer_size, offset);
    return offset;
}

// TODO:: Add support for float16
size_t getDtypeSize(DTYPE dtype)
{
    switch (dtype)
    {
    case INT32:
        return sizeof(int32_t);
        break;
    case FLOAT32:
        return sizeof(float);
        break;
    case INT8:
        return sizeof(int8_t);
        break;
    case UINT8:
        return sizeof(uint8_t);
        break;
    case CHAR:
        return sizeof(char);
        break;
    case QI8_256:
        return sizeof(qi8_256);
        break;
    case QU8_ASSYM:
        return sizeof(qu8_assym);
        break;
    default:
        ELOG("Unknown DataType\n");
        break;
    }
}