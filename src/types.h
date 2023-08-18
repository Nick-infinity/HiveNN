#pragma once
#include <cstdint>
#include <cstddef>
#include <iostream>

#define BLOCK_SIZE 256

typedef struct
{
    float scale = 0.0f;                // scale
    int8_t quantized_vals[BLOCK_SIZE]; // quants with 0 zeropoint
} qi8_256;

typedef struct
{
    float min;
    float scale;
    uint8_t quantized_vals[];
} qu8_assym;

enum DTYPE
{
    CUSTOM = -1,
    INT32 = 0,
    FLOAT32,
    FLOAT16,
    INT8,
    UINT8,
    CHAR,
    QI8_256,
    QU8_ASSYM,
    VOID,
};

enum HNNStatus
{
    OK = 0,
    FAIL = 1,
    READ_ERROR = 2,
    WRITE_ERROR = 3,
    MEM_ALLOC_ERROR = 4,
};

enum CONFIG_TYPE
{
    NONE = -1,
    LLAMA2 = 0,
};

typedef struct
{
    int dimension;
    int hidden_dimension;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;
    int sequence_len;
} Mconfig;
