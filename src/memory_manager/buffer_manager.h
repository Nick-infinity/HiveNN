#pragma once

#include <iostream>
#include <vector>
#include "tensor.h"
#include "model_structure/configs/transformers_config.h"
#include "types.h"
#include "buffer_allocator.h"
#include "utils.h"

class BufferManager
{
public:
    /**
     * Initialize tensor and their memory for a model
     */
    HNNStatus initialize_tensors(char *weights, Config *mc);

    Tensor **getSharedTensors()
    {
        return this->shared_tensors;
    }

    Tensor **getWeightTensors()
    {
        return this->weight_tensors;
    }

private:
    // Allocate Tensors for Shared Buffers and weights for LLAMA2
    Tensor *shared_tensors[(size_t)RUNTIME::LENGTH];
    Tensor *weight_tensors[(size_t)WEIGHTS::LENGTH];
};
