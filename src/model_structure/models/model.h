#pragma once
#include "utils.h"
#include "memory_manager/buffer_manager.h"
#include "common/runcontext.h"
#include "kernels/cpu/operators.h"
#include "src/common/runcontext.h"
class Model
{
public:
    /**
     * All model and tensor related preprations are done here
     */
    virtual HNNStatus open(char* &model_path, char*& tokenizer_path, char* &input_data)  = 0;
    /**
     * Model is executed
     */
    virtual HNNStatus execute()  = 0;
    /**
     * Model is closed and memory is freed
     */
    virtual HNNStatus close() = 0;

    RunContext* ctx = nullptr;
};