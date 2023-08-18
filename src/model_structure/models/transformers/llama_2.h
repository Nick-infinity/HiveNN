#pragma once
#include "kernels/cpu/operators.h"
#include "model_structure/models/model.h"
#include "types.h"
#include "model_structure/configs/transformers_config.h"
#include "memory_manager/tensor.h"
class Llama2Model : public Model
{
public:
    HNNStatus open(char *&model_path, char *&tokenizer_path, char *&input_data);
    HNNStatus execute();
    HNNStatus close();

private:
    /**
     * Main llama 2 execute function
     */
    HNNStatus llama_2_execute(TransformerConfig *&config, int &pos, Tensor **&sht, Tensor **&wt, int &token);
    /**
     * Llama 2 uses RoPE rotation . Decode it
     */
    inline void rotationRoPE(int dimension, int head_size, int current_step, float *qData, float *kData, int kv_dim);

    /**
     * Run the operators that are common for each all the layers
     */
    // TODO:: Reduce the params. Currently full config passing gives slower perf as compared to inidividual variable passing
    inline void multi_layer_process(const int &l, Tensor **&sht, Tensor **&wt, const int &head_size, const int &sequence_len, const int &kv_dim, const int &current_step, const int &dimension, const int &hidden_dimension, const int &kv_mul, const int &num_heads);

    HNNStatus cleanupAndReturn(HNNStatus status, RunContext* ctx, Config* config, BufferManager* buffer_manager);

};