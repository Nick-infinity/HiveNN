#include "llama_2.h"
#include "kernels/cpu/operators.h"
#include "model_structure/configs/transformers_config.h"
#include "types.h"
#include <cstring>
#include <cmath>
#include "src/common/helper.h"
#include "src/parser/model_parser.h"
#include <cmath>
#include <algorithm>

HNNStatus Llama2Model::open(char *&model_path, char *&tokenizer_path, char *&prompt)
{

    ctx = new RunContext();
    if (!ctx)
    {
        ELOG("Failed to create RunContext\n");
        return HNNStatus::FAIL;
    }

    BufferManager *buffer_manager = new BufferManager();
    if (!buffer_manager)
    {
        ELOG("Failed to create BufferManager\n");
        delete ctx;
        return HNNStatus::FAIL;
    }

    Config *config = new TransformerConfig();
    if (!config)
    {
        ELOG("Failed to create TransformerConfig\n");
        delete buffer_manager;
        delete ctx;
        return HNNStatus::FAIL;
    }

    ctx->setConfig(config);
    ctx->setBufferManager(buffer_manager);

    float *vocab_scores = nullptr;
    unsigned int max_token_length = 0;

    if (HNNStatus::OK != parse_model(ctx, model_path, tokenizer_path, ctx->vocab, vocab_scores, max_token_length))
    {
        ELOG("Failed to parse model\n");
        return cleanupAndReturn(HNNStatus::FAIL, ctx, config, buffer_manager);
    }

    if (!ctx->getBufferManager() || !ctx->getConfig() || !ctx->getModelFile())
    {
        ELOG("RunContext is not initialized properly\n");
        return cleanupAndReturn(HNNStatus::FAIL, ctx, config, buffer_manager);
    }

    ctx->getBufferManager()->initialize_tensors(ctx->getModelFile(), ctx->getConfig());

    TransformerConfig *tc = static_cast<TransformerConfig *>(config);

    if (vocab_scores == nullptr)
    {
        ELOG("Vocab Score is NULL\n");
    }

    if (prompt != nullptr)
    {
        ctx->prompt_tokens = (int *)malloc((strlen(prompt) + 1) * sizeof(int));
        if (ctx->prompt_tokens)
        {
            byte_pair_encoder(prompt, ctx->vocab, vocab_scores, tc->getVocabSize(), max_token_length, ctx->prompt_tokens, &(ctx->num_prompt_tokens));
        }
        else
        {
            ELOG("Failed to allocate memory for prompt tokens\n");
        }
    }
    else
    {
        DLOG("Null Input Prompt is provided\n");
    }
    return HNNStatus::OK;
}

HNNStatus Llama2Model::execute()
{
    // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int token = 1;
    int current_step = 0;
    int predcited_token;
    int *prompt_tokens = ctx->prompt_tokens;
    // We can limit the output to 256 steps. We can take this as input from user as well
    const int max_allowed_steps = 256;

    // Get the context data for this run
    auto sht = ctx->getBufferManager()->getSharedTensors();
    auto wt = ctx->getBufferManager()->getWeightTensors();
    int num_prompt_tokens = ctx->num_prompt_tokens;
    TransformerConfig *tc = static_cast<TransformerConfig *>(ctx->getConfig());

    while (current_step < max_allowed_steps)
    {
        // forward the transformer to get logits for the predcited_token token
        llama_2_execute(tc, current_step, sht, wt, token);

        if (current_step < num_prompt_tokens)
        {
            // if we are still processing the input prompt, force the predcited_token prompt token
            predcited_token = prompt_tokens[current_step];
        }
        else
        {
            // greedy argmax sampling: take the token with the highest probability
            predcited_token = argmax((sht[st(RUNTIME::LOGITS)])->getData<float>(), tc->getVocabSize());
        }
        current_step++;

        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (predcited_token == 1)
        {
            break;
        }

        // We can print the token here or get the decoded token using decode_token() api
        print_token(token, predcited_token, ctx);
        token = predcited_token;
    }
    printf("\n");

    return HNNStatus::OK;
}

HNNStatus Llama2Model::llama_2_execute(TransformerConfig *&config, int &current_step, Tensor **&sht, Tensor **&wt, int &token)
{
    const int dimension = config->getDimension();
    const int hidden_dimension = config->getHiddenDimension();
    const int num_layers = config->getNumLayers();
    const int num_heads = config->getNumHeads();
    const int num_kv_heads = config->getNumKVHeads();
    const int vocab_size = config->getVocabSize();
    const int sequence_len = config->getSequenceLength();

    // int dim = dimension;
    int kv_dim = (dimension * num_kv_heads) / num_heads;
    int kv_mul = num_heads / num_kv_heads; // integer multiplier of the kv sharing in multiquery

    int head_size = dimension / num_heads;

    // copy the token embedding into x
    dequantize_token((sht[st(RUNTIME::ACT)])->getData<float>(), (wt[st(WEIGHTS::WTOKEN_EMBEDDING)])->getData<char>(), token, dimension);

    // forward all the layers
    for (int l = 0; l < num_layers; l++)
    {
        multi_layer_process(l, sht, wt, head_size, sequence_len, kv_dim, current_step, dimension, hidden_dimension, kv_mul, num_heads);
    }

    // final rms norm
    rms_normalization_f32((sht[st(RUNTIME::ACT)])->getData<float>(), (sht[st(RUNTIME::ACT)])->getData<float>(), (wt[st(WEIGHTS::WRMS_FIN)])->getData<float>(), dimension);
    // classify into logits
    fully_connected_f32((sht[st(RUNTIME::LOGITS)])->getData<float>(), (sht[st(RUNTIME::ACT)])->getData<float>(), (wt[st(WEIGHTS::WOUTFINAL)])->getData<float>(), dimension, vocab_size);

    return HNNStatus::OK;
}

HNNStatus Llama2Model::close()
{
    // TODO:: Implement cleanups & mem cleanups for buffers
    /**
     * Free the mmmap , fd and allocated buffers for shared buffers
     */
    return HNNStatus::OK;
}

// head_size sequence_len kv_dim current_step dimension hidden_dimension
inline void Llama2Model::multi_layer_process(const int &l, Tensor **&sht, Tensor **&wt, const int &head_size, const int &sequence_len, const int &kv_dim, const int &current_step, const int &dimension, const int &hidden_dimension, const int &kv_mul, const int &num_heads)
{
    // attention rmsnorm
    rms_normalization_f32((sht[st(RUNTIME::ACT1)])->getData<float>(), (sht[st(RUNTIME::ACT)])->getData<float>(), (wt[st(WEIGHTS::WRMS_ATTN)])->getData<float>() + l * dimension, dimension);

    // qkv fully_connected_u8 for this current_stepition
    fully_connected_u8((sht[st(RUNTIME::QUERY)])->getData<float>(), (sht[st(RUNTIME::ACT1)])->getData<float>(), (wt[st(WEIGHTS::WQUERY)])->getData<char>(), l, dimension, dimension);
    fully_connected_u8((sht[st(RUNTIME::KEY)])->getData<float>(), (sht[st(RUNTIME::ACT1)])->getData<float>(), (wt[st(WEIGHTS::WKEY)])->getData<char>(), l, dimension, kv_dim);
    fully_connected_u8((sht[st(RUNTIME::VALUE)])->getData<float>(), (sht[st(RUNTIME::ACT1)])->getData<float>(), (wt[st(WEIGHTS::WVAL)])->getData<char>(), l, dimension, kv_dim);

    // RoPE relative current_stepitional encoding: complex-valued rotate q and k by freq_cis in each head
    rotationRoPE(dimension, head_size, current_step, (sht[st(RUNTIME::QUERY)])->getData<float>(), (sht[st(RUNTIME::KEY)])->getData<float>(), kv_dim);

    // save key,value at this time step (current_step) to our kv cache
    int layer_offset = l * sequence_len * kv_dim; // kv cache layer offset for convenience
    float *key_cache_row = (sht[st(RUNTIME::KEY_CACHE)])->getData<float>() + layer_offset + current_step * kv_dim;
    float *value_cache_row = (sht[st(RUNTIME::VALUE_CACHE)])->getData<float>() + layer_offset + current_step * kv_dim;
    memcpy(key_cache_row, (sht[st(RUNTIME::KEY)])->getData<float>(), kv_dim * sizeof(*key_cache_row));
    memcpy(value_cache_row, (sht[st(RUNTIME::VALUE)])->getData<float>(), kv_dim * sizeof(*value_cache_row));

    // multihead attention. iterate over all heads
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < num_heads; h++)
    {
        // get the query vector for this head
        float *q = (sht[st(RUNTIME::QUERY)])->getData<float>() + h * head_size;
        // attention scores for this head
        float *att = (sht[st(RUNTIME::ATTN)])->getData<float>() + h * sequence_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= current_step; t++)
        {
            // get the key vector for this head and at this timestep
            float *k = (sht[st(RUNTIME::KEY_CACHE)])->getData<float>() + layer_offset + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++)
            {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..current_step inclusively
        softmax_f32(att, current_step + 1);

        // weighted sum of the values, store them for fiether use
        float *xb = (sht[st(RUNTIME::ACT1)])->getData<float>() + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= current_step; t++)
        {
            // get the value vector for this head and at this timestep
            float *v = (sht[st(RUNTIME::VALUE_CACHE)])->getData<float>() + layer_offset + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++)
            {
                xb[i] += a * v[i];
            }
        }
    }

    // final fully connected to get the output of the attention
    fully_connected_u8((sht[st(RUNTIME::ACT2)])->getData<float>(), (sht[st(RUNTIME::ACT1)])->getData<float>(), (wt[st(WEIGHTS::WOUT)])->getData<char>(), l, dimension, dimension);

    // residual connection back into x
    for (int i = 0; i < dimension; i++)
    {
        (sht[st(RUNTIME::ACT)])->getData<float>()[i] += (sht[st(RUNTIME::ACT2)])->getData<float>()[i];
    }

    // ffn rmsnorm
    rms_normalization_f32((sht[st(RUNTIME::ACT1)])->getData<float>(), (sht[st(RUNTIME::ACT)])->getData<float>(), (wt[st(WEIGHTS::WRMS_FF)])->getData<float>() + l * dimension, dimension);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    fully_connected_u8((sht[st(RUNTIME::HID)])->getData<float>(), (sht[st(RUNTIME::ACT1)])->getData<float>(), (wt[st(WEIGHTS::WHID1)])->getData<char>(), l, dimension, hidden_dimension);
    fully_connected_u8((sht[st(RUNTIME::HID2)])->getData<float>(), (sht[st(RUNTIME::ACT1)])->getData<float>(), (wt[st(WEIGHTS::WHID3)])->getData<char>(), l, dimension, hidden_dimension);

    // silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for (int i = 0; i < hidden_dimension; i++)
    {
        (sht[st(RUNTIME::HID)])->getData<float>()[i] = (sht[st(RUNTIME::HID)])->getData<float>()[i] * (1.0f / (1.0f + expf(-((sht[st(RUNTIME::HID)])->getData<float>()[i]))));
    }

    // elementwise multiply with 3rd hidden dimension
    for (int i = 0; i < hidden_dimension; i++)
    {
        (sht[st(RUNTIME::HID)])->getData<float>()[i] = (sht[st(RUNTIME::HID)])->getData<float>()[i] * (sht[st(RUNTIME::HID2)])->getData<float>()[i];
    }

    // final fully conncted to get the output of the ffn
    fully_connected_u8((sht[st(RUNTIME::ACT1)])->getData<float>(), (sht[st(RUNTIME::HID)])->getData<float>(), (wt[st(WEIGHTS::WHID2)])->getData<char>(), l, hidden_dimension, dimension);

    for (int i = 0; i < dimension; i++)
    {
        (sht[st(RUNTIME::ACT)])->getData<float>()[i] += (sht[st(RUNTIME::ACT1)])->getData<float>()[i];
    }
}

inline void Llama2Model::rotationRoPE(int dimension, int head_size, int current_step, float *qData, float *kData, int kv_dim)
{
    for (int i = 0; i < dimension; i += 2)
    {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = current_step * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++)
        {
            float *vec = v == 0 ? qData : kData; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

HNNStatus Llama2Model::cleanupAndReturn(HNNStatus status, RunContext *ctx, Config *config, BufferManager *buffer_manager)
{
    delete ctx;
    delete config;
    delete buffer_manager;
    return status;
}