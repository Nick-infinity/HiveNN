#include "buffer_manager.h"

HNNStatus BufferManager::initialize_tensors(char *weights, Config *mc)
{

    // TODO:: Make Tensor parsing automatic when model has more information
    // For Now we parse the weight buffers and shared buffers manually
    if (nullptr == weights || mc == nullptr)
    {
        ELOG("Weights or Configs are NULL %d\n", HNNStatus::READ_ERROR);
        // TODO:: return from here in production build
    }
    DLOG("Data pointer in initialize_tensors %p\n", static_cast<void *>(weights));
    if (mc->config_type == CONFIG_TYPE::LLAMA2)
    {

        TransformerConfig *tc = static_cast<TransformerConfig *>(mc);
        size_t kv_dimension = (tc->getDimension() * tc->getNumKVHeads()) / tc->getNumHeads();

        // Create tensor & Allocate memory for shared tensors
        // TODO:: Run this in a loop once we get the sizes and tensor mapping from model directly
        shared_tensors[st(RUNTIME::ACT)] = new Tensor(DTYPE::FLOAT32, tc->getDimension());
        shared_tensors[st(RUNTIME::ACT1)] = new Tensor(DTYPE::FLOAT32, tc->getDimension());
        shared_tensors[st(RUNTIME::ACT2)] = new Tensor(DTYPE::FLOAT32, tc->getDimension());
        shared_tensors[st(RUNTIME::HID)] = new Tensor(DTYPE::FLOAT32, tc->getHiddenDimension());
        shared_tensors[st(RUNTIME::HID2)] = new Tensor(DTYPE::FLOAT32, tc->getHiddenDimension());
        shared_tensors[st(RUNTIME::QUERY)] = new Tensor(DTYPE::FLOAT32, tc->getDimension());
        shared_tensors[st(RUNTIME::KEY)] = new Tensor(DTYPE::FLOAT32, kv_dimension);
        shared_tensors[st(RUNTIME::VALUE)] = new Tensor(DTYPE::FLOAT32, kv_dimension);
        shared_tensors[st(RUNTIME::ATTN)] = new Tensor(DTYPE::FLOAT32, tc->getNumHeads() * tc->getSequenceLength());
        shared_tensors[st(RUNTIME::LOGITS)] = new Tensor(DTYPE::FLOAT32, tc->getVocabSize());
        shared_tensors[st(RUNTIME::PINDEX)] = new Tensor(DTYPE::CUSTOM, tc->getVocabSize(), sizeof(ProbIndex));
        shared_tensors[st(RUNTIME::KEY_CACHE)] = new Tensor(DTYPE::FLOAT32, tc->getNumLayers() * tc->getSequenceLength() * kv_dimension);
        shared_tensors[st(RUNTIME::VALUE_CACHE)] = new Tensor(DTYPE::FLOAT32, tc->getNumLayers() * tc->getSequenceLength() * kv_dimension);

        DLOG("Shared Tensors creation is done successfully\n");

        if (weights != nullptr)
        {
            // Set the weight tensor buffers, these are preallocated using mmap and model
            // TODO:: Parse these in loop when we get the offset for each layer directly from the model
            int head_size = tc->getDimension() / tc->getNumHeads();
            weight_tensors[st(WEIGHTS::WTOKEN_EMBEDDING)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getVocabSize() * tc->getDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WTOKEN_EMBEDDING)])->getDType(), 1, tc->getVocabSize() * tc->getDimension());

            weight_tensors[st(WEIGHTS::WRMS_ATTN)] = new Tensor(DTYPE::FLOAT32, weights, tc->getNumLayers() * tc->getDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WRMS_ATTN)])->getDType(), tc->getNumLayers(), tc->getDimension());

            weight_tensors[st(WEIGHTS::WQUERY)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getDimension() * (tc->getNumHeads() * head_size));
            weights += getOffset((weight_tensors[st(WEIGHTS::WQUERY)])->getDType(), tc->getNumLayers(), tc->getDimension() * (tc->getNumHeads() * head_size));

            weight_tensors[st(WEIGHTS::WKEY)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getDimension() * (tc->getNumKVHeads() * head_size));
            weights += getOffset((weight_tensors[st(WEIGHTS::WKEY)])->getDType(), tc->getNumLayers(), tc->getDimension() * (tc->getNumKVHeads() * head_size));

            weight_tensors[st(WEIGHTS::WVAL)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getDimension() * (tc->getNumKVHeads() * head_size));
            weights += getOffset((weight_tensors[st(WEIGHTS::WVAL)])->getDType(), tc->getNumLayers(), tc->getDimension() * (tc->getNumKVHeads() * head_size));

            weight_tensors[st(WEIGHTS::WOUT)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getDimension() * (tc->getNumHeads() * head_size));
            weights += getOffset((weight_tensors[st(WEIGHTS::WOUT)])->getDType(), tc->getNumLayers(), tc->getDimension() * (tc->getNumHeads() * head_size));

            weight_tensors[st(WEIGHTS::WRMS_FF)] = new Tensor(DTYPE::FLOAT32, weights, tc->getNumLayers() * tc->getDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WRMS_FF)])->getDType(), tc->getNumLayers(), tc->getDimension());

            weight_tensors[st(WEIGHTS::WHID1)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getDimension() * tc->getHiddenDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WHID1)])->getDType(), tc->getNumLayers(), tc->getDimension() * tc->getHiddenDimension());

            weight_tensors[st(WEIGHTS::WHID2)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getHiddenDimension() * tc->getDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WHID2)])->getDType(), tc->getNumLayers(), tc->getHiddenDimension() * tc->getDimension());

            weight_tensors[st(WEIGHTS::WHID3)] = new Tensor(DTYPE::QU8_ASSYM, weights, tc->getNumLayers() * tc->getDimension() * tc->getHiddenDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WHID3)])->getDType(), tc->getNumLayers(), tc->getDimension() * tc->getHiddenDimension());

            weight_tensors[st(WEIGHTS::WRMS_FIN)] = new Tensor(DTYPE::FLOAT32, weights, tc->getDimension());
            weights += getOffset((weight_tensors[st(WEIGHTS::WRMS_FIN)])->getDType(), 1, tc->getDimension());

            weight_tensors[st(WEIGHTS::WFREQ_REAL)] = new Tensor(DTYPE::FLOAT32, weights, tc->getSequenceLength() * head_size / 2);
            weights += getOffset((weight_tensors[st(WEIGHTS::WFREQ_REAL)])->getDType(), 1, tc->getSequenceLength() * head_size / 2);

            weight_tensors[st(WEIGHTS::WFREQ_IMG)] = new Tensor(DTYPE::FLOAT32, weights, tc->getSequenceLength() * head_size / 2);
            weights += getOffset((weight_tensors[st(WEIGHTS::WFREQ_IMG)])->getDType(), 1, tc->getSequenceLength() * head_size / 2);
            // Skip Quantizing last decoder
            weight_tensors[st(WEIGHTS::WOUTFINAL)] = new Tensor(DTYPE::FLOAT32, weights, tc->getDimension() * tc->getVocabSize());
            weights += getOffset((weight_tensors[st(WEIGHTS::WOUTFINAL)])->getDType(), 1, tc->getDimension() * tc->getVocabSize());

            DLOG("Weight Tensors creation is done successfully\n");
        }
    }
    return HNNStatus::OK;
}