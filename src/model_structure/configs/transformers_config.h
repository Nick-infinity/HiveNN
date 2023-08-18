#pragma once
#include "configs.h"

class TransformerConfig : public Config
{
public:
    // Allow empty config creation
    TransformerConfig()
    {
        config_type = CONFIG_TYPE::LLAMA2;
        DLOG("Created Transormer Config\n");
    }
    TransformerConfig(int dim, int hiddenDim, int layers, int heads, int kvHeads, int vocab, int seqLen)
        : dimension(dim), hidden_dimension(hiddenDim), num_layers(layers),
          num_heads(heads), num_kv_heads(kvHeads), vocab_size(vocab), sequence_len(seqLen)
    {
        config_type = CONFIG_TYPE::LLAMA2;
        DLOG("Created Transormer Config\n");
        DLOG("Transformer Configuration\nDimension: % d\n HiddenDimentions: % d\n Number of Layers: % d\n Number of Heads: % d\n Number of KV Heads: % d\n Vocaburaly size: % d\n Sequence Length: % d\n ", dim, hiddenDim, layers, heads, kvHeads, vocab, seqLen);
    }

    // Getter functions
    int getDimension() const
    {
        return this->dimension;
    }

    int getHiddenDimension() const
    {
        return this->hidden_dimension;
    }

    int getNumLayers() const
    {
        return this->num_layers;
    }

    int getNumHeads() const
    {
        return this->num_heads;
    }

    int getNumKVHeads() const
    {
        return this->num_kv_heads;
    }

    int getVocabSize() const
    {
        return this->vocab_size;
    }

    int getSequenceLength() const
    {
        return this->sequence_len;
    }

    void setConfig(int dimension, int hidden_dimension, int num_layers, int num_heads, int num_kv_heads, int vocab_size, int sequence_len)
    {
        this->dimension = dimension;
        this->hidden_dimension = hidden_dimension;
        this->num_layers = num_layers;
        this->num_heads = num_heads;
        this->num_kv_heads = num_kv_heads;
        this->vocab_size = vocab_size;
        this->sequence_len = sequence_len;
        DLOG("Transformer Configuration\nDimension: % d\nHiddenDimentions: % d\nNumber of Layers: % d\nNumber of Heads: % d\nNumber of KV Heads: % d\nVocaburaly size: % d\nSequence Length: % d\n", dimension, hidden_dimension, num_layers, num_heads, num_kv_heads, vocab_size, sequence_len);
    }

private:
    int dimension = 0;        // transformer dimension
    int hidden_dimension = 0; // for ffn layers
    int num_layers = 0;       // number of layers
    int num_heads = 0;        // number of query heads
    int num_kv_heads = 0;     // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size = 0;       // vocabulary size, usually 256 (byte-level)
    int sequence_len = 0;     // max sequence length
};

enum class RUNTIME : size_t
{
    ACT = 0,
    ACT1,
    ACT2,
    HID,
    HID2,
    QUERY,
    KEY,
    VALUE,
    ATTN,
    LOGITS,
    PINDEX,
    KEY_CACHE,
    VALUE_CACHE,
    // Stores the last index and the lenght of this enum
    // Alwasy append before these enteries
    MAX = VALUE_CACHE,
    LENGTH,
};

enum class WEIGHTS : size_t
{
    WTOKEN_EMBEDDING = 0,
    WRMS_ATTN,
    WQUERY,
    WKEY,
    WVAL,
    WOUT,
    WRMS_FF,
    WHID1,
    WHID2,
    WHID3,
    WRMS_FIN,
    WFREQ_REAL,
    WFREQ_IMG,
    WOUTFINAL,
    // Stores the last index and the lenght of this enum
    // Alwasy append before these enteries
    MAX = WOUTFINAL,
    LENGTH
};

typedef struct
{
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling