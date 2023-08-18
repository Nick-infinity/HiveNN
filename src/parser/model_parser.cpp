#include "model_parser.h"
#include <fcntl.h>    // FILE
#include <sys/mman.h> // MMAP

HNNStatus parse_model(RunContext *&ctx, char *&model_path, char *&tokenizer, char **&vocab, float *&vocab_scores, unsigned int &max_token_length)
{
    // Check each parameter individually to track errors better
    if (ctx == nullptr)
    {
        ELOG("Context is nullptr\n");
        return HNNStatus::FAIL;
    }

    if (ctx->getConfig() == nullptr)
    {
        ELOG("Config is nullptr\n");
        return HNNStatus::FAIL;
    }

    if (ctx->getConfig()->config_type != CONFIG_TYPE::LLAMA2)
    {
        ELOG("Wrong model config %d \n", (int)ctx->getConfig()->config_type);
        return HNNStatus::FAIL;
    }

    if (model_path == nullptr)
    {
        ELOG("Model path is nullptr\n");
        return HNNStatus::FAIL;
    }

    TransformerConfig *config = static_cast<TransformerConfig *>(ctx->getConfig());

    // Parse the model file
    FILE *file = fopen(model_path, "rb");
    if (!file)
    {
        ELOG("Couldn't open file %s\n", model_path);
        return HNNStatus::FAIL;
    }

    // Temporary config saver
    Mconfig mconfig;

    if (fread(&mconfig, sizeof(Mconfig), 1, file) != 1)
    {
        ELOG("Error reading config params form model file %s\n", model_path);
        return HNNStatus::READ_ERROR;
    }

    // TODO:: Fix the model vocab size once we fix in extraction script
    if (mconfig.vocab_size < 0)
    {
        mconfig.vocab_size = -mconfig.vocab_size;
    }

    // Set the configuration in context
    config->setConfig(mconfig.dimension, mconfig.hidden_dimension, mconfig.num_layers, mconfig.num_heads, mconfig.num_kv_heads, mconfig.vocab_size, mconfig.sequence_len);

    fseek(file, 0, SEEK_END);        // move file pointer to end of file
    ssize_t file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    DLOG("File size for model %lu\n", file_size);
    // memory map the Transformer weights into the data pointer
    int fd = open(model_path, O_RDONLY); // open in read only mode
    if (fd == -1)
    {
        ELOG("Model open failed! %s\n", model_path);
        return HNNStatus::READ_ERROR;
    }

    // map the model weight files
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    DLOG("Address of MMAPed model %p \n", static_cast<void *>(data));
    if (data == MAP_FAILED)
    {
        ELOG("Model mmap failed! %s\n", model_path);
        return HNNStatus::WRITE_ERROR;
    }

    // Move the file ahead by config size
    // Model packing [Config params][Weights for each type of layer]
    char *weight_ptr = static_cast<char *>(data);
    weight_ptr = weight_ptr + sizeof(Mconfig);

    DLOG("Weight Pointer %p \n", static_cast<void *>(weight_ptr));

    // Populate the context with model weight pointer
    ctx->setModelFile(weight_ptr);

    // Read the tokenizer
    int vocab_size = config->getVocabSize();
    vocab = static_cast<char **>(malloc(vocab_size * sizeof(char *)));
    vocab_scores = static_cast<float *>(malloc(vocab_size * sizeof(float)));

    file = fopen(tokenizer, "rb");

    if (!file)
    {
        ELOG("Couldn't load Tokenizer %s\n", tokenizer);
        return HNNStatus::READ_ERROR;
    }
    if (fread(&max_token_length, sizeof(int), 1, file) != 1)
    {
        ELOG("Failed to read max lenght %d\n", max_token_length);
        return HNNStatus::READ_ERROR;
    }

    int len;
    for (int i = 0; i < vocab_size; i++)
    {
        if (fread(vocab_scores + i, sizeof(float), 1, file) != 1)
        {
            ELOG("Failed to read vocab_scores\n");
            return HNNStatus::READ_ERROR;
        }
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            ELOG("Failed to read len\n");
            return HNNStatus::READ_ERROR;
        }
        vocab[i] = (char *)malloc(len + 1);
        if (fread(vocab[i], len, 1, file) != 1)
        {
            ELOG("Failed to vocab\n");
            return HNNStatus::READ_ERROR;
        }
        vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
    return HNNStatus::OK;
}