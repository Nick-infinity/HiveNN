#include <chrono>

#include "src/memory_manager/buffer_allocator.h"
#include "src/memory_manager/buffer_manager.h"
#include "src/common/runcontext.h"
#include "src/model_structure/configs/transformers_config.h"
#include "src/model_structure/models/transformers/llama_2.h"
#include "src/types.h"
#include "src/utils.h"
#include "src/parser/model_parser.h"

#define MEASURE_TIME(func, ...)                                                                     \
    do                                                                                              \
    {                                                                                               \
        auto start = std::chrono::high_resolution_clock::now();                                     \
        func(__VA_ARGS__);                                                                          \
        auto end = std::chrono::high_resolution_clock::now();                                       \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
        PROFLOG("Time taken by %s: %lld microseconds.\n", #func, static_cast<long long>(duration)); \
    } while (false)

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        printf("Usage: %s <model> <tokenizer> <prompt>\n", argv[0]);
        return 1;
    }

    char *model_path = argv[1];
    char *tokenizer_file = argv[2];
    char *prompt = argv[3];

    // TODO:: Accept this param for user
    int max_tokens = 256;
    ILOG("Max tokens allowed are %d\n", max_tokens);

    if (model_path == NULL || tokenizer_file == NULL || prompt == NULL)
    {
        printf("Invalid arguements\n");
        return 1;
    }

    Model *model = new Llama2Model();
    MEASURE_TIME(model->open, model_path, tokenizer_file, prompt);
    MEASURE_TIME(model->execute);
    MEASURE_TIME(model->close);
}