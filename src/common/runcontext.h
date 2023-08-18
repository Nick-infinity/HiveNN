#pragma once
#include "memory_manager/buffer_manager.h"
#include "model_structure/configs/configs.h"

class RunContext
{
public:
    RunContext() = default;
    RunContext(BufferManager *buffer_manager, Config *config, char *model_file)
    {
        this->buffer_manager = buffer_manager;
        this->config = config;
    }
    void setBufferManager(BufferManager *&buffer_manager);
    void setConfig(Config *&config);
    void setModelFile(char *&model_file);
    Config *getConfig();
    BufferManager *getBufferManager();
    char *getModelFile();

    // they are frequently accessed , its better to not wrap them in accessors for speed
    int *prompt_tokens = nullptr;
    int num_prompt_tokens = 0;
    char **vocab;

private:
    Config *config = nullptr;
    BufferManager *buffer_manager = nullptr;
    char *model_file = nullptr;
};