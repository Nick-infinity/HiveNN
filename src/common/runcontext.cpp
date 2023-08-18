#include "runcontext.h"

void RunContext::setBufferManager(BufferManager *&buffer_manager)
{
    this->buffer_manager = buffer_manager;
}

void RunContext::setConfig(Config *&config)
{
    this->config = config;
}

 void RunContext::setModelFile(char *&model_file)
{
    this->model_file = model_file;
}

Config *RunContext::getConfig()
{
    return this->config;
}

BufferManager *RunContext::getBufferManager()
{
    return this->buffer_manager;
}

char *RunContext::getModelFile()
{
    return this->model_file;
}