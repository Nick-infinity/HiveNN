#pragma once
#include "model_structure/configs/configs.h"
#include "common/runcontext.h"
#include "types.h"
#include "utils.h"

HNNStatus parse_model(RunContext *&ctx, char *&model_path, char* &tokenizer, char **&vocab, float *&vocab_scores, unsigned int &max_token_length);