#pragma once
#include "op_utils.h"

void dequantize_token(float* x, const char* wptr, int token, int dim);
void dequantize_token_neon(float* x, const char* wptr, int token, int dim);