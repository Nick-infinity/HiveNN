#pragma once
#include "op_utils.h"

void rms_normalization_f32(float* o, const float* x, const float* weight, int size);
void rms_normalization_qu8(float* o, const float* x, const char *wptr, int l, int size);
