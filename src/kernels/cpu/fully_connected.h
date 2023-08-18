#pragma once
#include "op_utils.h"

void fully_connected_f32(float* xout, const float* x, const float* w, int n, int d);
void fully_connected_u8(float* xout, const float* x, const char* wptr, int l, int n, int d);