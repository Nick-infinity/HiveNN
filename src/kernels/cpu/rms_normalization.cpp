#include "rms_normalization.h"


/**
 * Currently code uses f32 norms. optimize neon code for f32 as of now
*/
void rms_normalization_f32(float *o, const float *x, const float *weight, int size)
{

#ifdef NEON_OPT
    float32x4_t sum4 = vdupq_n_f32(0.0f);

    int i;
    for (i = 0; i <= size - 4; i += 4)
    {
        float32x4_t x4 = vld1q_f32(x + i);
        float32x4_t x_squared = vmulq_f32(x4, x4);
        sum4 = vaddq_f32(sum4, x_squared);
    }

    // Accumulate sum of squares from remaining elements
    float sum = 0.0f;
    for (; i < size; i++)
    {
        sum += x[i] * x[i];
    }
    sum += vgetq_lane_f32(sum4, 0) + vgetq_lane_f32(sum4, 1) + vgetq_lane_f32(sum4, 2) + vgetq_lane_f32(sum4, 3);
    float ss = sum / size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    // Normalize and scale using NEON intrinsics
    float32x4_t ss4 = vdupq_n_f32(ss);
    for (i = 0; i <= size - 4; i += 4)
    {
        float32x4_t x4 = vld1q_f32(x + i);
        float32x4_t weight4 = vld1q_f32(weight + i);
        float32x4_t result4 = vmulq_f32(ss4, vmulq_f32(weight4, x4));
        vst1q_f32(o + i, result4);
    }

    // Handle remaining elements
    for (; i < size; i++)
    {
        o[i] = weight[i] * (ss * x[i]);
    }
#else
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    // normalize and scale & unroll loop
    int j = 0;
    for (; j <= size - 4; j += 4)
    {
        float x1 = x[j];
        float x2 = x[j + 1];
        float x3 = x[j + 2];
        float x4 = x[j + 3];

        float w1 = weight[j];
        float w2 = weight[j + 1];
        float w3 = weight[j + 2];
        float w4 = weight[j + 3];

        o[j] = w1 * (ss * x1);
        o[j + 1] = w2 * (ss * x2);
        o[j + 2] = w3 * (ss * x3);
        o[j + 3] = w4 * (ss * x4);
    }

    // Handle the remaining elements
    for (; j < size; j++)
    {
        o[j] = weight[j] * (ss * x[j]);
    }
#endif
}

void rms_normalization_qu8(float *o, const float *x, const char *wptr, int l, int size)
{
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    // normalize and scale
    qu8_assym *q8 = (qu8_assym *)(wptr + getOffset(DTYPE::QU8_ASSYM, l, size));
    float base = q8->min;
    float scale = q8->scale;
    uint8_t *qweight = q8->quantized_vals;

    // Calculate the common factor for scaling outside the loop
    float factor = base * ss;
    // unroll the loop
    int j = 0;
    for (; j <= size - 4; j += 4)
    {
        float x1 = x[j];
        float x2 = x[j + 1];
        float x3 = x[j + 2];
        float x4 = x[j + 3];

        float weight1 = base + scale * qweight[j];
        float weight2 = base + scale * qweight[j + 1];
        float weight3 = base + scale * qweight[j + 2];
        float weight4 = base + scale * qweight[j + 3];

        o[j] = factor * weight1 * x1;
        o[j + 1] = factor * weight2 * x2;
        o[j + 2] = factor * weight3 * x3;
        o[j + 3] = factor * weight4 * x4;
    }

    // Handle the remaining elements
    for (; j < size; j++)
    {
        float weight = base + scale * qweight[j];
        o[j] = weight * (ss * x[j]);
    }
}
