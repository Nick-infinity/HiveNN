#include "op_utils.h"
#include "dequant_token.h"

void dequantize_token(float *x, const char *wptr, int token, int dim)
{
    qu8_assym *q8 = (qu8_assym *)wptr;
    float base = q8->min;
    float scale = q8->scale;
    uint8_t *qweight = q8->quantized_vals;
    int row = token * dim;

    // Unroll the loop to process multiple elements at a time
    int i = 0;
    for (; i <= dim - 4; i += 4)
    {
        x[i] = base + scale * qweight[row + i];
        x[i + 1] = base + scale * qweight[row + i + 1];
        x[i + 2] = base + scale * qweight[row + i + 2];
        x[i + 3] = base + scale * qweight[row + i + 3];
    }

    // Handle the remaining elements
    for (; i < dim; i++)
    {
        x[i] = base + scale * qweight[row + i];
    }
}

/**
 * NEON Kernel for dequantizing tokens
 * TODO: Keeping it separate as it needs further verification
 */
void dequantize_token_neon(float *x, const char *wptr, int token, int dim)
{
    qu8_assym *q8 = (qu8_assym *)wptr;
    float base = q8->min;
    float scale = q8->scale;
    uint8_t *qweight = q8->quantized_vals;
    int row = token * dim;
#ifdef NEON_OPT_

    int dim_16 = dim / 16 * 16; // Process 16 elements at a time

    for (int i = 0; i < dim_16; i += 16)
    {
        uint8x16_t qweights = vld1q_u8(qweight + row + i);
        float32x4_t weights_low = vaddq_f32(vdupq_n_f32(base), vmulq_f32(vdupq_n_f32(scale), vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(qweights))))));
        float32x4_t weights_high = vaddq_f32(vdupq_n_f32(base), vmulq_f32(vdupq_n_f32(scale), vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(qweights))))));

        float32x4_t result_low = vld1q_f32(x + i);
        float32x4_t result_high = vld1q_f32(x + i + 4);

        result_low = vaddq_f32(result_low, weights_low);
        result_high = vaddq_f32(result_high, weights_high);

        vst1q_f32(x + i, result_low);
        vst1q_f32(x + i + 4, result_high);
    }

    for (int i = dim_16; i < dim; i++)
    {
        x[i] = base + scale * qweight[row + i];
    }

#else
    for (int i = 0; i < dim; i++)
    {
        x[i] = base + scale * qweight[row + i];
    }
#endif
}
