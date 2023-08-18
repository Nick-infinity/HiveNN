#include "fully_connected.h"

void fully_connected_f32(float *xout, const float *x, const float *w, int n, int d)
{
    // Parallalize if platform supports omp
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void fully_connected_u8(float *xout, const float *x, const char *wptr, int l, int n, int d)
{
    qu8_assym *q8 = (qu8_assym *)(wptr + getOffset(DTYPE::QU8_ASSYM, l, n * d));
    float base = q8->min;
    float scale = q8->scale;
    uint8_t *qweight = q8->quantized_vals;

#ifdef NEON_OPT_

    for (int i = 0; i < d; i++)
    {
        float32x4_t val = vdupq_n_f32(0.0f);

        for (int j = 0; j <= n - 16; j += 16)
        {
            uint8x16_t qweights = vld1q_u8(qweight + i * n + j);

            uint16x8_t qweights_low = vmovl_u8(vget_low_u8(qweights));
            uint16x8_t qweights_high = vmovl_u8(vget_high_u8(qweights));

            float32x4_t weights_low = vaddq_f32(vdupq_n_f32(base), vmulq_f32(vdupq_n_f32(scale), vcvtq_f32_u32(vmovl_u16(vget_low_u16(qweights_low)))));
            float32x4_t weights_high = vaddq_f32(vdupq_n_f32(base), vmulq_f32(vdupq_n_f32(scale), vcvtq_f32_u32(vmovl_u16(vget_high_u16(qweights_low)))));

            float32x4_t input = vld1q_f32(x + j);
            val = vmlaq_f32(val, weights_low, input);

            float32x4_t input_high = vld1q_f32(x + j + 4);
            val = vmlaq_f32(val, weights_high, input_high);
        }

        // Accumulate the results from the unrolled loop
        val = vpaddq_f32(val, val);
        val = vpaddq_f32(val, val);

        // Handle remaining iterations
        float32x2_t val_low = vget_low_f32(val);
        float32x2_t val_high = vget_high_f32(val);
        val_low = vadd_f32(val_low, val_high);
        val_low = vpadd_f32(val_low, val_low);

        float val_scalar = vget_lane_f32(val_low, 0);
        for (int j = (n / 16) * 16; j < n; j++)
        {
            float weight = base + scale * qweight[i * n + j];
            val_scalar += weight * x[j];
        }

        xout[i] = val_scalar;
    }
#else
    // Loop unrolling factor
    const int unroll_factor = 4;

// Use openmp if available
#pragma omp parallel for
    for (int i = 0; i < d; i++) {
        int j;
        float val[unroll_factor] = {0.0f};

        // Unrolled loop
        for (j = 0; j <= n - unroll_factor; j += unroll_factor) {
            for (int k = 0; k < unroll_factor; k++) {
                int idx = i * n + j + k;
                float weight = base + scale * qweight[idx];
                val[k] += weight * x[j + k];
            }
        }

        // Handle remaining iterations
        for (; j < n; j++) {
            int idx = i * n + j;
            float weight = base + scale * qweight[idx];
            val[0] += weight * x[j];
        }

        // Accumulate the results from the unrolled loop
        for (int k = 1; k < unroll_factor; k++) {
            val[0] += val[k];
        }

        xout[i] = val[0];
    }
#endif
}