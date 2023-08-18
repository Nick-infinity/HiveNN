#include "argmax.h"

/**
 * We ue argmax for finding best probability. Optimize this using neon as its called for every generated token
 */
int argmax(const float *input_data, int size)
{
    int32_t max_index = 0;
    float max_value = input_data[0];
    int32_t i = 1;
#ifdef NEON_OPT
    if (size >= 4)
    {
        float32x4_t max_value_f32x4 = vld1q_f32(input_data);
        const int32_t index_init[4] = {0, 1, 2, 3};
        int32x4_t max_index_s32x4 = vld1q_s32(index_init);
        int32x4_t index_s32x4 = max_index_s32x4;
        int32x4_t inc = vdupq_n_s32(4);
        for (i = 4; i <= size - 4; i += 4)
        {
            // Increase indices by 4.
            index_s32x4 = vaddq_s32(index_s32x4, inc);
            float32x4_t v = vld1q_f32(&input_data[i]);
            uint32x4_t mask = vcgtq_f32(v, max_value_f32x4);
            max_value_f32x4 = vmaxq_f32(max_value_f32x4, v);
            max_index_s32x4 = vbslq_s32(mask, index_s32x4, max_index_s32x4);
        }
        // Find max element within float32x4_t.
#ifdef __aarch64__
        max_value = vmaxvq_f32(max_value_f32x4);
#else
        float32x2_t max_value_f32x2 = vpmax_f32(vget_low_f32(max_value_f32x4),
                                                vget_high_f32(max_value_f32x4));
        max_value_f32x2 = vpmax_f32(max_value_f32x2, max_value_f32x2);
        max_value = vget_lane_f32(max_value_f32x2, 0);
#endif // __aarch64__
       // Mask indices of non-max values with max int32_t.
        float32x4_t fill_max_value_f32x4 = vdupq_n_f32(max_value);
        uint32x4_t mask = vceqq_f32(max_value_f32x4, fill_max_value_f32x4);
        int32x4_t all_set = vdupq_n_s32(std::numeric_limits<int>::max());
        max_index_s32x4 = vbslq_s32(mask, max_index_s32x4, all_set);
        // Find min index of max values.
#ifdef __aarch64__
        max_index = vminvq_s32(max_index_s32x4);
#else
        int32x2_t max_index_s32x2 = vpmin_s32(vget_low_s32(max_index_s32x4),
                                              vget_high_s32(max_index_s32x4));
        max_index_s32x2 = vpmin_s32(max_index_s32x2, max_index_s32x2);
        max_index = vget_lane_s32(max_index_s32x2, 0);
#endif // __aarch64__
    }
#endif // USE_NEON
    // Leftover loop.
    for (; i < size; ++i)
    {
        const float curr_value = input_data[i];
        if (curr_value > max_value)
        {
            max_value = curr_value;
            max_index = i;
        }
    }
    return max_index;
}