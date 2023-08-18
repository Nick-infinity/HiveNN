/**
 * This file contains common utilities used by ops/kernels for CPU
 */

#pragma once

#include <cstdint>
#include "utils.h"
#include <cmath>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define NEON_OPT
#include <arm_neon.h>
#endif
