#pragma once

#define DEVICE __device__ __forceinline__
#define u32 uint32_t
#define u64 uint64_t
#define i32 int32_t

#ifdef USE_FP16
#define MAT_VAL_TYPE __half
#endif
#ifdef USE_BF16
#define MAT_VAL_TYPE __nv_bfloat16
#endif
#ifdef USE_TF32
#define MAT_VAL_TYPE float
#endif