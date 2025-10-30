#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DEVICE __device__ __forceinline__
#define u32 uint32_t
#define u64 uint64_t
#define i32 int32_t

#define MAT_PTR_TYPE uint32_t
#define MAT_IDX_TYPE uint32_t

#ifdef USE_FP16
#define MAT_VAL_TYPE __half
#define MAT_MAP_TYPE uint64_t
#define ROW_WINDOW 8
#define COL_WINDOW 16
#endif
#ifdef USE_BF16
#define MAT_VAL_TYPE __nv_bfloat16
#define MAT_MAP_TYPE uint64_t
#define ROW_WINDOW 8
#define COL_WINDOW 16
#endif
#ifdef USE_TF32
#define MAT_VAL_TYPE float
    #ifdef MMA_M16N8K4
    #define MMA_M 16
    #define MMA_N 8
    #define MMA_K 4
    #define MAT_MAP_TYPE uint32_t
    #define ROW_WINDOW 8
    #define COL_WINDOW 4
    #else
    #define MMA_M 16
    #define MMA_N 8
    #define MMA_K 8
    #define MAT_MAP_TYPE uint64_t
    #define ROW_WINDOW 8
    #define COL_WINDOW 8
    #endif
#endif

#define vint uint32_t
#define ATOMIC_TYPE uint8_t
#define IDX_MAX UINT32_MAX
#define WARMUP_RUNS 3
#define RUNS 5

struct PERF_RESULT {
    float time;
    int runs;
    void print() {
        printf("SPMM 总耗时: %.3f ms\n", time);
        printf("SPMM 总运行次数: %d\n", runs);
        printf("SPMM 平均耗时: %.3f ms\n", time / static_cast<float>(runs));
    }
};