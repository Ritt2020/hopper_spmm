/*
* @brief: WGMA 相关的 PTX 函数封装
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include "ptx_utils.h"

DEVICE u32 encode(u32 x){
    return (x & 0x3FFFF) >> 4;
}

// 创建 wgmma 描述符，没有 swizzle
DEVICE u64 create_wgmma_descriptor_no_swizzle(MAT_VAL_TYPE *ptr, u32 lbo, u32 sbo){
    uint64_t desc = 0;
    desc |= (u64)encode(lbo) << 16;
    desc |= (u64)encode(sbo) << 32;
    desc |= (u64)encode(__cvta_generic_to_shared(ptr));
    return desc;
}
// wgmma fence 函数
DEVICE void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
// wgmma commit group 函数
DEVICE void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;");
}
// wgmma wait group 函数
DEVICE void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned 0;");
}
#ifdef USE_TF32
/*
* @brief: wgmma tf32 m64n8k8 函数， A 和 B 在 shared memory， SS 模式
* @param: d_A: A 矩阵的指针， float 类型
* @param: d_B: B 矩阵的指针， float 类型
* @param: d_C: C 矩阵的指针
*/
DEVICE void wgmma_tf32_m64n8k8_no_trans_ss(float *d_A, float *d_B, float *d_C){
    uint32_t const* A   = reinterpret_cast<uint32_t const*>(d_A);
    uint32_t const* B   = reinterpret_cast<uint32_t const*>(d_B);
    // 转换 fp32 为 tf32
    convert_fp32_to_tf32_shared(A, 64 * 8);
    convert_fp32_to_tf32_shared(B, 8 * 8);
    // 创建 wgmma 描述符
    uint64_t desc_a = create_wgmma_descriptor_no_swizzle(d_A, 256 * sizeof(float), 32 * sizeof(float));
    uint64_t desc_b = create_wgmma_descriptor_no_swizzle(d_B, 32 * sizeof(float), 32 * sizeof(float));
    // 执行 wgmma 操作
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32"
                "{%0, %1, %2, %3},"
                "%4, %5, 1, 1, 1;\n"
                :"+f"(d_C[0]), "+f"(d_C[1]), "+f"(d_C[2]), "+f"(d_C[3])
                :"l"(desc_a), "l"(desc_b)
            );
}
#endif