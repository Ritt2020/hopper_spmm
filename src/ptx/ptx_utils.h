/*
* @brief: PTX 相关的工具函数
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"

/*
* @brief: 指针转换
* @param: ptr: 原始共享内存指针
* @return: 转换后的指针
*/
DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
    return static_cast<u32>(__cvta_generic_to_shared(ptr));
}

/*
* @brief: fence proxy async shared 函数，使共享内存的修改对异步 proxy 可见
*/
DEVICE void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

/*
* @brief: 将 fp32 转换为 tf32，shared memory
* @param: A: 原始 fp32 指针
* @param: num: 需要转换的元素个数
*/
DEVICE void convert_fp32_to_tf32_shared(uint32_t const *A, uint32_t num){
    for(int i = threadIdx.x; i < num; i += blockDim.x){
        asm volatile("cvt.rna.tf32.f32 %0, %0;\n"
            :: "r"(A[i]));
    }
}

/*
* @brief: 限制寄存器使用，增加寄存器数量
* @param: REGS: 需要增加的寄存器数量
*/
template <u32 REGS>
DEVICE void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

/*
* @brief: 限制寄存器使用，减少寄存器数量
* @param: REGS: 需要减少的寄存器数量
*/
template <u32 REGS>
DEVICE void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(REGS));
}