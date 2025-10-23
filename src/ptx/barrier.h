/*
* @brief: mbarrier 相关的 PTX 函数封装
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include "ptx_utils.h"

/*
* @brief: mbarrier 初始化函数
* @param: barrier: mbarrier 变量
* @param: count: mbarrier 期望的 arrive 次数
*/
DEVICE void mbarrier_init(u64 &barrier, i32 count) {
    auto smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                "mbarrier.init.shared::cta.b64 [%0], %1;\n"
                "\t}" ::"r"(smem_addr),
                "r"(count));
}

/*
* @brief: mbarrier wait 函数
* @param: barrier: mbarrier 变量
* @param: phase: mbarrier 期望的 phase
*/
DEVICE void mbarrier_wait(u64& barrier, i32 phase) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_addr),
        "r"(phase));
  }

/*
* @brief: mbarrier arrive 函数
* @param: barrier: mbarrier 变量
*/
DEVICE void mbarrier_arrive(u64 &barrier) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                    "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                    "\t}"
                    :: "r"(smem_addr));
}

/*
* @brief: mbarrier arrive 函数 有 count
* @param: barrier: mbarrier 变量
* @param: count: mbarrier 需要 arrive 的次数
*/
DEVICE void mbarrier_arrive(u64 &barrier, i32 count) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" 
        ::"r"(smem_addr), "r"(count)
        : "memory");
}

/*
* @brief: mbarrier arrive and wait 函数
* @param: barrier: mbarrier 变量
*/
DEVICE void mbarrier_arrive_and_wait(u64 &barrier) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                ".reg .b64 phase;\n\t"
                ".reg .pred p;\n\t"
                "mbarrier.arrive.shared::cta.b64 phase, [%0];\n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.shared.b64 p, [%0], phase; \n\t"
                "@p bra.uni DONE; \n\t"
                "bra.uni     LAB_WAIT; \n\t"
                "DONE: \n\t"
                "}"
                :
                : "r"(smem_addr));
}

/*
* @brief: mbarrier expect tx 函数
* @param: barrier: mbarrier 变量
* @param: bytes: mbarrier 期望的传输字节数
*/
DEVICE void mbarrier_expect_tx(u64 &barrier, u32 bytes) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" 
        ::"r"(smem_addr),
        "r"(bytes)
    );
}