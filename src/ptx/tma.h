/*
* @brief: TMA 相关的 PTX 函数封装
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include "ptx_utils.h"

/*
* @brief: 1D TMA 异步拷贝函数
* @param: smem_dst: 目标共享内存指针
* @param: src: 源数据指针
* @param: size: 拷贝大小
* @param: mbarrier: mbarrier 变量
*/
DEVICE void tma_cp_async_bulk_1d(
    MAT_VAL_TYPE* smem_dst,
    MAT_VAL_TYPE* src,
    u32 size,
    u64 &mbarrier
){
    u32 mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
    u32 smem_addr = cast_smem_ptr_to_uint(smem_dst);
    u32 src_addr = cast_smem_ptr_to_uint(src);
    // 使用 cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr),
           "r"(src_addr),
           "r"(size),
           "r"(mbar_addr)
        : "memory"
    );
}

/*
* @brief: 2D TMA 异步拷贝函数
* @param: smem_dst: 目标共享内存指针
* @param: src_tma_desc: TMA 描述符
* @param: tile_coord_i: 目标 tile 的 i 坐标
* @param: tile_coord_j: 目标 tile 的 j 坐标
* @param: mbarrier: mbarrier 变量
*/
DEVICE void tma_cp_async_bulk_2d_shared_global_tile_mbarrier_bytes(
    MAT_VAL_TYPE* smem_dst,
    void const* const src_tma_desc,
    i32 tile_coord_i,
    i32 tile_coord_j,
    u64 &mbarrier)
{
    u64 tma_ptr = reinterpret_cast<u64>(src_tma_desc);
    u32 smem_addr = cast_smem_ptr_to_uint(smem_dst);
    u32 mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr),
           "l"(tma_ptr),
           "r"(tile_coord_i),
           "r"(tile_coord_j),
           "r"(mbar_addr)
        : "memory"
    );
}

/*
* @brief: 3D TMA 异步拷贝函数
* @param: smem_dst: 目标共享内存指针
* @param: src_tma_desc: TMA 描述符
* @param: tile_coord_i: 目标 tile 的 i 坐标
* @param: tile_coord_j: 目标 tile 的 j 坐标
* @param: tile_coord_k: 目标 tile 的 k 坐标
* @param: mbarrier: mbarrier 变量
*/
DEVICE void tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(
    MAT_VAL_TYPE* smem_dst,
    void const* const src_tma_desc,
    i32 tile_coord_i,
    i32 tile_coord_j,
    i32 tile_coord_k,
    u64 &mbarrier)
{
    u64 tma_ptr = reinterpret_cast<u64>(src_tma_desc);
    u32 smem_addr = cast_smem_ptr_to_uint(smem_dst);
    u32 mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(smem_addr),
           "l"(tma_ptr),
           "r"(tile_coord_i),
           "r"(tile_coord_j),
           "r"(tile_coord_k),
           "r"(mbar_addr)
        : "memory"
    );
}

/*
* @brief: 4D TMA 异步拷贝函数
* @param: smem_dst: 目标共享内存指针
* @param: src_tma_desc: TMA 描述符
* @param: tile_coord_i: 目标 tile 的 i 坐标
* @param: tile_coord_j: 目标 tile 的 j 坐标
* @param: tile_coord_k: 目标 tile 的 k 坐标
* @param: tile_coord_l: 目标 tile 的 l 坐标
* @param: mbarrier: mbarrier 变量
*/
DEVICE void tma_cp_async_bulk_4d_shared_global_tile_mbarrier_bytes(
    MAT_VAL_TYPE* smem_dst,
    void const* const src_tma_desc,
    i32 tile_coord_i,
    i32 tile_coord_j,
    i32 tile_coord_k,
    i32 tile_coord_l,
    u64 &mbarrier)
{
    u64 tma_ptr = reinterpret_cast<u64>(src_tma_desc);
    u32 smem_addr = cast_smem_ptr_to_uint(smem_dst);
    u32 mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(smem_addr),
           "l"(tma_ptr),
           "r"(tile_coord_i),
           "r"(tile_coord_j),
           "r"(tile_coord_k),
           "r"(tile_coord_l),
           "r"(mbar_addr)
        : "memory"
    );
}