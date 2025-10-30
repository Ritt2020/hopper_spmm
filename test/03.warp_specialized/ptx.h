#pragma once
#define DEVICE __device__ __forceinline__

#if defined(USE_BF16)
    #define MAT_VAL_TYPE bf16
#else
    #define MAT_VAL_TYPE float
#endif

#define u32 uint32_t
#define u64 uint64_t
#define i32 int32_t

DEVICE u32 encode(u32 x){
    return (x & 0x3FFFF) >> 4;
}
// 指针转换
DEVICE u32 cast_smem_ptr_to_uint(void const *const ptr) {
    return static_cast<u32>(__cvta_generic_to_shared(ptr));
}
// mbarrier 初始化函数
DEVICE void mbarrier_init(u64 &barrier, i32 count) {
    auto smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                "mbarrier.init.shared::cta.b64 [%0], %1;\n"
                "\t}" ::"r"(smem_addr),
                "r"(count));
}
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
// mbarrier arrive 函数
DEVICE void mbarrier_arrive(u64 &barrier) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                    "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                    "\t}"
                    :: "r"(smem_addr));
}
// mbarrier arrive 函数 有 count
DEVICE void mbarrier_arrive(u64 &barrier, i32 count) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" 
        ::"r"(smem_addr), "r"(count)
        : "memory");
}

// mbarrier arrive and wait 函数
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
// 设置 mbarrier 期望的传输字节数
DEVICE void mbarrier_expect_tx(u64 &barrier, u32 bytes) {
    u32 smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" 
        ::"r"(smem_addr),
        "r"(bytes)
    );
}
// 1D TMA
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
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr),
           "r"(src_addr),
           "r"(size),
           "r"(mbar_addr)
        : "memory"
    );
}
// 2D TMA
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
// copy 3D TMA
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

// fence proxy async shared 函数
DEVICE void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
#if not defined(USE_BF16)
// convert fp32 to tf32 函数，shared memory
DEVICE void convert_fp32_to_tf32_shared(uint32_t const *A, uint32_t num){
    for(int i = threadIdx.x; i < num; i += blockDim.x){
        asm volatile("cvt.rna.tf32.f32 %0, %0;\n"
            :: "r"(A[i]));
    }
}
// wgmma tf32 m64n8k8 函数， A 和 B 在 shared memory
DEVICE void wgmma_tf32_m64n8k8_no_trans_ss(float *d_A, float *d_B, float *d_C){
    uint32_t const* A   = reinterpret_cast<uint32_t const*>(d_A);
    uint32_t const* B   = reinterpret_cast<uint32_t const*>(d_B);
    convert_fp32_to_tf32_shared(A, 64 * 8);
    convert_fp32_to_tf32_shared(B, 8 * 8);
    uint64_t desc_a = create_wgmma_descriptor_no_swizzle(d_A, 256 * sizeof(float), 32 * sizeof(float));
    uint64_t desc_b = create_wgmma_descriptor_no_swizzle(d_B, 32 * sizeof(float), 32 * sizeof(float));
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32"
                "{%0, %1, %2, %3},"
                "%4, %5, 1, 1, 1;\n"
                :"+f"(d_C[0]), "+f"(d_C[1]), "+f"(d_C[2]), "+f"(d_C[3])
                :"l"(desc_a), "l"(desc_b)
            );
}
#endif

template <u32 REGS>
DEVICE void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

template <u32 REGS>
DEVICE void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(REGS));
}