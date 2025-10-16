#pragma once
#define DEVICE __device__ __forceinline__

DEVICE uint32_t encode(uint32_t x){
    return (x & 0x3FFFF) >> 4;
}
// 指针转换
DEVICE uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
// mbarrier 初始化函数
DEVICE void mbarrier_init(uint64_t &barrier, int32_t count) {
    auto smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                "mbarrier.init.shared::cta.b64 [%0], %1;\n"
                "\t}" ::"r"(smem_addr),
                "r"(count));
}
DEVICE void mbarrier_wait(uint64_t& barrier, int phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
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
DEVICE void mbarrier_arrive(uint64_t &barrier) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                    "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                    "\t}"
                    :: "r"(smem_addr));
}
// mbarrier arrive 函数 有 count
DEVICE void mbarrier_arrive(uint64_t &barrier, int count) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" 
        ::"r"(smem_addr), "r"(count)
        : "memory");
}

// mbarrier arrive and wait 函数
DEVICE void mbarrier_arrive_and_wait(uint64_t &barrier) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
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
DEVICE void mbarrier_expect_tx(uint64_t &barrier, uint32_t bytes) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" 
        ::"r"(smem_addr),
        "r"(bytes)
    );
}
// 2D TMA
DEVICE void tma_cp_async_bulk_2d_shared_global_tile_mbarrier_bytes(
    float* smem_dst,
    void const* const src_tma_desc,
    int32_t tile_coord_i,
    int32_t tile_coord_j,
    uint64_t &mbarrier)
{
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_desc);
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_dst);
    uint32_t mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
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
    float* smem_dst,
    void const* const src_tma_desc,
    int32_t tile_coord_i,
    int32_t tile_coord_j,
    int32_t tile_coord_k,
    uint64_t &mbarrier)
{
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_desc);
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_dst);
    uint32_t mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
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
DEVICE uint64_t create_wgmma_descriptor_no_swizzle(float *ptr, uint32_t lbo, uint32_t sbo){
    uint64_t desc = 0;
    desc |= (uint64_t)encode(lbo) << 16;
    desc |= (uint64_t)encode(sbo) << 32;
    desc |= (uint64_t)encode(__cvta_generic_to_shared(ptr));
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

template <uint32_t REGS>
DEVICE void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

template <uint32_t REGS>
DEVICE void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(REGS));
}