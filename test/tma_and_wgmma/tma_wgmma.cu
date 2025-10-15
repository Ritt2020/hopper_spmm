// 开 4 个 warp 一个 warp group。首先一个线程启动 TMA，然后执行 WGMMA

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#define DEVICE __device__ __forceinline__
#define TILE_M 64
#define TILE_N 8
#define TILE_K 8

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

namespace ptx {
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
    // mbarrier arrive 函数
    DEVICE void mbarrier_arrive(uint64_t &barrier) {
        uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
        asm volatile("{\n\t"
                        "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                        "\t}"
                        :
                        : "r"(smem_addr));
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
    // mbarrier expect tx 函数: 设置 mbarrier 期望的传输字节数
    DEVICE void mbarrier_expect_tx(uint64_t &barrier, uint32_t bytes) {
        uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
        asm volatile(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
            :: "r"(smem_addr), "r"(bytes)
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
}

// producer kernel
__global__ void tma_wgmma_kernel(
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    float* d_C
) {
    __shared__ uint64_t mbarrier;
    __shared__ alignas(128) float sA[TILE_M * TILE_K];
    __shared__ alignas(128) float sB[TILE_N * TILE_K];
    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t row_group = (threadIdx.x >> 5) << 4;
    uint32_t row_in_group = (threadIdx.x & 31) >> 2;
    uint32_t col_group = threadIdx.x & 3;
    if(threadIdx.x == 0) {
        ptx::mbarrier_init(mbarrier, blockDim.x);
    }
    __syncthreads();
    /*
        TMA 搬运 A 和 B
    */
    if (threadIdx.x == 0) {
        uint32_t expected_bytes_A = TILE_M * TILE_K * sizeof(float);
        uint32_t expected_bytes_B = TILE_N * TILE_K * sizeof(float);
        // ptx::mbarrier_expect_tx(mbarrier, expected_bytes_A);
        ptx::mbarrier_expect_tx(mbarrier, expected_bytes_A + expected_bytes_B);
        ptx::tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(sA, &tensorMapA, 0, 0, 0, mbarrier);
        ptx::tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(sB, &tensorMapB, 0, 0, 0, mbarrier);
    }
    // 等待完成
    ptx::mbarrier_arrive_and_wait(mbarrier);
    __syncthreads();
    ptx::fence_proxy_async_shared();
    // 转换 tf32
    ptx::convert_fp32_to_tf32_shared(reinterpret_cast<uint32_t const*>(sA), TILE_M * TILE_K);
    ptx::convert_fp32_to_tf32_shared(reinterpret_cast<uint32_t const*>(sB), TILE_N * TILE_K);
    __syncthreads();
    // fence
    ptx::wgmma_fence();
    ptx::fence_proxy_async_shared();
    // compute
    ptx::wgmma_tf32_m64n8k8_no_trans_ss(sA, sB, C);
    // sync
    ptx::wgmma_commit_group();
    ptx::wgmma_wait_group();
    // store C to global memory
    d_C[(row_group + row_in_group) * 8 + (col_group << 1)] = C[0];
    d_C[(row_group + row_in_group) * 8 + (col_group << 1) + 1] = C[1];
    d_C[(row_group + row_in_group + 8) * 8 + (col_group << 1)] = C[2];
    d_C[(row_group + row_in_group + 8) * 8 + (col_group << 1) + 1] = C[3];
}

bool test_result_cpu(float *d_C, float *h_A, float *h_B, int M, int N, int K, float relative_tolerance = 0.01f){
    // 分配 host 端 C 空间，然后计算C = A * B
    float *h_C = (float *)malloc(M * N * sizeof(float));
    for(int i = 0; i < M * N; i++){
        h_C[i] = 0.0f;
    }
    // B 列主序存储
    // B 的下标: 行优先 [k * N + j] -> 列主序 [j * K + k]
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < K; k++){
                h_C[i * N + j] += h_A[i * K + k] * h_B[j * K + k];
            }
        }
    }
    // 把 device 拷贝过来对比
    float *d_C_copy = (float *)malloc(M * N * sizeof(float));
    cudaMemcpy(d_C_copy, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for(int i = 0; i < M * N; i++){
        float expected = h_C[i];
        float actual = d_C_copy[i];
        float relative_error = fabs(expected - actual) / (fabs(expected) + 1e-8f);
        if(relative_error > relative_tolerance){
            printf("测试失败，数据不正确 at index %d: h_C[%d] = %f, d_C_copy[%d] = %f, 相对误差 = %f\n", i, i, expected, i, actual, relative_error);
            is_correct = false;
        }
    }
    free(h_C);
    free(d_C_copy);
    return is_correct;
}

CUtensorMap create_tma_desc_A(float *d_A){
    // 创建 TensorMap
    alignas(64) CUtensorMap tensorMap;
    // Tensor Map 参数
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    uint32_t tensorRank = 3;  // 3D tensor
    
    uint64_t globalDim[3] = { 4ULL, 64ULL, 2ULL };
    
    uint64_t globalStride[2] = { 8 * sizeof(float), 4 * sizeof(float) };
    
    uint32_t boxDim[3] = { 4, 64, 2 };
    uint32_t elementStride[3] = { 1, 1, 1 };
    
    // TMA 填充模式
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    // 创建 tiled tensor map
    CUresult res = cuTensorMapEncodeTiled(
        &tensorMap,
        dataType,
        tensorRank,
        (void*)d_A,           // 全局内存基地址
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        interleave,
        swizzle,
        l2Promotion,
        oobFill
    );
    
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled 失败: %s\n", errStr);
    }
    return tensorMap;
}

CUtensorMap create_tma_desc_B(float *d_B){
    // 创建 TensorMap
    alignas(64) CUtensorMap tensorMap;
    // Tensor Map 参数
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    uint32_t tensorRank = 3;  // 3D tensor
    
    uint64_t globalDim[3] = { 4ULL, 8ULL, 2ULL };
    
    uint64_t globalStride[2] = { 8 * sizeof(float), 4 * sizeof(float) };
    
    uint32_t boxDim[3] = { 4, 8, 2 };
    uint32_t elementStride[3] = { 1, 1, 1 };
    
    // TMA 填充模式
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    // 创建 tiled tensor map
    CUresult res = cuTensorMapEncodeTiled(
        &tensorMap,
        dataType,
        tensorRank,
        (void*)d_B,           // 全局内存基地址
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        interleave,
        swizzle,
        l2Promotion,
        oobFill
    );
    
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled 失败: %s\n", errStr);
    }
    return tensorMap;
}

int main() {
    // A 和 B 都必须为 K-Major
    float *h_A = (float *)malloc(TILE_M * TILE_K * sizeof(float));
    for(int i = 0; i < TILE_M * TILE_K; i++){
        h_A[i] = i + 1;
    }
    // B 列主序
    float *h_B = (float *)malloc(TILE_K * TILE_N * sizeof(float));
    for(int i = 0; i < TILE_K * TILE_N; i++){
        h_B[i] = 1.0f + i * 0.01f;
    }

    // 分配 device 端空间
    alignas(16) float *d_A;
    alignas(16) float *d_B;
    float *d_C;
    cudaMalloc(&d_A, TILE_M * TILE_K * sizeof(float));
    cudaMalloc(&d_B, TILE_K * TILE_N * sizeof(float));
    cudaMalloc(&d_C, TILE_M * TILE_N * sizeof(float));

    // 拷贝数据到 device
    cudaMemcpy(d_A, h_A, TILE_M * TILE_K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, TILE_K * TILE_N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 tensor map
    auto tensorMapA = create_tma_desc_A(d_A);
    auto tensorMapB = create_tma_desc_B(d_B);

    CHECK_CUDA(cudaGetLastError());

    // 调用 kernel
    tma_wgmma_kernel<<<1, 128>>>(tensorMapA, tensorMapB, d_C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 测试结果
    bool is_correct = test_result_cpu(d_C, h_A, h_B, TILE_M, TILE_N, TILE_K, 0.01f);
    if(is_correct){
        printf("TMA WGMMA 测试通过，所有数据正确\n");
    }else{
        printf("TMA WGMMA 测试失败，数据不正确\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    return 0;
}