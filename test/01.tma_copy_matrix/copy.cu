/*
    这个文件实现一个简单的 TMA 搬运矩阵
    搬运模式符合 WGMMA 特性
    以 tf32 精度， m64n8k8 为例
*/
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

#define CHECK_CU(call) do { \
  CUresult e = (call); \
  if (e != CUDA_SUCCESS) { \
    const char* errStr; \
    cuGetErrorString(e, &errStr); \
    fprintf(stderr, "CUDA Driver API 错误 %s:%d: %s\n", __FILE__, __LINE__, errStr); \
    exit(1); \
  } \
} while(0)

// 辅助函数：将共享内存指针转换为 uint32_t
__forceinline__ __device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// mbarrier 初始化函数
__forceinline__ __device__ void mbarrier_init(uint64_t &barrier, int32_t count) {
  auto smem_addr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               "mbarrier.init.shared::cta.b64 [%0], %1;\n"
               "\t}" ::"r"(smem_addr),
               "r"(count));
}

// mbarrier arrive 函数
__forceinline__ __device__ void mbarrier_arrive(uint64_t &barrier) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
               "\t}"
               :
               : "r"(smem_addr));
}

// mbarrier arrive and wait 函数
__forceinline__ __device__ void mbarrier_arrive_and_wait(uint64_t &barrier) {
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

// 问题规模
constexpr int M = 64;  // 行数
constexpr int N = 8;   // 列数

// TMA 拷贝和验证 kernel
// 使用 TMA bulk tensor copy 从全局内存搬运数据到共享内存
__global__ void copy_and_verify_kernel(
    const __grid_constant__ CUtensorMap tensorMap,
    int *out_fail_count,
    float *out_smem_data)  // 新增：用于输出共享内存内容
{
    // 共享内存用于存放 B tile: M x N 个 float
    __shared__ alignas(128) float s_tile[M * N];
    
    // 共享内存 mbarrier，用于 TMA 同步
    __shared__ uint64_t mbarrier;

    const int tid = threadIdx.x;

    // 初始化 mbarrier：所有线程数作为 arrive count
    if (tid == 0) {
        mbarrier_init(mbarrier, blockDim.x);
    }
    __syncthreads();

    // 线程 0 发起 TMA 拷贝
    if (tid == 0) {
        uint32_t smem_addr = cast_smem_ptr_to_uint(s_tile);
        uint32_t mbar_addr = cast_smem_ptr_to_uint(&mbarrier);
        
        // TMA 坐标：拷贝第 (0, 0) 个 tile
        int32_t tile_coord_m = 0;
        int32_t tile_coord_n = 0;
        int32_t tile_coord_k = 0;
        // 预期传输的字节数
        uint32_t expected_bytes = M * N * sizeof(float);
        
        // 设置 mbarrier 预期的传输字节数
        asm volatile(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_addr),
               "r"(expected_bytes)
        );
        
        // 发起 TMA 拷贝
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4}], [%5];"
            :: "r"(smem_addr),
               "l"(&tensorMap),
               "r"(tile_coord_m),
               "r"(tile_coord_n),
               "r"(tile_coord_k),
               "r"(mbar_addr)
            : "memory"
        );
    }

    // 所有线程都需要 arrive and wait
    mbarrier_arrive_and_wait(mbarrier);
    
    // 确保拷贝完成后所有线程才能继续
    __syncthreads();

    // 将共享内存内容拷贝到全局内存（用于打印）
    const int total_elems = M * N;
    for (int i = tid; i < total_elems; i += blockDim.x) {
        out_smem_data[i] = s_tile[i];
    }
    __syncthreads();

    // 现在每个线程验证共享内存中的一些元素
    // 我们将 M*N 个元素大致平均分配给 32 个线程
    const int per_thread = (total_elems + 32 - 1) / 32;  // 向上取整
    const int start = tid * per_thread;
    const int end = min(total_elems, start + per_thread);

    int local_fail = 0;
    for (int idx = start; idx < end; ++idx) {
        int r = idx / N;
        int c = idx % N;
        float v_shared = s_tile[r * N + c];
        // 期望值：从全局内存的原始布局中获取
        // 注意：由于我们使用 tensor map，实际数据应该已经正确搬运到共享内存
        // 验证时需要知道原始数据的值，这里假设数据是 row * 1000 + col
        float v_expected = float(r * 1000 + c);
        
        if (v_shared != v_expected) {
            // 对于浮点数相等性：这些是按位拷贝的，所以精确比较是可行的
            local_fail++;
            // 调试时可以打印不匹配的情况（但从多个线程打印会很嘈杂）
            // if (tid == 0) printf("不匹配位置 (%d,%d): shared=%f expected=%f\n", r, c, v_shared, v_expected);
        }
    }

    // 将 local_fail 跨 warp 归约 -> 线程 0 写入全局 out_fail_count
    // 简单的原子加法
    if (local_fail > 0) {
        atomicAdd(out_fail_count, local_fail);
    }
}

int main() {
    printf("使用真实 TMA 的共享内存拷贝 + 布局验证 (m64n8k8 示例)\n");

    // 初始化 CUDA Driver API
    CHECK_CU(cuInit(0));

    // 在 Host 端分配 B 矩阵 (M x N)，连续的行主序，leading dimension = N
    const int ld_host = N;  // 紧密打包
    size_t sz = (size_t)M * N;
    float *h_B = (float*)malloc(sz * sizeof(float));
    assert(h_B);

    // 填充一个简单的模式以便验证：value = row * 1000 + col
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            h_B[r * ld_host + c] = float(r * 1000 + c);
        }
    }

    // Device 缓冲区
    alignas(16) float* d_B = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_B, sz * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sz * sizeof(float), cudaMemcpyHostToDevice));

    int *d_fail = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_fail, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_fail, 0, sizeof(int)));

    // 分配输出缓冲区用于存储共享内存内容
    float *d_smem_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_smem_data, sz * sizeof(float)));

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
        return 1;
    }
    
    printf("TensorMap 创建成功\n");
    printf("  全局维度: [%lu, %lu, %lu]\n", globalDim[0], globalDim[1], globalDim[2]);
    printf("  Box 维度: [%u, %u, %u]\n", boxDim[0], boxDim[1], boxDim[2]);
    printf("  全局步长: [%lu, %lu] 字节\n", globalStride[0], globalStride[1]);

    // 启动 kernel: 1 block, 32 threads (一个 warp)
    const int threads = 32;
    const int blocks = 1;
    // 共享内存大小 = M*N*sizeof(float) + mbarrier 大小
    // 实际上 mbarrier 是在共享内存中声明的，不需要额外的动态共享内存
    size_t shared_bytes = 0;  // 使用静态共享内存

    // 注意：使用 __grid_constant__ 参数需要 CUDA 12.0+ 和 Hopper 架构
    copy_and_verify_kernel<<<blocks, threads, shared_bytes>>>(tensorMap, d_fail, d_smem_data);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_fail = 0;
    CHECK_CUDA(cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // 拷贝共享内存内容回主机并打印
    float *h_smem_data = (float*)malloc(sz * sizeof(float));
    assert(h_smem_data);
    CHECK_CUDA(cudaMemcpy(h_smem_data, d_smem_data, sz * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n共享内存内容（按行顺序）:\n");
    printf("格式: [行,列] = 值\n");
    printf("----------------------------\n");
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * N + c;
            printf("[%2d,%2d] = %8.1f  ", r, c, h_smem_data[idx]);
            if ((c + 1) % 8 == 0) {
                printf("\n");
            }
        }
        if ((r + 1) % 8 == 0) {
            printf("\n");  // 每 8 行额外换行，方便阅读
        }
    }
    printf("----------------------------\n\n");

    if (h_fail == 0) {
        printf("✅ 通过：TMA 拷贝后所有元素都匹配。\n");
    } else {
        printf("❌ 失败：发现 %d 处不匹配。\n", h_fail);
    }

    free(h_smem_data);

    // 清理资源
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_fail));
    CHECK_CUDA(cudaFree(d_smem_data));
    free(h_B);

    return 0;
}
