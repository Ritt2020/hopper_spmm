#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"
#include "ptx.h"

#define TILE_M 64
#define TILE_N 8
#define TILE_K 8

int TILE_NUM = 1;  // 默认值，可通过命令行参数修改

constexpr int STAGES = 3;
constexpr int NUM_SMS = 1;
constexpr int WARPGROUP_SIZE = 128;
constexpr int WARPGROUPS = 2;
constexpr int NUM_THREADS = WARPGROUPS * WARPGROUP_SIZE;

struct SharedStorage {
    alignas(128) float A[TILE_M * TILE_K * STAGES];
    alignas(128) float B[TILE_K * TILE_N * STAGES];
};

__global__ __launch_bounds__(NUM_THREADS) void ws_kernel(
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    float *d_C,
    int tile_num
) {
    // 共享内存空间， 128 字节对齐
    extern __shared__ __align__(128) uint8_t dynamic_smem[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(dynamic_smem);

    int tid = threadIdx.x;
    int wgid = tid / WARPGROUP_SIZE;
    int wg_tid = tid % WARPGROUP_SIZE;

    // 同步用 barriers
    __shared__ __align__(8) uint64_t pro_bar[STAGES];
    __shared__ __align__(8) uint64_t con_bar[STAGES];
    if(tid == 0){
        for(int i = 0; i < STAGES; i++){
            mbarrier_init(pro_bar[i], 1);
            mbarrier_init(con_bar[i], WARPGROUPS-1);
        }
    }
    fence_proxy_async_shared();
    __syncthreads();

    if(wgid == 0){
        // 生产者，取数据
        // setmaxnreg_dec<40>();
        // 主循环
        if(wg_tid == 0){
            int phase = 0;
            int stage = 0;
            for(auto bid = blockIdx.x; bid < tile_num; bid += gridDim.x){
                // 等待消费者
                mbarrier_wait(con_bar[stage], phase);
                // 设置expect tx
                mbarrier_expect_tx(pro_bar[stage], (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float));
                // load A 和 B
                tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(&smem.A[stage * TILE_M * TILE_K], &tensorMapA, 0, 0, bid * 2, pro_bar[stage]);
                tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(&smem.B[stage * TILE_K * TILE_N], &tensorMapB, 0, 0, bid * 2, pro_bar[stage]);
                stage++;
                if(stage == STAGES){
                    stage = 0;
                    phase ^= 1;
                }
            }
        }
    } else {
        // 消费者，计算，寄存器累加
        // setmaxnreg_inc<232>();
        float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int stage = 0;
        int phase = 0;
        if(wg_tid == 0){
            for(int i = 0; i < STAGES; i++){
                mbarrier_arrive(con_bar[i], 1);
            }
        }
        for(int bid = blockIdx.x; bid < tile_num; bid += gridDim.x){
            // 主循环
            mbarrier_wait(pro_bar[stage], phase);
            // wgmma
            wgmma_fence();
            // 转换 tf32
            convert_fp32_to_tf32_shared(reinterpret_cast<uint32_t const*>(&smem.A[stage * TILE_M * TILE_K]), TILE_M * TILE_K);
            convert_fp32_to_tf32_shared(reinterpret_cast<uint32_t const*>(&smem.B[stage * TILE_K * TILE_N]), TILE_K * TILE_N);
            // wgmma
            fence_proxy_async_shared();
            wgmma_tf32_m64n8k8_no_trans_ss(&smem.A[stage * TILE_M * TILE_K], &smem.B[stage * TILE_K * TILE_N], C);
            wgmma_commit_group();
            wgmma_wait_group();
            // arrive at consumer barrier
            if(wg_tid == 0){
                mbarrier_arrive(con_bar[stage], 1);
            }
            stage++;
            if(stage == STAGES){
                stage = 0;
                phase ^= 1;
            }
        }
        // write back C
        uint32_t row_group = (wg_tid >> 5) << 4;
        uint32_t row_in_group = (wg_tid & 31) >> 2;
        uint32_t col_group = wg_tid & 3;
        d_C[(row_group + row_in_group) * 8 + (col_group << 1)] = C[0];
        d_C[(row_group + row_in_group) * 8 + (col_group << 1) + 1] = C[1];
        d_C[(row_group + row_in_group + 8) * 8 + (col_group << 1)] = C[2];
        d_C[(row_group + row_in_group + 8) * 8 + (col_group << 1) + 1] = C[3];
    }
}

CUtensorMap create_tma_desc_A(float *d_A){
    // 创建 TensorMap
    alignas(64) CUtensorMap tensorMap;
    // Tensor Map 参数
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    uint32_t tensorRank = 3;  // 3D tensor
    
    uint64_t globalDim[3] = { 4ULL, (uint64_t)TILE_M, (uint64_t)2 * TILE_NUM };
    
    uint64_t globalStride[2] = { TILE_K * TILE_NUM * sizeof(float), 4 * sizeof(float) };
    
    uint32_t boxDim[3] = { 4, TILE_M, 2 };
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
    
    uint64_t globalDim[3] = { 4ULL, (uint64_t) TILE_N, (uint64_t)2 * TILE_NUM };
    
    uint64_t globalStride[2] = { TILE_K * TILE_NUM * sizeof(float), 4 * sizeof(float) };
    
    uint32_t boxDim[3] = { 4, TILE_N, 2 };
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


int main(int argc, char* argv[]) {
    // 解析命令行参数
    if (argc > 1) {
        TILE_NUM = atoi(argv[1]);
        if (TILE_NUM <= 0) {
            printf("错误：TILE_NUM 必须为正整数\n");
            printf("用法：%s [TILE_NUM]\n", argv[0]);
            printf("示例：%s 4\n", argv[0]);
            return -1;
        }
    }
    printf("使用 TILE_NUM = %d\n", TILE_NUM);
    /*
        准备数据。本程序测试 TILE_NUM 次 WGMMA。
        A 矩阵大小为 TILE_M, TILE_K * TILE_NUM 行主序
        B 矩阵大小为 TILE_K * TILE_NUM, TILE_N 列主序
        C 矩阵大小为 TILE_M, TILE_N 行主序
        初始化为随机数
        每次取 一个 TILE_M * TILE_K 的 A 块，一个 TILE_K * TILE_N 的 B 块，进行 WGMMA 计算，并累加到 同一个 C 中
    */
    float *h_A = (float *)malloc(TILE_M * TILE_K * TILE_NUM * sizeof(float));
    float *h_B = (float *)malloc(TILE_K * TILE_NUM * TILE_N * sizeof(float));
    
    // 随机初始化A矩阵和B矩阵，值在0到1之间
    srand(42); // 固定种子，方便复现
    for(int i = 0; i < TILE_M * TILE_K * TILE_NUM; i++){
        h_A[i] = (float)rand() / (float)RAND_MAX;
    }
    for(int i = 0; i < TILE_K * TILE_NUM * TILE_N; i++) {
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }
    // 拷贝到 GPU
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, TILE_M * TILE_K * TILE_NUM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, TILE_K * TILE_NUM * TILE_N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, TILE_M * TILE_K * TILE_NUM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, TILE_K * TILE_NUM * TILE_N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_C, TILE_M * TILE_N * sizeof(float)));
    // 创建 tensor map
    CUtensorMap tensorMapA = create_tma_desc_A(d_A);
    CUtensorMap tensorMapB = create_tma_desc_B(d_B);
    // 计算shared memory 大小
    size_t shared_mem_size = sizeof(SharedStorage);
    // 执行 kernel
    ws_kernel<<<NUM_SMS, NUM_THREADS, shared_mem_size>>>(tensorMapA, tensorMapB, d_C, TILE_NUM);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 将结果拷贝回主机并打印用于调试
    float *h_C = (float *)malloc(TILE_M * TILE_N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, TILE_M * TILE_N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 结果验证
    bool result_correct = verify_result(d_C, h_A, h_B, TILE_M, TILE_N, TILE_K, TILE_NUM);
    if(!result_correct) {
        printf("\n验证失败！计算结果不正确。\n");
        return -1;
    } else {
        printf("\n验证通过！计算结果正确。\n");
    }

    // 释放内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
/*
A 矩阵初始值：
0 1 2 3 4 5 6 7
8 9 10 11 12 13 14 15
16 17 18 19 20 21 22 23
24 25 26 27 28 29 30 31
32 33 34 35 36 37 38 39
40 41 42 43 44 45 46 47
48 49 50 51 52 53 54 55
56 57 58 59 60 61 62 63
...


*/

