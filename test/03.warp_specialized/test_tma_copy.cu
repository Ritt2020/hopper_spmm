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

constexpr int NUM_THREADS = 128;

// 简单的TMA测试kernel - 只做数据拷贝
__global__ void test_tma_copy_kernel(
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    float *d_A_out,
    float *d_B_out,
    int tile_num
) {
    extern __shared__ __align__(128) float smem[];
    
    // A 和 B 的共享内存区域
    float* smem_A = smem;
    float* smem_B = smem + TILE_M * TILE_K;
    
    int tid = threadIdx.x;
    
    // 使用一个简单的barrier
    __shared__ __align__(8) uint64_t barrier;
    
    if(tid == 0) {
        mbarrier_init(barrier, 1);
    }
    __syncthreads();
    
    // 每个tile进行测试
    for(int bid = blockIdx.x; bid < tile_num; bid += gridDim.x) {
        if(tid == 0) {
            // 设置期望的数据量
            mbarrier_expect_tx(barrier, (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float));
            
            // 使用TMA加载A和B
            // 注意：bid 需要乘以 boxDim[2]=2，因为每个tile在第三维占用2个单位
            tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(
                smem_A, &tensorMapA, 0, 0, bid * 2, barrier);
            tma_cp_async_bulk_3d_shared_global_tile_mbarrier_bytes(
                smem_B, &tensorMapB, 0, 0, bid * 2, barrier);
        }
        
        // 等待TMA完成
        mbarrier_wait(barrier, 0);
        __syncthreads();
        
        // 将数据从共享内存拷贝回全局内存
        int offset_A = bid * TILE_M * TILE_K;
        int offset_B = bid * TILE_K * TILE_N;
        
        // 每个线程拷贝一部分数据
        for(int i = tid; i < TILE_M * TILE_K; i += NUM_THREADS) {
            d_A_out[offset_A + i] = smem_A[i];
        }
        
        for(int i = tid; i < TILE_K * TILE_N; i += NUM_THREADS) {
            d_B_out[offset_B + i] = smem_B[i];
        }
        
        __syncthreads();
        
        // 重置barrier
        if(tid == 0) {
            mbarrier_arrive(barrier, 1);
        }
        __syncthreads();
    }
}

CUtensorMap create_tma_desc_A(float *d_A){
    // 创建 TensorMap
    alignas(64) CUtensorMap tensorMap;
    // Tensor Map 参数
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    uint32_t tensorRank = 3;  // 3D tensor
    
    uint64_t globalDim[3] = { 4ULL, (uint64_t)TILE_M, (uint64_t)2 * TILE_NUM };
    
    // globalStride[0]: 在dim1(TILE_M)方向移动一个单位的stride = 一行的字节数
    // globalStride[1]: 在dim2(2*TILE_NUM)方向移动一个单位的stride = boxDim[2]=2 的字节数
    uint64_t globalStride[2] = { TILE_K * TILE_NUM * sizeof(float), 4 * sizeof(float) };
    
    uint32_t boxDim[3] = { 4, TILE_M, 2 };
    uint32_t elementStride[3] = { 1, 1, 1 };
    
    // TMA 填充模式
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    printf("=== TensorMap A 参数 ===\n");
    printf("globalDim: [%llu, %llu, %llu]\n", globalDim[0], globalDim[1], globalDim[2]);
    printf("globalStride: [%llu, %llu]\n", globalStride[0], globalStride[1]);
    printf("boxDim: [%u, %u, %u]\n", boxDim[0], boxDim[1], boxDim[2]);
    printf("TILE_K=%d, TILE_M=%d, TILE_NUM=%d\n", TILE_K, TILE_M, TILE_NUM);
    printf("矩阵A形状: [%d, %d] (行主序)\n", TILE_M, TILE_K * TILE_NUM);
    
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
    
    printf("\n=== TensorMap B 参数 ===\n");
    printf("globalDim: [%llu, %llu, %llu]\n", globalDim[0], globalDim[1], globalDim[2]);
    printf("globalStride: [%llu, %llu]\n", globalStride[0], globalStride[1]);
    printf("boxDim: [%u, %u, %u]\n", boxDim[0], boxDim[1], boxDim[2]);
    printf("TILE_K=%d, TILE_N=%d, TILE_NUM=%d\n", TILE_K, TILE_N, TILE_NUM);
    printf("矩阵B形状: [%d, %d] (列主序)\n", TILE_K * TILE_NUM, TILE_N);
    
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

// 验证TMA拷贝的正确性：检查每个拷贝出来的tile是否相同
bool verify_tma_copy_tiles(float *h_copied, int tile_size, int tile_num, const char* name) {
    printf("\n=== 验证 %s (检查 %d 个 tile，每个 %d 元素) ===\n", name, tile_num, tile_size);
    int error_count = 0;
    int max_errors_to_show = 10;
    
    // 以第一个tile作为参考
    float* ref_tile = h_copied;
    
    // 检查其他tile是否与第一个相同
    for(int tile = 1; tile < tile_num; tile++) {
        float* current_tile = h_copied + tile * tile_size;
        for(int i = 0; i < tile_size; i++) {
            if(ref_tile[i] != current_tile[i]) {
                if(error_count < max_errors_to_show) {
                    printf("Tile %d 位置[%d] 不匹配: tile0=%.6f, tile%d=%.6f\n", 
                        tile, i, ref_tile[i], tile, current_tile[i]);
                }
                error_count++;
            }
        }
    }
    
    if(error_count > 0) {
        printf("总错误数: %d / %d (在 %d 个tile中)\n", error_count, tile_size * tile_num, tile_num);
        return false;
    } else {
        printf("✓ 所有 %d 个 tile 完全一致！\n", tile_num);
        
        // 打印第一个tile的前几个元素作为参考
        printf("  参考数据（tile 0 的前10个元素）: ");
        for(int i = 0; i < std::min(10, tile_size); i++) {
            printf("%.1f ", ref_tile[i]);
        }
        printf("\n");
        return true;
    }
}

// 打印矩阵的一部分用于调试
void print_matrix_sample(float *data, int rows, int cols, const char* name, bool row_major) {
    printf("\n=== %s 样本数据 (前4x4) ===\n", name);
    for(int i = 0; i < std::min(4, rows); i++) {
        for(int j = 0; j < std::min(4, cols); j++) {
            int idx;
            if(row_major) {
                idx = i * cols + j;
            } else {
                idx = j * rows + i;
            }
            printf("%.3f ", data[idx]);
        }
        printf("\n");
    }
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
    printf("TILE_M = %d, TILE_K = %d, TILE_N = %d\n", TILE_M, TILE_K, TILE_N);
    
    // 准备数据
    printf("\n准备测试数据...\n");
    int size_A = TILE_M * TILE_K * TILE_NUM;
    int size_B = TILE_K * TILE_NUM * TILE_N;
    
    float *h_A = (float *)malloc(size_A * sizeof(float));
    float *h_B = (float *)malloc(size_B * sizeof(float));
    float *h_A_result = (float *)malloc(size_A * sizeof(float));
    float *h_B_result = (float *)malloc(size_B * sizeof(float));
    
    // 初始化：每个tile的数据相同，便于验证TMA是否正确读取每个tile
    // 对于矩阵A（行主序，64x80）：将它分成10个tile，每个tile是64x8
    // 每8列填充相同的模式
    for(int row = 0; row < TILE_M; row++) {
        for(int tile = 0; tile < TILE_NUM; tile++) {
            for(int col = 0; col < TILE_K; col++) {
                int idx = row * (TILE_K * TILE_NUM) + tile * TILE_K + col;
                // 使用行内偏移作为值，这样每个tile在同一行同一列的值相同
                h_A[idx] = (float)(row * TILE_K + col) + 1.0f;
            }
        }
    }
    
    // 对于矩阵B（列主序，80x8）：将它分成10个tile，每个tile是8x8
    // 每8行填充相同的模式
    for(int col = 0; col < TILE_N; col++) {
        for(int tile = 0; tile < TILE_NUM; tile++) {
            for(int row = 0; row < TILE_K; row++) {
                int idx = col * (TILE_K * TILE_NUM) + tile * TILE_K + row;
                // 使用列内偏移作为值，这样每个tile在同一行同一列的值相同
                h_B[idx] = (float)(col * TILE_K + row + 1000) + 1.0f;
            }
        }
    }
    
    print_matrix_sample(h_A, TILE_M, TILE_K * TILE_NUM, "矩阵A (行主序)", true);
    print_matrix_sample(h_B, TILE_K * TILE_NUM, TILE_N, "矩阵B (列主序)", false);
    
    // 拷贝到 GPU
    printf("\n拷贝数据到GPU...\n");
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_A_out = nullptr;
    float *d_B_out = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A_out, size_A * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_out, size_B * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_A_out, 0, size_A * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_B_out, 0, size_B * sizeof(float)));
    
    // 创建 tensor map
    printf("\n创建 Tensor Maps...\n");
    CUtensorMap tensorMapA = create_tma_desc_A(d_A);
    CUtensorMap tensorMapB = create_tma_desc_B(d_B);
    
    // 计算shared memory 大小
    size_t shared_mem_size = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    printf("\nShared Memory 大小: %zu bytes\n", shared_mem_size);
    
    // 执行 kernel
    printf("\n执行 TMA Copy Kernel...\n");
    test_tma_copy_kernel<<<1, NUM_THREADS, shared_mem_size>>>(
        tensorMapA, tensorMapB, d_A_out, d_B_out, TILE_NUM);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel 执行完成\n");
    
    // 拷贝结果回CPU
    CHECK_CUDA(cudaMemcpy(h_A_result, d_A_out, size_A * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_result, d_B_out, size_B * sizeof(float), cudaMemcpyDeviceToHost));
    
    print_matrix_sample(h_A_result, TILE_M, TILE_K * TILE_NUM, "TMA拷贝后矩阵A (行主序)", true);
    print_matrix_sample(h_B_result, TILE_K * TILE_NUM, TILE_N, "TMA拷贝后矩阵B (列主序)", false);
    
    // 验证结果：检查每个tile是否都拷贝成相同的数据
    bool a_correct = verify_tma_copy_tiles(h_A_result, TILE_M * TILE_K, TILE_NUM, "矩阵A");
    bool b_correct = verify_tma_copy_tiles(h_B_result, TILE_K * TILE_N, TILE_NUM, "矩阵B");
    
    printf("\n=== 最终结果 ===\n");
    if(a_correct && b_correct) {
        printf("✓ TMA拷贝验证通过！\n");
    } else {
        printf("✗ TMA拷贝验证失败！\n");
        if(!a_correct) printf("  - 矩阵A拷贝错误\n");
        if(!b_correct) printf("  - 矩阵B拷贝错误\n");
    }
    
    // 释放内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_A_out));
    CHECK_CUDA(cudaFree(d_B_out));
    free(h_A);
    free(h_B);
    free(h_A_result);
    free(h_B_result);
    
    return (a_correct && b_correct) ? 0 : -1;
}

