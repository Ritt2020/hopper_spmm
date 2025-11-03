#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utils/utils.h"
#include "ptx/wgmma.h"
#include "ptx/tma.h"
#include "ptx/barrier.h"
#include "ptx/ptx_utils.h"
#include "scheduler.h"
#include "utils/convert.h"
#include "utils/class.h"
#include "utils/load.h"

#define WGMMA_M 64
#define WGMMA_N 8
#define WGMMA_K 16

constexpr int STAGES = 2;
constexpr int WARPGROUP_SIZE = 128;
constexpr int WARP_SIZE = 32;
constexpr int PRODUCER_WARPS = 4;
constexpr int CONSUMER_WARPS = 8;
constexpr int WARPGROUPS = 3;
constexpr int NUM_SMS = 132; // Hopper 132 SMs
constexpr int NUM_THREADS = (PRODUCER_WARPS + CONSUMER_WARPS) * WARP_SIZE;

struct SharedStorage {
    alignas(128) __half A[WGMMA_N * WGMMA_K * STAGES];
    alignas(128) __half B[WGMMA_K * WGMMA_M * STAGES * (WARPGROUPS-1)];
    alignas(128) u32 a2b[WGMMA_K * STAGES];
};

__global__ __launch_bounds__(NUM_THREADS) void fp16_spmm_kernel(
    const __grid_constant__ CUtensorMap tensorMapA,
    MAT_PTR_TYPE *d_rowOffset,
    MAT_IDX_TYPE *d_tcA2B,
    MAT_VAL_TYPE *d_data,
    MAT_IDX_TYPE *d_rowIdx,
    u32 eff_row_windows,
    u32 feature_dim,
    MAT_VAL_TYPE *d_dense_b,
    MAT_VAL_TYPE *d_dense_c
){
    // 共享内存
    extern __shared__ __align__(128) uint8_t dynamic_smem[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(dynamic_smem);

    int tid = threadIdx.x;
    int wgid = tid / WARPGROUP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int wg_tid = tid % WARPGROUP_SIZE;
    bool is_producer = warp_id < PRODUCER_WARPS;
    // barriers
    __shared__ __align__(8) u64 pro_bar[STAGES]; // 生产者 barrier
    __shared__ __align__(8) u64 con_bar[STAGES]; // 消费者 barrier
    __shared__ __align__(8) u64 row_bar; // 行窗口 barrier
    __shared__ __align__(8) u64 pro; // 生产者 barrier
    // 初始化 barriers
    if(tid == 0){
        for(int i = 0; i < STAGES; i++){
            mbarrier_init(pro_bar[i], 1);
            mbarrier_init(con_bar[i], WARPGROUPS-1);  
        }
        mbarrier_init(row_bar, WARPGROUPS-1);
        mbarrier_init(pro, 1);
    }
    fence_proxy_async_shared();
    __syncthreads();

    if(is_producer){
        /*
            生产者：
            大循环：行窗口循环
            循环内：首先等待行窗口barrier，然后取数据：
            1. 等待共享内存的 barrier， 然后每次取一个 A 和 B 的 128 维，到 shared memory
            2. 到达消费者的 barrier 通知消费者可以开始计算
            3. 等待下一个生产者 barrier
            4. 重复上述步骤，直到行窗口结束，结束后等待行窗口 barrier，进行下一个循环
        */ 
        int stage = 0;
        int phase = 0;
        int row_phase = 0;
        for(int row = blockIdx.x; row < eff_row_windows; row += gridDim.x){
            // 等待行窗口开始
            mbarrier_wait(row_bar, row_phase);
            // 内层循环：A块
            for(auto tile = d_rowOffset[row]; tile < d_rowOffset[row+1]; tile++){
                // 等待消费者 barrier
                if(wg_tid == 0){
                    // 设置expect tx
                    mbarrier_wait(con_bar[stage], phase);
                    mbarrier_expect_tx(pro_bar[stage], (WGMMA_M * WGMMA_K * (WARPGROUPS-1) + WGMMA_K * WGMMA_N) * sizeof(MAT_VAL_TYPE));
                    mbarrier_arrive(pro, 1);
                }
                mbarrier_wait(pro, 0);
                // load A and B
                if(wg_tid == 0){
                    tma_cp_async_bulk_4d_shared_global_tile_mbarrier_bytes(&smem.A[stage * WGMMA_N * WGMMA_K], &tensorMapA, 0, 0, 0, tile, pro_bar[stage]);
                }
                int r = wg_tid >> 4;
                int c = wg_tid & 15;
                tma_cp_async_bulk_1d(&smem.B[stage * WGMMA_K * WGMMA_M + WGMMA_K * 8 * c + r * 8], d_dense_b + r * feature_dim + c * 8, 8 * sizeof(MAT_VAL_TYPE), pro_bar[stage]);
                tma_cp_async_bulk_1d(&smem.B[WGMMA_K * WGMMA_M * STAGES + stage * WGMMA_K * WGMMA_M + WGMMA_K * 8 * c + r * 8], d_dense_b + r * feature_dim + c * 8 + 64, 8 * sizeof(MAT_VAL_TYPE), pro_bar[stage]);
                stage++;
                if(stage == STAGES){
                    stage = 0;
                    phase ^= 1;
                }
            }
            row_phase ^= 1;
        }
    }
    else{
        /*
            消费者：
            大循环：行窗口循环
            循环前：
            arrive 所有共享内存的生产者 barrier 和行窗口 barrier
            循环内：
            1. 等待对应的共享内存的 barrier， 然后进行一个 wgmma 计算
            2. arrive生产者的 barrier 通知生产者可以开始下一个循环取数据
            3. 等待下一个消费者 barrier
            4. 重复上述步骤，直到行窗口结束，结束后到达行窗口 barrier
        */ 
        int con_wgid = wgid - 1; // 消费者组ID
        int stage = 0;
        int phase = 0;
        float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if(wg_tid == 0){
            mbarrier_arrive(row_bar, 1);
            for(int i = con_wgid; i < STAGES; i += WARPGROUPS - 1){
                mbarrier_arrive(con_bar[i], 1);
            }
        }
        for(int row = blockIdx.x; row < eff_row_windows; row += gridDim.x){
            // 计算行
            for(auto tile = d_rowOffset[row]; tile < d_rowOffset[row+1]; tile++){
                // 等待生产者 barrier
                mbarrier_wait(pro_bar[stage], phase);
                // wgmma操作
                wgmma_fence();
                fence_proxy_async_shared();
                // 启动wgmma                
                wgmma_fp16_m64n8k16_ss(&smem.B[stage * WGMMA_K * WGMMA_M + (con_wgid) * STAGES * WGMMA_K * WGMMA_M], &smem.A[stage * WGMMA_N * WGMMA_K], C);
                wgmma_commit_group();
                wgmma_wait_group();
                // arrive at consumer barrier
                if(wg_tid == 0){
                    mbarrier_arrive(con_bar[stage], 1);
                }
                stage ++;
                if(stage == STAGES){
                    stage = 0;
                    phase ^= 1;
                }
            }
            if(wg_tid == 0){
                mbarrier_arrive(row_bar, 1);
            }
            uint32_t row_group = (threadIdx.x >> 5) << 4;
            uint32_t row_in_group = (threadIdx.x & 31) >> 2;
            uint32_t col_group = threadIdx.x & 3;
            //write back C
            d_dense_c[(d_rowIdx[row] + (col_group << 1)) * feature_dim + (row_group + row_in_group)] = C[0];
            d_dense_c[(d_rowIdx[row] + (col_group << 1) + 1) * feature_dim + (row_group + row_in_group)] = C[1];
            d_dense_c[(d_rowIdx[row] + (col_group << 1)) * feature_dim + (row_group + row_in_group + 8)] = C[2];
            d_dense_c[(d_rowIdx[row] + (col_group << 1) + 1) * feature_dim + (row_group + row_in_group + 8)] = C[3];
            
        }
    }

}

CUtensorMap create_tma_desc_A(MAT_VAL_TYPE *d_A, u32 total_tiles){
    // 创建 TensorMap
    alignas(64) CUtensorMap tensorMap;
    // Tensor Map 参数
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    uint32_t tensorRank = 4;  // 4D tensor
    
    uint64_t globalDim[4] = { 8ULL, 8ULL, 2ULL, (u64)total_tiles };
    
    uint64_t globalStride[3] = { 16 * sizeof(MAT_VAL_TYPE), 8 * sizeof(MAT_VAL_TYPE), 128 * sizeof(MAT_VAL_TYPE) };
    
    uint32_t boxDim[4] = { 8, 8, 2, 1 };
    uint32_t elementStride[4] = { 1, 1, 1, 1 };
    
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

PERF_RESULT fp16_spmm(BTCF_MTX_NO_BITMAP &btcf_a, DENSE_MTX &dense_b, DENSE_MTX &dense_c){

    PERF_RESULT perf_result;
    perf_result.time = 0.0f;
    perf_result.runs = RUNS;

    // 准备 GPU 结构
    MAT_PTR_TYPE *d_rowOffset = nullptr;
    MAT_IDX_TYPE * d_tcA2B = nullptr;
    MAT_VAL_TYPE * d_data = nullptr;
    MAT_IDX_TYPE * d_rowIdx = nullptr;
    MAT_VAL_TYPE *d_dense_b = nullptr;
    MAT_VAL_TYPE *d_dense_c = nullptr;

    // 设备内存分配
    CHECK_CUDA(cudaMalloc(&d_rowOffset, btcf_a.rowOffset.size() * sizeof(MAT_PTR_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_tcA2B, btcf_a.tcA2B.size() * sizeof(MAT_IDX_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_data, btcf_a.data.size() * sizeof(MAT_VAL_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_rowIdx, btcf_a.rowIdx.size() * sizeof(MAT_IDX_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_dense_b, dense_b.rows * dense_b.cols * sizeof(MAT_VAL_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_dense_c, dense_c.rows * dense_c.cols * sizeof(MAT_VAL_TYPE)));

    // 数据拷贝
    CHECK_CUDA(cudaMemcpy(d_rowOffset, btcf_a.rowOffset.data(), btcf_a.rowOffset.size() * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tcA2B, btcf_a.tcA2B.data(), btcf_a.tcA2B.size() * sizeof(MAT_IDX_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_data, btcf_a.data.data(), btcf_a.data.size() * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowIdx, btcf_a.rowIdx.data(), btcf_a.rowIdx.size() * sizeof(MAT_IDX_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dense_b, dense_b.values, dense_b.rows * dense_b.cols * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dense_c, dense_c.values, dense_c.rows * dense_c.cols * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice));
    
    // 创建 tensor map
    CUtensorMap tensorMapA = create_tma_desc_A(d_data, btcf_a.total_tiles);
    
    // 开始warm up
    for(int i = 0; i < WARMUP_RUNS; i++){
        fp16_spmm_kernel<<<NUM_SMS, NUM_THREADS, sizeof(SharedStorage)>>>(tensorMapA, d_rowOffset, d_tcA2B, d_data, d_rowIdx, btcf_a.eff_row_windows, dense_b.cols, d_dense_b, d_dense_c);
    }
    // 计时，正式运行
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    for(int i = 0; i < RUNS; i++){
        fp16_spmm_kernel<<<NUM_SMS, NUM_THREADS, sizeof(SharedStorage)>>>(tensorMapA, d_rowOffset, d_tcA2B, d_data, d_rowIdx, btcf_a.eff_row_windows, dense_b.cols, d_dense_b, d_dense_c);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&perf_result.time, start, stop));
    // 验证结果

    // 清理资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_rowOffset));
    CHECK_CUDA(cudaFree(d_tcA2B));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_rowIdx));
    CHECK_CUDA(cudaFree(d_dense_b));
    CHECK_CUDA(cudaFree(d_dense_c));
    return perf_result;
}

int main(int argc, char* argv[]){
    // 读取矩阵文件，转换格式
    if(argc != 3){
        printf("Usage: %s <matrix_file> <feature_dim>\n", argv[0]);
        return 1;
    }
    const char* matrix_file = argv[1];
    vint feature_dim = atoi(argv[2]);
    CSR_MTX csr = load_matrix(matrix_file, false);
    BTCF_MTX_NO_BITMAP btcf;
    CSR2BTCFNOBITMAP(csr, btcf);
    // 打印BTCF内容
    printf("=== BTCF 矩阵信息 ===\n");
    printf("行数: %d\n", btcf.rows);
    printf("列数: %d\n", btcf.cols);
    printf("非零元素数: %d\n", btcf.nnzs);
    printf("总行窗口数: %d\n", btcf.total_row_windows);
    printf("非零行窗口数: %d\n", btcf.eff_row_windows);
    printf("总TC块（8*16）数: %lu\n", btcf.tcA2B.size() / COL_WINDOW);
    // 随机填充dense_b
    DENSE_MTX dense_b;
    dense_b.rows = btcf.cols;
    dense_b.cols = feature_dim;
    dense_b.values = new __half[dense_b.rows * dense_b.cols];
    for(int i = 0; i < dense_b.rows * dense_b.cols; i++){
        dense_b.values[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    DENSE_MTX dense_c;
    dense_c.rows = btcf.rows;
    dense_c.cols = feature_dim;
    dense_c.values = new __half[dense_c.rows * dense_c.cols];
    // 执行SPMM
    PERF_RESULT perf_result = fp16_spmm(btcf, dense_b, dense_c);
    perf_result.print();
    // 结果验证

}

