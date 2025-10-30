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
    MAT_PTR_TYPE *d_rowOffset,
    MAT_IDX_TYPE *d_tcA2B,
    MAT_VAL_TYPE *d_data,
    MAT_IDX_TYPE *d_rowIdx,
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
    __shared__ __align__(8) uint64_t pro_bar[STAGES]; // 生产者 barrier
    __shared__ __align__(8) uint64_t con_bar[STAGES]; // 消费者 barrier
    __shared__ __align__(8) uint64_t row_bar; // 行窗口 barrier
    // 初始化 barriers
    if(tid == 0){
        for(int i = 0; i < STAGES; i++){
            mbarrier_init(pro_bar[i], 1);
            mbarrier_init(con_bar[i], WARPGROUPS-1);
            mbarrier_init(row_bar, WARPGROUPS-1);
        }
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
        for(int row = blockIdx.x; row < btcf_a.eff_row_windows; row += gridDim.x){
            int tile_num = btcf_a.rowOffset[row+1] - btcf_a.rowOffset[row];
            // 等待行窗口开始
            mbarrier_wait(row_bar, 0);
            // 内层循环：A块
            for(auto tile = 0; tile < tile_num; tile++){
                // 等待消费者 barrier
                mbarrier_wait(con_bar[stage], phase);
                // 设置expect tx
                mbarrier_expect_tx(pro_bar[stage], (WGMMA_M * WGMMA_K * (WARPGROUPS-1) + WGMMA_K * WGMMA_N) * sizeof(float));
                // load A and B
                tma_cp_async_bulk_1d(&smem.A[stage * WGMMA_N * WGMMA_K], d_data + (btcf_a.rowOffset[row] + tile) * WGMMA_N * WGMMA_K, (WGMMA_N * WGMMA_K) * sizeof(__half), pro_bar[stage]);
                // 这里其实是不对的，需要2D load 但是不知道转置之后的矩阵布局
                stage++;
                if(stage == STAGES){
                    stage = 0;
                    phase ^= 1;
                }
            }

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
        int stage = con_wgid;
        int phase = 0;
        float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if(wg_tid == 0){
            mbarrier_arrive(row_bar, 1);
            for(int i = con_wgid; i < STAGES; i += WARPGROUPS - 1){
                mbarrier_arrive(con_bar[i], 1);
            }
        }
        for(int row = blockIdx.x; row < btcf_a.eff_row_windows; row += gridDim.x){
            // 到达行窗口 
            mbarrier_arrive(row_bar, 1);
            // 计算行
            int tile_num = btcf_a.rowOffset[row+1] - btcf_a.rowOffset[row];
            for(auto tile = 0; tile < tile_num; tile++){
                // 等待生产者 barrier
                mbarrier_wait(pro_bar[stage], phase);
                // wgmma操作
                wgmma_fence();
                fence_proxy_async_shared();
                // 启动wgmma                

                wgmma_commit_group();
                wgmma_wait_group();
                // arrive at consumer barrier
                if(wg_tid == 0){
                    mbarrier_arrive(con_bar[stage], 1);
                }
                stage ++;
                if(stage >= STAGES){
                    stage -= STAGES;
                    phase ^= 1;
                }
            }
            
        }
    }

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
    
    // 开始warm up
    for(int i = 0; i < WARMUP_RUNS; i++){
        fp16_spmm_kernel<<<NUM_SMS, NUM_THREADS, sizeof(SharedStorage)>>>(d_rowOffset, d_tcA2B, d_data, d_rowIdx, d_dense_b, d_dense_c);
    }
    // 计时，正式运行
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    for(int i = 0; i < RUNS; i++){
        fp16_spmm_kernel<<<NUM_SMS, NUM_THREADS, sizeof(SharedStorage)>>>(d_rowOffset, d_tcA2B, d_data, d_rowIdx, d_dense_b, d_dense_c);
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

