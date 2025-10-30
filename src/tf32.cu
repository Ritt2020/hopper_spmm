#include "common.h"
#include "utils/utils.h"
#include "ptx/wgmma.h"
#include "ptx/tma.h"
#include "ptx/barrier.h"
#include "ptx/ptx_utils.h"

// 验证函数：计算TILE_NUM个矩阵块乘法的累加结果
// tile_m: 每个tile的M维度, tile_n: 每个tile的N维度, tile_k: 每个tile的K维度
// tile_num: tile的数量
bool verify_result(float *d_C, float *h_A, float *h_B, int tile_m, int tile_n, int tile_k, int tile_num, double relative_tolerance = 0.01){
    // 从设备拷贝结果到主机
    float *h_C = (float*)malloc(tile_m * tile_n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, tile_m * tile_n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 计算主机上的正确结果
    float *h_C_ref = (float*)calloc(tile_m * tile_n, sizeof(float));
    
    // 累加TILE_NUM个矩阵块的乘积
    for(int i = 0; i < tile_num; i++){
        for(int r = 0; r < tile_m; r++){
            for(int c = 0; c < tile_n; c++){
                for(int k = 0; k < tile_k; k++){
                    h_C_ref[r * tile_n + c] += h_A[r * tile_k * tile_num + k + i * tile_k] * h_B[c * tile_k * tile_num + k + i * tile_k];
                }
            }
        }
    }
    
    
    // 计算相对误差
    double max_error = 0.0;
    double max_ref = 0.0;
    bool passed = true;
    int error_count = 0;
    const int max_error_print = 10; // 最多打印10个错误
    
    for(int i = 0; i < tile_m * tile_n; i++) {
        double error = std::abs(h_C[i] - h_C_ref[i]);
        double ref_val = std::abs(h_C_ref[i]);
        
        max_error = std::max(max_error, error);
        max_ref = std::max(max_ref, ref_val);
        
        // 如果参考值不为0，检查相对误差
        if(ref_val > 1e-6) {
            double rel_error = error / ref_val;
            if(rel_error > relative_tolerance) {
                passed = false;
                if(error_count < max_error_print) {
                    int row = i / tile_n;
                    int col = i % tile_n;
                    printf("误差过大: 位置[%d,%d], GPU结果=%.6f, CPU结果=%.6f, 相对误差=%.6f\n", 
                           row, col, h_C[i], h_C_ref[i], rel_error);
                }
                error_count++;
            }
        } else {
            // 参考值接近0时，检查绝对误差
            if(error > 1e-5) {
                passed = false;
                if(error_count < max_error_print) {
                    int row = i / tile_n;
                    int col = i % tile_n;
                    printf("误差过大: 位置[%d,%d], GPU结果=%.6f, CPU结果=%.6f, 绝对误差=%.6f\n", 
                           row, col, h_C[i], h_C_ref[i], error);
                }
                error_count++;
            }
        }
    }
    
    if(error_count > max_error_print) {
        printf("... 还有 %d 个位置存在误差（未全部显示）\n", error_count - max_error_print);
    }
    
    // 清理内存
    free(h_C);
    free(h_C_ref);
    
    return passed;
}
