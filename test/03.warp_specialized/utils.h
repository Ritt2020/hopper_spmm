#pragma once

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void checkCudaErrors(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
      fprintf(
          stderr,
          "CUDA error at %s:%d: %s\n",
          file,
          line,
          cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }
  }
  
  #define CHECK_CUDA(err) checkCudaErrors(err, __FILE__, __LINE__)

inline void fill_matrix_random(float* ptr, int n) {
    for(int i = 0; i < n; ++i) {
        ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

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
    // 对于每个tile: C += A_tile * B_tile
    // A_tile: tile_m x tile_k (行主序)
    // B_tile: tile_k x tile_n (列主序)
    for(int tile_idx = 0; tile_idx < tile_num; tile_idx++) {
        // 获取当前tile在A和B中的起始位置
        float *A_tile = h_A + tile_idx * tile_m * tile_k;
        float *B_tile = h_B + tile_idx * tile_k * tile_n;
        
        // 计算当前tile的矩阵乘法并累加到C
        for(int i = 0; i < tile_m; i++) {
            for(int j = 0; j < tile_n; j++) {
                for(int k = 0; k < tile_k; k++) {
                    // A_tile[i][k] = A_tile[i * tile_k + k] (行主序)
                    // B_tile[k][j] = B_tile[j * tile_k + k] (列主序)
                    h_C_ref[i * tile_n + j] += A_tile[i * tile_k + k] * B_tile[j * tile_k + k];
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
    
    // 输出统计信息
    printf("\n=== 验证统计 ===\n");
    printf("验证结果: %s\n", passed ? "✓ 通过" : "✗ 失败");
    printf("矩阵维度: C[%d,%d] = sum(A_i[%d,%d] * B_i[%d,%d]) for i=0..%d\n", 
           tile_m, tile_n, tile_m, tile_k, tile_k, tile_n, tile_num-1);
    printf("最大绝对误差: %.6e\n", max_error);
    printf("最大参考值: %.6e\n", max_ref);
    if(max_ref > 1e-6) {
        printf("最大相对误差: %.6e (%.4f%%)\n", max_error / max_ref, (max_error / max_ref) * 100);
    }
    printf("相对误差阈值: %.6e (%.4f%%)\n", relative_tolerance, relative_tolerance * 100);
    printf("错误数量: %d / %d\n", error_count, tile_m * tile_n);
    
    // 清理内存
    free(h_C);
    free(h_C_ref);
    
    return passed;
}
