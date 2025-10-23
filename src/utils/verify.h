/*
* @brief: 验证矩阵
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include "class.h"
#include <math.h>
#include <stdio.h>

/*
* @brief: 验证DENSE矩阵
* @author: Haoyu Wang
* @date: 2025-10-23
*/
bool verify_dense_matrix(DENSE_MTX *A, DENSE_MTX *B) {
    vint total_elements = A->rows * A->cols;
    vint failed_elements = 0;
    vint print_total = 0;
    
    for (vint i = 0; i < A->rows; i++) {
        for (vint j = 0; j < A->cols; j++) {
            // 绝对值小于 1e-1 的误差视为相等
            if (fabs(A->values[i * A->cols + j] - B->values[i * B->cols + j]) > 1) {
                failed_elements++;
                if(print_total ++ <= 20)
                    printf("验证失败: A[%d][%d] = %f, B[%d][%d] = %f\n", i, j, A->values[i * A->cols + j], i, j, B->values[i * B->cols + j]);
            }
        }
    }
    
    if (failed_elements > 0) {
        double failure_rate = (double)failed_elements / total_elements * 100.0;
        printf("验证统计: 失败元素个数 = %d, 总元素个数 = %d, 失败占比 = %.2f%%\n", 
               failed_elements, total_elements, failure_rate);
        return false;
    }
    
    return true;
}