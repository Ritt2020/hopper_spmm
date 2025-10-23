/*
* @brief: 加载矩阵
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include "class.h"

/*
* @brief: 全局变量用于排序比较函数
* @author: Haoyu Wang
* @date: 2025-10-23
*/
COO_MTX *global_coo_mtx = NULL;

/*
* @brief: 比较函数用于排序COO格式数据
* @author: Haoyu Wang
* @date: 2025-10-23
*/
int compare_coo_indices(const void *a, const void *b) {
    vint idx_a = *(vint *)a;
    vint idx_b = *(vint *)b;
    
    if (global_coo_mtx->row_idx[idx_a] != global_coo_mtx->row_idx[idx_b]) {
        return global_coo_mtx->row_idx[idx_a] - global_coo_mtx->row_idx[idx_b];  // 按行排序
    }
    return global_coo_mtx->col_idx[idx_a] - global_coo_mtx->col_idx[idx_b];      // 按列排序
}

/*
* @brief: 加载矩阵并返回CSR格式，可选择是否进行验证
* @author: Haoyu Wang
* @date: 2025-10-23
*/
CSR_MTX load_matrix(const char *filename, int enable_validation) {
    COO_MTX coo_mtx;
    FILE *file;
    char line[1024];
    vint rows, cols, nnzs;
    vint coo_count = 0;

    CSR_MTX mtx;

    mtx.rows = 0;
    mtx.cols = 0;
    mtx.nnzs = 0;
    mtx.row_ptr = NULL;
    mtx.col_idx = NULL;
    mtx.values = NULL;
    
    coo_mtx.rows = 0;
    coo_mtx.cols = 0;
    coo_mtx.nnzs = 0;
    coo_mtx.row_idx = NULL;
    coo_mtx.col_idx = NULL;
    coo_mtx.values = NULL;
    
    // 打开文件
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("错误：无法打开文件 %s\n", filename);
        return mtx;
    }
    
    // 跳过注释行
    while (fgets(line, sizeof(line), file)) {
        if (line[0] != '%') {
            break;
        }
    }
    
    // 读取矩阵信息
    if (sscanf(line, "%u %u %u", &rows, &cols, &nnzs) != 3) {
        printf("错误：无法解析矩阵信息行\n");
        fclose(file);
        return mtx;
    }
    
    // 分配COO数据数组
    coo_mtx.rows = rows;
    coo_mtx.cols = cols;
    coo_mtx.nnzs = nnzs;
    coo_mtx.row_idx = (MAT_IDX_TYPE *)malloc(nnzs * sizeof(MAT_IDX_TYPE));
    coo_mtx.col_idx = (MAT_IDX_TYPE *)malloc(nnzs * sizeof(MAT_IDX_TYPE));
    coo_mtx.values = (MAT_VAL_TYPE *)malloc(nnzs * sizeof(MAT_VAL_TYPE));
    
    if (coo_mtx.row_idx == NULL || coo_mtx.col_idx == NULL || coo_mtx.values == NULL) {
        printf("错误：COO 主机内存分配失败\n");
        fclose(file);
        // 清理已分配的内存
        if (coo_mtx.row_idx) free(coo_mtx.row_idx);
        if (coo_mtx.col_idx) free(coo_mtx.col_idx);
        if (coo_mtx.values) free(coo_mtx.values);
        return mtx;
    }
    
    // 读取数据行
    while (fgets(line, sizeof(line), file) && coo_count < nnzs) {
        vint row, col;
        MAT_VAL_TYPE val;
        
        if (sscanf(line, "%u %u %f", &row, &col, &val) == 3) {
            // MTX格式通常使用1-based索引，转换为0-based
            coo_mtx.row_idx[coo_count] = row - 1;
            coo_mtx.col_idx[coo_count] = col - 1;
            coo_mtx.values[coo_count] = val;
            coo_count++;
        }
    }
    
    fclose(file);
    
    if (coo_count != nnzs) {
        printf("警告：读取的非零元素数量 (%u) 与声明的不符 (%u)\n", coo_count, nnzs);
        nnzs = coo_count;
        coo_mtx.nnzs = nnzs;
    }
    
    // 创建索引数组用于排序
    vint *sort_indices = (vint *)malloc(nnzs * sizeof(vint));
    if (sort_indices == NULL) {
        printf("错误：排序索引内存分配失败\n");
        // 清理已分配的内存
        if (coo_mtx.row_idx) free(coo_mtx.row_idx);
        if (coo_mtx.col_idx) free(coo_mtx.col_idx);
        if (coo_mtx.values) free(coo_mtx.values);
        return mtx;
    }
    
    for (vint i = 0; i < nnzs; i++) {
        sort_indices[i] = i;
    }
    
    // 设置全局变量并排序索引
    global_coo_mtx = &coo_mtx;
    qsort(sort_indices, nnzs, sizeof(vint), compare_coo_indices);
    
    // 分配CSR格式的内存
    mtx.rows = rows;
    mtx.cols = cols;
    mtx.nnzs = nnzs;
    mtx.row_ptr = (MAT_PTR_TYPE *)calloc(rows + 1, sizeof(MAT_PTR_TYPE));
    mtx.col_idx = (MAT_IDX_TYPE *)malloc(nnzs * sizeof(MAT_IDX_TYPE));
    mtx.values = (MAT_VAL_TYPE *)malloc(nnzs * sizeof(MAT_VAL_TYPE));
    
    if (mtx.row_ptr == NULL || mtx.col_idx == NULL || mtx.values == NULL) {
        printf("错误：CSR内存分配失败\n");
        // coo_data已不存在，无需释放
        if (mtx.row_ptr) free(mtx.row_ptr);
        if (mtx.col_idx) free(mtx.col_idx);
        if (mtx.values) free(mtx.values);
        mtx.rows = mtx.cols = mtx.nnzs = 0;
        mtx.row_ptr = NULL;
        mtx.col_idx = NULL;
        mtx.values = NULL;
        return mtx;
    }
    
    // 转换为CSR格式
    vint current_row = 0;
    mtx.row_ptr[0] = 0;
    
    for (vint i = 0; i < nnzs; i++) {
        vint sorted_idx = sort_indices[i];
        // 填充当前行的数据
        mtx.col_idx[i] = coo_mtx.col_idx[sorted_idx];
        mtx.values[i] = coo_mtx.values[sorted_idx];
        
        // 更新row_ptr
        while (current_row < coo_mtx.row_idx[sorted_idx]) {
            current_row++;
            mtx.row_ptr[current_row] = i;
        }
    }
    
    // 填充剩余的行指针
    for (vint i = current_row + 1; i <= rows; i++) {
        mtx.row_ptr[i] = nnzs;
    }
    
    // 如果启用验证，进行CSR格式验证
    if (enable_validation) {
        printf("\n=== 开始CSR格式验证 ===\n");
        int validation_result = validate_csr_from_coo(&mtx, &coo_mtx);
        if (validation_result) {
            printf("验证通过：CSR格式转换正确\n");
        } else {
            printf("验证失败：CSR格式转换存在问题\n");
        }
        printf("=== 验证完成 ===\n\n");
    }
    
    // 释放COO数据和排序索引
    free(coo_mtx.row_idx);
    free(coo_mtx.col_idx);
    free(coo_mtx.values);
    free(sort_indices);
    
    printf("成功加载矩阵：%u x %u，%u 个非零元素\n", rows, cols, nnzs);
    return mtx;
}

/*
* @brief: 向后兼容的函数，默认不进行验证
* @author: Haoyu Wang
* @date: 2025-10-23
*/
CSR_MTX load_matrix(const char *filename) {
    return load_matrix(filename, 0);
}

/*
* @brief: 释放CSR_MTX结构体内存
* @author: Haoyu Wang
* @date: 2025-10-23
*/
void free_csr_matrix(CSR_MTX *mtx) {
    if (mtx != NULL) {
        if (mtx->row_ptr != NULL) {
            free(mtx->row_ptr);
            mtx->row_ptr = NULL;
        }
        if (mtx->col_idx != NULL) {
            free(mtx->col_idx);
            mtx->col_idx = NULL;
        }
        if (mtx->values != NULL) {
            free(mtx->values);
            mtx->values = NULL;
        }
        mtx->rows = 0;
        mtx->cols = 0;
        mtx->nnzs = 0;
    }
}

/*
* @brief: 验证CSR格式是否正确转换自COO格式
* @author: Haoyu Wang
* @date: 2025-10-23
*/
int validate_csr_from_coo(const CSR_MTX *csr, const COO_MTX *coo) {
    if (csr == NULL || coo == NULL) {
        printf("错误：CSR或COO矩阵指针为空\n");
        return 0;
    }
    
    if (csr->rows != coo->rows || csr->cols != coo->cols || csr->nnzs != coo->nnzs) {
        printf("错误：CSR和COO矩阵维度不匹配\n");
        printf("CSR: %u x %u, %u 个非零元素\n", csr->rows, csr->cols, csr->nnzs);
        printf("COO: %u x %u, %u 个非零元素\n", coo->rows, coo->cols, coo->nnzs);
        return 0;
    }
    
    printf("开始验证CSR格式转换的正确性...\n");
    printf("矩阵大小: %u x %u, 非零元素: %u\n", csr->rows, csr->cols, csr->nnzs);
    
    // 使用更高效的验证方法：检查CSR的row_ptr是否正确
    int validation_errors = 0;
    
    // 1. 验证row_ptr数组的正确性
    printf("验证row_ptr数组...\n");
    if (csr->row_ptr[0] != 0) {
        printf("错误：row_ptr[0] 应该为0，实际为 %u\n", csr->row_ptr[0]);
        validation_errors++;
    }
    
    if (csr->row_ptr[csr->rows] != csr->nnzs) {
        printf("错误：row_ptr[%u] 应该为 %u，实际为 %u\n", 
               csr->rows, csr->nnzs, csr->row_ptr[csr->rows]);
        validation_errors++;
    }
    
    // 检查row_ptr是否单调递增
    for (vint i = 0; i < csr->rows; i++) {
        if (csr->row_ptr[i] > csr->row_ptr[i + 1]) {
            printf("错误：row_ptr[%u] = %u > row_ptr[%u] = %u\n", 
                   i, csr->row_ptr[i], i + 1, csr->row_ptr[i + 1]);
            validation_errors++;
        }
    }
    
    // 2. 验证每行的列索引是否在有效范围内
    printf("验证列索引范围...\n");
    for (vint row = 0; row < csr->rows; row++) {
        for (vint idx = csr->row_ptr[row]; idx < csr->row_ptr[row + 1]; idx++) {
            if (csr->col_idx[idx] >= csr->cols) {
                printf("错误：行 %u 的列索引 %u 超出范围 [0, %u)\n", 
                       row, csr->col_idx[idx], csr->cols);
                validation_errors++;
            }
        }
    }
    
    // 3. 抽样验证：只验证前1000个元素（对于大矩阵）
    printf("抽样验证元素匹配...\n");
    vint sample_size = (csr->nnzs > 1000) ? 1000 : csr->nnzs;
    vint step = csr->nnzs / sample_size;
    int sample_errors = 0;
    
    for (vint i = 0; i < sample_size; i++) {
        vint csr_idx = i * step;
        if (csr_idx >= csr->nnzs) break;
        
        // 找到CSR元素对应的行列
        vint row = 0;
        while (row < csr->rows && csr->row_ptr[row + 1] <= csr_idx) {
            row++;
        }
        vint col = csr->col_idx[csr_idx];
        MAT_VAL_TYPE val = csr->values[csr_idx];
        
        // 在COO中查找对应元素
        int found = 0;
        for (vint coo_idx = 0; coo_idx < coo->nnzs; coo_idx++) {
            if (coo->row_idx[coo_idx] == row && coo->col_idx[coo_idx] == col) {
                if (coo->values[coo_idx] == val) {
                    found = 1;
                    break;
                } else {
                    printf("错误：位置(%u, %u)的值不匹配 - CSR: %f, COO: %f\n", 
                           row, col, val, coo->values[coo_idx]);
                    sample_errors++;
                    found = 1;
                    break;
                }
            }
        }
        
        if (!found) {
            printf("错误：CSR中的元素(%u, %u) = %f 在COO中未找到\n", row, col, val);
            sample_errors++;
        }
    }
    
    validation_errors += sample_errors;
    
    if (validation_errors == 0) {
        printf("验证成功：CSR格式转换正确！\n");
        printf("抽样验证了 %u 个元素，全部匹配\n", sample_size);
        return 1;
    } else {
        printf("验证失败：发现 %d 个错误\n", validation_errors);
        if (sample_errors > 0) {
            printf("其中抽样验证发现 %d 个错误\n", sample_errors);
        }
        return 0;
    }
}

/*
* @brief: 释放COO_MTX结构体内存
* @author: Haoyu Wang
* @date: 2025-10-23
*/
void free_coo_matrix(COO_MTX *mtx) {
    if (mtx != NULL) {
        if (mtx->row_idx != NULL) {
            free(mtx->row_idx);
            mtx->row_idx = NULL;
        }
        if (mtx->col_idx != NULL) {
            free(mtx->col_idx);
            mtx->col_idx = NULL;
        }
        if (mtx->values != NULL) {
            free(mtx->values);
            mtx->values = NULL;
        }
        mtx->rows = 0;
        mtx->cols = 0;
        mtx->nnzs = 0;
    }
}
