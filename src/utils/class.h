/*
* @brief: 矩阵的结构体定义
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include <vector>

/*
* @brief: COO 矩阵结构体
* @author: Haoyu Wang
* @date: 2025-10-23
*/
struct COO_MTX {
    vint rows;
    vint cols;
    vint nnzs;
    MAT_IDX_TYPE *row_idx;
    MAT_IDX_TYPE *col_idx;
    MAT_VAL_TYPE *values;
};

/*
* @brief: CSR 矩阵结构体
* @author: Haoyu Wang
* @date: 2025-10-23
*/
struct CSR_MTX {
    vint rows;
    vint cols;
    vint nnzs;
    MAT_PTR_TYPE *row_ptr;
    MAT_IDX_TYPE *col_idx;
    MAT_VAL_TYPE *values;
};

/*
* @brief: DENSE 矩阵结构体
* @author: Haoyu Wang
* @date: 2025-10-23
*/
struct DENSE_MTX {
    vint rows;
    vint cols;
    MAT_VAL_TYPE *values;
};

/*
* @brief: BTCF 矩阵结构体
* @author: Haoyu Wang
* @date: 2025-10-23
*/
struct BTCF_MTX {
    vint rows;
    vint cols;
    vint nnzs;
    vint total_row_windows; // 全部行窗口数
    vint eff_row_windows; // 有效行窗口数（包含了非零元素的）
    std::vector<MAT_PTR_TYPE> rowOffset; // 每组开始第一个块的偏移
    std::vector<MAT_PTR_TYPE> nnzOffset; // 每个块内第一个nnz的索引
    std::vector<MAT_IDX_TYPE> tcA2B; // 每个块内每一列的下标
    std::vector<MAT_MAP_TYPE> tcBit; // 每个块位图表示
    std::vector<MAT_VAL_TYPE> data; // 每个块内数据，只包含非零元素。通过nnzOffset进行索引
    std::vector<MAT_IDX_TYPE> rowIdx; // 每行的原始行位置
};

/*
* @brief: BTCF 矩阵结构体，不压缩
* @author: Haoyu Wang
* @date: 2025-10-23
*/
struct BTCF_MTX_NO_BITMAP {
    vint rows;
    vint cols;
    vint nnzs;
    vint total_row_windows; // 全部行窗口数
    vint eff_row_windows; // 有效行窗口数（包含了非零元素的）
    std::vector<MAT_PTR_TYPE> rowOffset; // 每组开始第一个块的偏移
    std::vector<MAT_IDX_TYPE> tcA2B; // 每个块内每一列的下标
    std::vector<MAT_VAL_TYPE> data; // 每个块内数据，只包含非零元素。通过nnzOffset进行索引
    std::vector<MAT_IDX_TYPE> rowIdx; // 每行的原始行位置
    u32 total_tiles; // 总块数
};


/*
* @brief: GBTCF 矩阵结构体
* @author: Haoyu Wang
* @date: 2025-10-23
*/
struct GBTCF_MTX {
    vint rows;
    vint cols;
    vint nnzs;
    vint total_row_windows; // 全部行窗口数
    vint total_groups; // 分组个数
    vint eff_row_windows; // 有效行窗口数（包含了非零元素的）
    std::vector<MAT_PTR_TYPE> groupOffset; // 每组开始第一个块的偏移
    std::vector<MAT_PTR_TYPE> nnzOffset; // 每个块内第一个nnz的索引
    std::vector<MAT_IDX_TYPE> tcA2B; // 每个块内每一列的下标
    std::vector<MAT_MAP_TYPE> tcBit; // 每个块位图表示
    std::vector<MAT_VAL_TYPE> data; // 每个块内数据，只包含非零元素。通过nnzOffset进行索引
    std::vector<MAT_IDX_TYPE> rowIdx; // 每组行索引
    std::vector<ATOMIC_TYPE> atomic; // 每组是否是原子操作
    
};
