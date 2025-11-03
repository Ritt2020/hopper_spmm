/*
* @brief: 转换矩阵格式
* @author: Haoyu Wang
* @date: 2025-10-23
*/
#pragma once

#include "../common.h"
#include "class.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cstdio>


/*
* @brief: CSR到BTCF转换
* @author: Haoyu Wang
* @date: 2025-10-23
*/
void CSR2BTCF(const CSR_MTX &csr, BTCF_MTX &btcf){
    // 初始化btcf
    btcf.rows = csr.rows;
    btcf.cols = csr.cols;
    btcf.nnzs = csr.nnzs;
    btcf.total_row_windows = (btcf.rows + ROW_WINDOW - 1) / ROW_WINDOW;
    btcf.rowOffset.push_back(0);
    vint eff_row_windows = 0;
    printf("开始CSR到BTCF转换，总行数: %d, 总行窗口数: %d\n", csr.rows, btcf.total_row_windows);
    vint windowId = 0;
    for(vint iter = 0; iter < csr.rows; iter += ROW_WINDOW){
        vint row_start = iter;
        vint row_end = min(row_start + ROW_WINDOW, csr.rows);
        vint block_start = csr.row_ptr[row_start];
        vint block_end = csr.row_ptr[row_end];
        vint nnzs_total = block_end - block_start;
        if(nnzs_total == 0) continue;
        eff_row_windows++;
        // 对这个行窗口所有列索引去重排序
        std::vector<MAT_IDX_TYPE> neighbor_window(csr.col_idx + block_start, csr.col_idx + block_end);
        std::sort(neighbor_window.begin(), neighbor_window.end());
        std::vector<MAT_IDX_TYPE> unique_edges;
        std::unique_copy(neighbor_window.begin(), neighbor_window.end(), std::back_inserter(unique_edges));
        // 映射
        std::unordered_map<MAT_IDX_TYPE, MAT_IDX_TYPE> clean_edges2col;
        for(vint i = 0; i < unique_edges.size(); ++i){
            clean_edges2col[unique_edges[i]] = i;
        }
        // 填充tcA2B 注意填充不足COL_WINDOW的补足为IDX_MAX
        vint remainder = unique_edges.size() % COL_WINDOW;
        if(remainder != 0){
            unique_edges.insert(unique_edges.end(), COL_WINDOW - remainder, IDX_MAX);
        }
        // 复制到tcA2B
        btcf.tcA2B.insert(btcf.tcA2B.end(), unique_edges.begin(), unique_edges.end());
        vint window_tc_num = unique_edges.size() / COL_WINDOW;
        btcf.rowOffset.push_back(btcf.rowOffset.back() + window_tc_num);
        btcf.nnzOffset.resize(btcf.nnzOffset.size() + window_tc_num, 0);
        vint tcBit_size;
        #if defined(USE_BF16) || defined(USE_FP16)
        tcBit_size = window_tc_num * 2;
        #else
        tcBit_size = window_tc_num;
        #endif
        btcf.tcBit.resize(btcf.tcBit.size() + tcBit_size, 0);
        btcf.rowIdx.push_back(iter);
        // 填充nnzOffset 和 tcBit 和 data
        #ifdef SPARSE_A_TRANSPOSE
        // 列优先存储：使用二维数组按列优先顺序存储数据
        std::vector<std::vector<std::vector<MAT_VAL_TYPE>>> data_tmp_col_major(window_tc_num);
        for(vint i = 0; i < window_tc_num; ++i){
            data_tmp_col_major[i].resize(COL_WINDOW);
        }
        #else
        // 行优先存储：使用一维数组按行优先顺序存储数据
        std::vector<std::vector<MAT_VAL_TYPE>> data_tmp(window_tc_num);
        #endif
        
        for(vint r = iter; r < std::min(iter + ROW_WINDOW, csr.rows); ++r){
            for(vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id){
                vint c_idx = clean_edges2col[csr.col_idx[nnz_id]];
                vint offset_index = btcf.rowOffset[windowId] + c_idx / COL_WINDOW;
                btcf.nnzOffset[offset_index]++;
                
                // 计算位图索引：确保在64位范围内
                vint bit_index;
                #ifdef SPARSE_A_TRANSPOSE
                // 列优先存储：按列排列位图
                bit_index = (c_idx % COL_WINDOW) * ROW_WINDOW + (r % ROW_WINDOW);
                #else
                // 行优先存储：按行排列位图
                bit_index = (r % ROW_WINDOW) * COL_WINDOW + (c_idx % COL_WINDOW);
                #endif
                
                #if defined(USE_BF16) || defined(USE_FP16)
                if(bit_index < 64) {
                    btcf.tcBit[offset_index] |= (1ULL << bit_index);
                } else if(bit_index < 128) {
                    btcf.tcBit[offset_index + 1] |= (1ULL << (bit_index - 64));
                }
                #else
                // 确保位图索引在64位范围内
                if(bit_index < 64) {
                    btcf.tcBit[offset_index] |= (1ULL << bit_index);
                }
                #endif
                
                
                #ifdef SPARSE_A_TRANSPOSE
                // 列优先存储：按列存储数据
                data_tmp_col_major[c_idx / COL_WINDOW][c_idx % COL_WINDOW].push_back(csr.values[nnz_id]);
                #else
                // 行优先存储：按行存储数据
                data_tmp[c_idx / COL_WINDOW].push_back(csr.values[nnz_id]);
                #endif
            }
        }
        
        #ifdef SPARSE_A_TRANSPOSE
        // 列优先存储：按列顺序输出数据
        for(vint i = 0; i < window_tc_num; ++i){
            for(vint col = 0; col < COL_WINDOW; ++col){
                btcf.data.insert(btcf.data.end(), data_tmp_col_major[i][col].begin(), data_tmp_col_major[i][col].end());
            }
        }
        #else
        // 行优先存储：按行顺序输出数据
        for(vint i = 0; i < window_tc_num; ++i){
            btcf.data.insert(btcf.data.end(), data_tmp[i].begin(), data_tmp[i].end());
        }
        #endif
        windowId++;
    }
    btcf.eff_row_windows = eff_row_windows;
    btcf.nnzOffset.insert(btcf.nnzOffset.begin(), 0);
    std::partial_sum(btcf.nnzOffset.begin(), btcf.nnzOffset.end(), btcf.nnzOffset.begin());
    printf("CSR到BTCF转换完成！有效行窗口数: %d\n", eff_row_windows);
}

/*
* @brief: CSR到BTCF_NO_BITMAP转换（完整存储，不使用位图压缩）
* @author: Haoyu Wang
* @date: 2025-10-23
*/
void CSR2BTCFNOBITMAP(const CSR_MTX &csr, BTCF_MTX_NO_BITMAP &btcf){
    // 初始化btcf
    btcf.rows = csr.rows;
    btcf.cols = csr.cols;
    btcf.nnzs = csr.nnzs;
    btcf.total_row_windows = (btcf.rows + ROW_WINDOW - 1) / ROW_WINDOW;
    btcf.rowOffset.push_back(0);
    vint eff_row_windows = 0;
    printf("开始CSR到BTCF_NO_BITMAP转换，总行数: %d, 总行窗口数: %d\n", csr.rows, btcf.total_row_windows);
    
    for(vint iter = 0; iter < csr.rows; iter += ROW_WINDOW){
        vint row_start = iter;
        vint row_end = min(row_start + ROW_WINDOW, csr.rows);
        vint block_start = csr.row_ptr[row_start];
        vint block_end = csr.row_ptr[row_end];
        vint nnzs_total = block_end - block_start;
        if(nnzs_total == 0) continue;
        eff_row_windows++;
        
        // 对这个行窗口所有列索引去重排序
        std::vector<MAT_IDX_TYPE> neighbor_window(csr.col_idx + block_start, csr.col_idx + block_end);
        std::sort(neighbor_window.begin(), neighbor_window.end());
        std::vector<MAT_IDX_TYPE> unique_edges;
        std::unique_copy(neighbor_window.begin(), neighbor_window.end(), std::back_inserter(unique_edges));
        
        // 映射
        std::unordered_map<MAT_IDX_TYPE, MAT_IDX_TYPE> clean_edges2col;
        for(vint i = 0; i < unique_edges.size(); ++i){
            clean_edges2col[unique_edges[i]] = i;
        }
        
        // 填充tcA2B 注意填充不足COL_WINDOW的补足为IDX_MAX
        vint remainder = unique_edges.size() % COL_WINDOW;
        if(remainder != 0){
            unique_edges.insert(unique_edges.end(), COL_WINDOW - remainder, IDX_MAX);
        }
        
        // 复制到tcA2B
        btcf.tcA2B.insert(btcf.tcA2B.end(), unique_edges.begin(), unique_edges.end());
        vint window_tc_num = unique_edges.size() / COL_WINDOW;
        btcf.rowOffset.push_back(btcf.rowOffset.back() + window_tc_num);
        btcf.rowIdx.push_back(iter);
        
        // 填充完整数据（不使用位图压缩，每个块大小为 ROW_WINDOW × COL_WINDOW）
        // 预分配当前窗口所有块的空间，初始化为0
        vint window_data_size = window_tc_num * ROW_WINDOW * COL_WINDOW;
        vint window_data_start = btcf.data.size(); // 当前窗口数据在 btcf.data 中的起始位置
        btcf.data.resize(btcf.data.size() + window_data_size, static_cast<MAT_VAL_TYPE>(0.0f));
        
        // 直接写入对应位置
        for(vint r = iter; r < std::min(iter + ROW_WINDOW, csr.rows); ++r){
            for(vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id){
                vint col_in_edges = clean_edges2col[csr.col_idx[nnz_id]];
                vint tc_id = col_in_edges / COL_WINDOW; // 该元素属于哪个块
                vint local_col = col_in_edges % COL_WINDOW; // 块内列索引
                vint local_row = r % ROW_WINDOW; // 块内行索引
                
                // 计算该元素在 btcf.data 中的位置
                vint block_start = window_data_start + tc_id * ROW_WINDOW * COL_WINDOW; // 块起始位置
                vint offset_in_block;
                
                #ifdef SPARSE_A_TRANSPOSE
                // 列优先存储：offset = local_col × ROW_WINDOW + local_row
                offset_in_block = local_col * ROW_WINDOW + local_row;
                #else
                // 行优先存储：offset = local_row × COL_WINDOW + local_col
                offset_in_block = local_row * COL_WINDOW + local_col;
                #endif
                
                btcf.data[block_start + offset_in_block] = csr.values[nnz_id];
            }
        }
    }
    
    btcf.eff_row_windows = eff_row_windows;
    btcf.total_tiles = btcf.rowOffset.back();
    printf("CSR到BTCF_NO_BITMAP转换完成！有效行窗口数: %d\n", eff_row_windows);
    printf("总块数: %d, 总数据量: %lu (每块 %d × %d = %d 个元素)\n", 
           btcf.rowOffset.back(), btcf.data.size(), ROW_WINDOW, COL_WINDOW, ROW_WINDOW * COL_WINDOW);
}

/*
* @brief: BTCF到GBTCF转换
* @author: Haoyu Wang
* @date: 2025-10-23
*/
void BTCF2GBTCF(BTCF_MTX &btcf, GBTCF_MTX &gbtcf){
    // 首先应该进行评估，确定划分
    // 收集数据：平均每行TC块数、标准差、最大值、最小值
    std::vector<vint> tc_block_num(btcf.eff_row_windows);
    for(vint i = 0; i < btcf.eff_row_windows; ++i){
        tc_block_num[i] = btcf.rowOffset[i + 1] - btcf.rowOffset[i];
    }
    double avg_tc_block_num = std::accumulate(tc_block_num.begin(), tc_block_num.end(), 0.0) / tc_block_num.size();
    double std_tc_block_num = std::sqrt(std::accumulate(tc_block_num.begin(), tc_block_num.end(), 0.0, [avg_tc_block_num](double sum, vint num) { return sum + (num - avg_tc_block_num) * (num - avg_tc_block_num); }) / tc_block_num.size());
    // 确定平均组长度
    vint min_len = 32;
    vint tgl = std::max(static_cast<vint>(avg_tc_block_num), static_cast<vint>(min_len)); // 不能少于min_len个块
    printf("tgl : %d\n", tgl);
    std::vector<MAT_PTR_TYPE> groupOffset;
    std::vector<ATOMIC_TYPE> atomic;
    std::vector<MAT_IDX_TYPE> rowIdx;
    for(vint i = 0; i < btcf.eff_row_windows; ++i){
        vint group_num = (tc_block_num[i] + tgl - 1) / tgl;
        for(vint j = 0; j < group_num; j ++){
            vint group_start_offset = btcf.rowOffset[i] + (tc_block_num[i] + group_num - 1) / group_num * j;
            groupOffset.push_back(group_start_offset);
            rowIdx.push_back(btcf.rowIdx[i]);
            if(group_num == 1)
                atomic.push_back(0);
            else
                atomic.push_back(1);
        }
    }
    // 末尾
    groupOffset.push_back(btcf.rowOffset.back());
    printf("total groups : %lu\n", groupOffset.size()-1);
    gbtcf.total_groups = groupOffset.size()-1;
    gbtcf.rows = btcf.rows;
    gbtcf.cols = btcf.cols;
    gbtcf.nnzs = btcf.nnzs;
    gbtcf.total_row_windows = groupOffset.size();
    gbtcf.eff_row_windows = groupOffset.size();
    gbtcf.groupOffset = groupOffset;
    gbtcf.atomic = atomic;
    gbtcf.rowIdx = rowIdx;
    gbtcf.nnzOffset = btcf.nnzOffset;
    gbtcf.tcA2B = btcf.tcA2B;
    gbtcf.tcBit = btcf.tcBit;
    gbtcf.data = btcf.data;
}   

/*
* @brief: 验证BTCF到GBTCF转换的正确性
* @author: Haoyu Wang
* @date: 2025-10-23
*/
bool validateBTCF2GBTCFConversion(const BTCF_MTX &btcf, const GBTCF_MTX &gbtcf) {
    printf("开始验证BTCF到GBTCF转换...\n");
    
    // 1. 验证基本属性
    if (btcf.rows != gbtcf.rows || btcf.cols != gbtcf.cols || btcf.nnzs != gbtcf.nnzs) {
        printf("错误：基本属性不匹配！\n");
        printf("  BTCF: rows=%d, cols=%d, nnzs=%d\n", btcf.rows, btcf.cols, btcf.nnzs);
        printf("  GBTCF: rows=%d, cols=%d, nnzs=%d\n", gbtcf.rows, gbtcf.cols, gbtcf.nnzs);
        return false;
    }
    
    // 2. 验证总组数计算
    if (gbtcf.total_groups != gbtcf.groupOffset.size() - 1) {
        printf("错误：总组数计算不正确！\n");
        printf("  total_groups=%d, groupOffset.size()=%lu\n", gbtcf.total_groups, gbtcf.groupOffset.size());
        return false;
    }
    
    // 3. 验证groupOffset的连续性
    for (vint i = 0; i < gbtcf.groupOffset.size() - 1; ++i) {
        if (gbtcf.groupOffset[i] >= gbtcf.groupOffset[i + 1]) {
            printf("错误：groupOffset不连续！groupOffset[%d]=%d >= groupOffset[%d]=%d\n", 
                   i, gbtcf.groupOffset[i], i+1, gbtcf.groupOffset[i+1]);
            return false;
        }
    }
    
    // 4. 验证groupOffset与原始rowOffset的对应关系
    vint group_idx = 0;
    for (vint window_id = 0; window_id < btcf.total_row_windows; ++window_id) {
        vint window_tc_blocks = btcf.rowOffset[window_id + 1] - btcf.rowOffset[window_id];
        if (window_tc_blocks == 0) continue; // 跳过空窗口
        
        // 计算该窗口应该有多少组
        vint expected_groups = std::min(static_cast<vint>(4), 
            static_cast<vint>((window_tc_blocks + 3) / 4)); // tgl=4
        
        // 验证该窗口的组数
        vint actual_groups = 0;
        vint start_group = group_idx;
        while (group_idx < gbtcf.groupOffset.size() - 1 && 
               gbtcf.groupOffset[group_idx] < btcf.rowOffset[window_id + 1]) {
            group_idx++;
            actual_groups++;
        }
        
        if (actual_groups != expected_groups) {
            printf("错误：窗口%d的组数不匹配！期望=%d，实际=%d\n", 
                   window_id, expected_groups, actual_groups);
            return false;
        }
        
        // 验证该窗口的组偏移范围
        if (start_group < gbtcf.groupOffset.size() - 1) {
            if (gbtcf.groupOffset[start_group] != btcf.rowOffset[window_id]) {
                printf("错误：窗口%d的起始组偏移不匹配！\n", window_id);
                return false;
            }
        }
    }
    
    // 5. 验证atomic数组
    if (gbtcf.atomic.size() != gbtcf.total_groups) {
        printf("错误：atomic数组大小不匹配！\n");
        return false;
    }
    
    // 6. 验证rowIdx数组
    if (gbtcf.rowIdx.size() != gbtcf.total_groups) {
        printf("错误：rowIdx数组大小不匹配！\n");
        return false;
    }
    
    // 7. 验证数据完整性
    if (gbtcf.nnzOffset.size() != btcf.nnzOffset.size() ||
        gbtcf.tcA2B.size() != btcf.tcA2B.size() ||
        gbtcf.tcBit.size() != btcf.tcBit.size() ||
        gbtcf.data.size() != btcf.data.size()) {
        printf("错误：数据数组大小不匹配！\n");
        printf("  nnzOffset: BTCF=%lu, GBTCF=%lu\n", btcf.nnzOffset.size(), gbtcf.nnzOffset.size());
        printf("  tcA2B: BTCF=%lu, GBTCF=%lu\n", btcf.tcA2B.size(), gbtcf.tcA2B.size());
        printf("  tcBit: BTCF=%lu, GBTCF=%lu\n", btcf.tcBit.size(), gbtcf.tcBit.size());
        printf("  data: BTCF=%lu, GBTCF=%lu\n", btcf.data.size(), gbtcf.data.size());
        return false;
    }
    
    // 8. 验证数据内容一致性
    for (vint i = 0; i < btcf.nnzOffset.size(); ++i) {
        if (btcf.nnzOffset[i] != gbtcf.nnzOffset[i]) {
            printf("错误：nnzOffset[%d]不匹配！BTCF=%d, GBTCF=%d\n", 
                   i, btcf.nnzOffset[i], gbtcf.nnzOffset[i]);
            return false;
        }
    }
    
    for (vint i = 0; i < btcf.tcA2B.size(); ++i) {
        if (btcf.tcA2B[i] != gbtcf.tcA2B[i]) {
            printf("错误：tcA2B[%d]不匹配！BTCF=%d, GBTCF=%d\n", 
                   i, btcf.tcA2B[i], gbtcf.tcA2B[i]);
            return false;
        }
    }
    
    for (vint i = 0; i < btcf.tcBit.size(); ++i) {
        if (btcf.tcBit[i] != gbtcf.tcBit[i]) {
            printf("错误：tcBit[%d]不匹配！\n", 
                   i);
            return false;
        }
    }
    
    for (vint i = 0; i < btcf.data.size(); ++i) {
        if (btcf.data[i] != gbtcf.data[i]) {
            printf("错误：data[%d]不匹配！\n", 
                   i);
            return false;
        }
    }
    return true;
}
