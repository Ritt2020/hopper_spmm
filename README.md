# 一个高性能 Hopper SpMM算子
## 目标
实现一个基于 Warp Specialization 的 TMA + WGMMA SpMM 算子。
## 目标架构
NVIDIA H100, SM90
## 技术路线
1. 实现一个能够搬运满足 WGMMA 条件的 TMA 搬运模组。
2. 实现一个串行 TMA 搬运和 WGMMA 计算的模组。
3. 实现一个简单的 TMA 生产者、WGMMA 消费者模型。
4. 进一步优化。