# Makefile for Hopper SPMM Suite
# 支持: make fp16  make bf16  make tf32，默认 make 为 fp16

NVCC = nvcc
CXX = g++

# 源码和头文件目录已变动
SRCDIR = src
UTILSDIR = src/utils

NVCC_FLAGS_BASE = -O3 -arch=sm_90a -lineinfo
NVCC_FLAGS_FP16 = $(NVCC_FLAGS_BASE) -DUSE_FP16
NVCC_FLAGS_BF16 = $(NVCC_FLAGS_BASE) -DUSE_BF16
NVCC_FLAGS_TF32 = $(NVCC_FLAGS_BASE) -DUSE_TF32

INCLUDES = -I$(SRCDIR) -I$(UTILSDIR)
CXX_FLAGS = -O3 -std=c++17

MAIN_SRC_FP16 = $(SRCDIR)/fp16.cu
MAIN_SRC_BF16 = $(SRCDIR)/bf16.cu
MAIN_SRC_TF32 = $(SRCDIR)/tf32.cu

TARGET_FP16 = fp16_test
TARGET_BF16 = bf16_test
TARGET_TF32 = tf32_test

# 默认目标: make == make fp16
fp16: $(TARGET_FP16)

bf16: $(TARGET_BF16)

tf32: $(TARGET_TF32)

$(TARGET_FP16): $(MAIN_SRC_FP16)
	$(NVCC) $(NVCC_FLAGS_FP16) $(INCLUDES) $(MAIN_SRC_FP16) -o $(TARGET_FP16)

$(TARGET_BF16): $(MAIN_SRC_BF16)
	$(NVCC) $(NVCC_FLAGS_BF16) $(INCLUDES) $(MAIN_SRC_BF16) -o $(TARGET_BF16)

$(TARGET_TF32): $(MAIN_SRC_TF32)
	$(NVCC) $(NVCC_FLAGS_TF32) $(INCLUDES) $(MAIN_SRC_TF32) -o $(TARGET_TF32)

clean:
	rm -f $(TARGET_FP16) $(TARGET_BF16) $(TARGET_TF32)
	rm -f *.o

help:
	@echo "可用的make目标:"
	@echo "  fp16      - 编译fp16版本 (默认)"
	@echo "  bf16      - 编译bf16版本"
	@echo "  tf32      - 编译tf32版本"
	@echo "  clean     - 清理文件"
	@echo "  help      - 显示此帮助信息"
	@echo ""
	@echo "用法:"
	@echo "  make             # == make fp16"
	@echo "  make fp16        # 编译fp16"
	@echo "  make bf16        # 编译bf16"
	@echo "  make tf32        # 编译tf32"
	@echo "  make clean       # 删除编译产物"

.PHONY: fp16 bf16 tf32 clean help
