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
  
