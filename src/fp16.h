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

#define WGMMA_M 64
#define WGMMA_N 8
#define WGMMA_K 16




