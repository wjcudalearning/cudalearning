#include "00_device_info.h"

#include <cuda_runtime.h>
#include <cstring>

int cuda_get_device_count(int* count) {
    if (count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetDeviceCount(count));
}

int cuda_get_device_info(int device_index, CudaDeviceInfo* info) {
    if (device_index < 0 || info == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, device_index);
    if (error != cudaSuccess) {
        return static_cast<int>(error);
    }

    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->name, properties.name, sizeof(info->name) - 1);
    info->compute_major = properties.major;
    info->compute_minor = properties.minor;
    info->multiprocessor_count = properties.multiProcessorCount;
    info->max_threads_per_block = properties.maxThreadsPerBlock;
    info->warp_size = properties.warpSize;
    info->total_global_memory_bytes =
        static_cast<unsigned long long>(properties.totalGlobalMem);

    return static_cast<int>(cudaSuccess);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
