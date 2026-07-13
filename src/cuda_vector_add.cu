#include "cuda_vector_add.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = a[index] + b[index];
    }
}

}  // namespace

int cuda_get_device_count(int* count) {
    if (count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    return static_cast<int>(cudaGetDeviceCount(count));
}

int cuda_vector_add_f32(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    if (a == nullptr || b == nullptr || out == nullptr || n <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    cudaError_t error = cudaSuccess;

    error = cudaMalloc(reinterpret_cast<void**>(&device_a), bytes);
    if (error != cudaSuccess) {
        goto cleanup;
    }

    error = cudaMalloc(reinterpret_cast<void**>(&device_b), bytes);
    if (error != cudaSuccess) {
        goto cleanup;
    }

    error = cudaMalloc(reinterpret_cast<void**>(&device_out), bytes);
    if (error != cudaSuccess) {
        goto cleanup;
    }

    error = cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        goto cleanup;
    }

    error = cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        goto cleanup;
    }

    constexpr int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<blocks, threads_per_block>>>(
        device_a,
        device_b,
        device_out,
        n
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        goto cleanup;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        goto cleanup;
    }

    error = cudaMemcpy(out, device_out, bytes, cudaMemcpyDeviceToHost);

cleanup:
    if (device_out != nullptr) {
        cudaFree(device_out);
    }
    if (device_b != nullptr) {
        cudaFree(device_b);
    }
    if (device_a != nullptr) {
        cudaFree(device_a);
    }

    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
