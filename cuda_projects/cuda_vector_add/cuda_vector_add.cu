#include "cuda_vector_add.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count) {
        output[index] = a[index] + b[index];
    }
}

}  // namespace

int cuda_vector_add_f32(
    const float* a,
    const float* b,
    float* output,
    int count
) {
    if (
        a == nullptr ||
        b == nullptr ||
        output == nullptr ||
        count <= 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(float);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_output = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_a),
        bytes
    );

    if (error != cudaSuccess) {
        return static_cast<int>(error);
    }

    error = cudaMalloc(
        reinterpret_cast<void**>(&device_b),
        bytes
    );

    if (error != cudaSuccess) {
        cudaFree(device_a);
        return static_cast<int>(error);
    }

    error = cudaMalloc(
        reinterpret_cast<void**>(&device_output),
        bytes
    );

    if (error != cudaSuccess) {
        cudaFree(device_b);
        cudaFree(device_a);
        return static_cast<int>(error);
    }

    error = cudaMemcpy(
        device_a,
        a,
        bytes,
        cudaMemcpyHostToDevice
    );

    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_b,
            b,
            bytes,
            cudaMemcpyHostToDevice
        );
    }

    if (error == cudaSuccess) {
        constexpr int threads = 256;
        const int blocks = (count + threads - 1) / threads;

        vector_add_kernel<<<blocks, threads>>>(
            device_a,
            device_b,
            device_output,
            count
        );

        error = cudaGetLastError();
    }

    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }

    if (error == cudaSuccess) {
        error = cudaMemcpy(
            output,
            device_output,
            bytes,
            cudaMemcpyDeviceToHost
        );
    }

    cudaFree(device_output);
    cudaFree(device_b);
    cudaFree(device_a);

    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(
        static_cast<cudaError_t>(error_code)
    );
}
