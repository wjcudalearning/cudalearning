#include "02_threshold.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void threshold_kernel(
    const unsigned char* input,
    unsigned char* output,
    int count,
    unsigned char threshold,
    int invert
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    bool white = input[index] > threshold;
    if (invert != 0) {
        white = !white;
    }
    output[index] = white ? 255 : 0;
}

}  // namespace

int cuda_threshold_u8(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    unsigned char threshold,
    int invert
) {
    if (input == nullptr || output == nullptr || width <= 0 || height <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t count_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (count_size > static_cast<std::size_t>(0x7fffffff)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t bytes = count_size;
    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_input), bytes
    );
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_output), bytes
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_input, input, bytes, cudaMemcpyHostToDevice
        );
    }
    if (error == cudaSuccess) {
        constexpr int threads = 256;
        const int count = static_cast<int>(count_size);
        const int blocks = (count + threads - 1) / threads;
        threshold_kernel<<<blocks, threads>>>(
            device_input, device_output, count, threshold, invert
        );
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            output, device_output, bytes, cudaMemcpyDeviceToHost
        );
    }

    cudaFree(device_output);
    cudaFree(device_input);
    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
