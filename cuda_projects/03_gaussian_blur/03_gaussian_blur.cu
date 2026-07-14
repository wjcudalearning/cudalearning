#include "03_gaussian_blur.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__global__ void gaussian_blur_3x3_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int weights[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1},
    };

    int sum = 0;
    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int source_x = clamp_int(x + kernel_x, 0, width - 1);
            const int source_y = clamp_int(y + kernel_y, 0, height - 1);
            sum += static_cast<int>(input[source_y * width + source_x]) *
                   weights[kernel_y + 1][kernel_x + 1];
        }
    }

    output[y * width + x] = static_cast<unsigned char>((sum + 8) / 16);
}

}  // namespace

int cuda_gaussian_blur_3x3_u8(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    if (input == nullptr || output == nullptr || width <= 0 || height <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_input), pixel_count
    );
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_output), pixel_count
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_input, input, pixel_count, cudaMemcpyHostToDevice
        );
    }
    if (error == cudaSuccess) {
        const dim3 threads(16, 16);
        const dim3 blocks(
            (static_cast<unsigned int>(width) + threads.x - 1) / threads.x,
            (static_cast<unsigned int>(height) + threads.y - 1) / threads.y
        );
        gaussian_blur_3x3_kernel<<<blocks, threads>>>(
            device_input, device_output, width, height
        );
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            output, device_output, pixel_count, cudaMemcpyDeviceToHost
        );
    }

    cudaFree(device_output);
    cudaFree(device_input);
    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
