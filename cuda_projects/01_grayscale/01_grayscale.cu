#include "01_grayscale.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void bgr_to_gray_kernel(
    const unsigned char* bgr,
    unsigned char* gray,
    int pixel_count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }

    const int base = index * 3;
    const float value =
        0.114f * static_cast<float>(bgr[base]) +
        0.587f * static_cast<float>(bgr[base + 1]) +
        0.299f * static_cast<float>(bgr[base + 2]);
    gray[index] = static_cast<unsigned char>(value + 0.5f);
}

}  // namespace

int cuda_bgr_to_gray_u8(
    const unsigned char* bgr,
    unsigned char* gray,
    int width,
    int height
) {
    if (bgr == nullptr || gray == nullptr || width <= 0 || height <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixel_count > static_cast<std::size_t>(0x7fffffff)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t input_bytes = pixel_count * 3;
    const std::size_t output_bytes = pixel_count;
    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_input), input_bytes
    );
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_output), output_bytes
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_input, bgr, input_bytes, cudaMemcpyHostToDevice
        );
    }
    if (error == cudaSuccess) {
        constexpr int threads = 256;
        const int count = static_cast<int>(pixel_count);
        const int blocks = (count + threads - 1) / threads;
        bgr_to_gray_kernel<<<blocks, threads>>>(device_input, device_output, count);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            gray, device_output, output_bytes, cudaMemcpyDeviceToHost
        );
    }

    cudaFree(device_output);
    cudaFree(device_input);
    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
