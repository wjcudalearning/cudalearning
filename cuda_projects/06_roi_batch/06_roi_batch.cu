#include "06_roi_batch.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void roi_batch_threshold_kernel(
    const unsigned char* rois,
    unsigned char* output,
    int roi_pixels,
    int roi_count,
    unsigned char threshold
) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int roi = blockIdx.y;
    if (roi >= roi_count || pixel >= roi_pixels) {
        return;
    }

    const int index = roi * roi_pixels + pixel;
    output[index] = rois[index] > threshold ? 255 : 0;
}

}  // namespace

int cuda_roi_batch_threshold_u8(
    const unsigned char* rois,
    unsigned char* output,
    int roi_width,
    int roi_height,
    int roi_count,
    unsigned char threshold
) {
    if (
        rois == nullptr || output == nullptr ||
        roi_width <= 0 || roi_height <= 0 || roi_count <= 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const std::size_t roi_pixels_size =
        static_cast<std::size_t>(roi_width) * roi_height;
    const std::size_t total_pixels = roi_pixels_size * roi_count;
    if (roi_pixels_size > static_cast<std::size_t>(0x7fffffff)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_input), total_pixels
    );
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_output), total_pixels
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_input, rois, total_pixels, cudaMemcpyHostToDevice
        );
    }
    if (error == cudaSuccess) {
        constexpr int threads = 256;
        const int roi_pixels = static_cast<int>(roi_pixels_size);
        const dim3 blocks(
            (roi_pixels + threads - 1) / threads,
            static_cast<unsigned int>(roi_count)
        );
        roi_batch_threshold_kernel<<<blocks, threads>>>(
            device_input,
            device_output,
            roi_pixels,
            roi_count,
            threshold
        );
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            output, device_output, total_pixels, cudaMemcpyDeviceToHost
        );
    }

    cudaFree(device_output);
    cudaFree(device_input);
    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
