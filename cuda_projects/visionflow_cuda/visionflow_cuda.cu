#define VISIONFLOW_CUDA_EXPORTS
#include "visionflow_cuda.h"
#include "visionflow_cuda_internal.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int LINEAR_BLOCK = 256;

int cuda_result(cudaError_t error) {
    return visionflow_cuda::runtime_error(error);
}

int alloc_copy(
    const uint8_t* host,
    int width,
    int height,
    int stride,
    int channels,
    uint8_t** device) {
    return visionflow_cuda::allocate_and_upload(
        host, width, height, stride, channels, device);
}

int copy_back_free(
    uint8_t* host,
    int stride,
    int width,
    int height,
    int channels,
    uint8_t* device) {
    return visionflow_cuda::download_and_free(
        host, stride, width, height, channels, device);
}

void copy_text(char* output, int capacity, const char* text) {
    if (output == nullptr || capacity <= 0) return;
    const char* source = text == nullptr ? "" : text;
    std::strncpy(output, source, static_cast<std::size_t>(capacity - 1));
    output[capacity - 1] = '\0';
}

bool valid_output(
    uint8_t* dst,
    int width,
    int height,
    int stride,
    int channels) {
    return visionflow_cuda::valid_image(
        dst, width, height, stride, channels);
}

__device__ int reflect101(int value, int length) {
    if (length <= 1) return 0;
    while (value < 0 || value >= length) {
        value = value < 0 ? -value : 2 * length - value - 2;
    }
    return value;
}

__global__ void bgr_gray_kernel(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int index = (y * width + x) * 3;
    dst[y * width + x] = static_cast<uint8_t>(
        (29 * src[index] + 150 * src[index + 1] +
         77 * src[index + 2] + 128) >> 8);
}

__global__ void bgr_rgb_kernel(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int index = (y * width + x) * 3;
    dst[index] = src[index + 2];
    dst[index + 1] = src[index + 1];
    dst[index + 2] = src[index];
}

__global__ void crop_kernel(
    const uint8_t* src,
    uint8_t* dst,
    int src_width,
    int x0,
    int y0,
    int width,
    int height,
    int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int channel = 0; channel < channels; ++channel) {
        dst[(y * width + x) * channels + channel] =
            src[((y + y0) * src_width + x + x0) * channels + channel];
    }
}

__global__ void resize_gray_kernel(
    const uint8_t* src,
    uint8_t* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) return;

    if (dst_width <= src_width && dst_height <= src_height) {
        const float scale_x = static_cast<float>(src_width) / dst_width;
        const float scale_y = static_cast<float>(src_height) / dst_height;
        const float source_x0 = x * scale_x;
        const float source_x1 = (x + 1) * scale_x;
        const float source_y0 = y * scale_y;
        const float source_y1 = (y + 1) * scale_y;
        const int start_x = static_cast<int>(floorf(source_x0));
        const int end_x = static_cast<int>(ceilf(source_x1));
        const int start_y = static_cast<int>(floorf(source_y0));
        const int end_y = static_cast<int>(ceilf(source_y1));

        float sum = 0.0f;
        for (int source_y = start_y; source_y < end_y; ++source_y) {
            const float weight_y = fmaxf(
                0.0f,
                fminf(source_y1, source_y + 1.0f) -
                    fmaxf(source_y0, static_cast<float>(source_y)));
            const int clamped_y = max(0, min(src_height - 1, source_y));
            for (int source_x = start_x; source_x < end_x; ++source_x) {
                const float weight_x = fmaxf(
                    0.0f,
                    fminf(source_x1, source_x + 1.0f) -
                        fmaxf(source_x0, static_cast<float>(source_x)));
                const int clamped_x = max(0, min(src_width - 1, source_x));
                sum += src[clamped_y * src_width + clamped_x] *
                    weight_x * weight_y;
            }
        }
        dst[y * dst_width + x] = static_cast<uint8_t>(
            sum / (scale_x * scale_y) + 0.5f);
        return;
    }

    const float source_x =
        (x + 0.5f) * src_width / dst_width - 0.5f;
    const float source_y =
        (y + 0.5f) * src_height / dst_height - 0.5f;
    const int raw_x0 = static_cast<int>(floorf(source_x));
    const int raw_y0 = static_cast<int>(floorf(source_y));
    const int x0 = max(0, min(src_width - 1, raw_x0));
    const int y0 = max(0, min(src_height - 1, raw_y0));
    const int x1 = max(0, min(src_width - 1, raw_x0 + 1));
    const int y1 = max(0, min(src_height - 1, raw_y0 + 1));
    const float alpha_x = source_x - floorf(source_x);
    const float alpha_y = source_y - floorf(source_y);
    const float value =
        (1.0f - alpha_y) *
            ((1.0f - alpha_x) * src[y0 * src_width + x0] +
             alpha_x * src[y0 * src_width + x1]) +
        alpha_y *
            ((1.0f - alpha_x) * src[y1 * src_width + x0] +
             alpha_x * src[y1 * src_width + x1]);
    dst[y * dst_width + x] = static_cast<uint8_t>(value + 0.5f);
}

__global__ void gaussian_horizontal_kernel(
    const uint8_t* src,
    float* temporary,
    int width,
    int height,
    int channels,
    const float* weights,
    int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int channel = 0; channel < channels; ++channel) {
        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int source_x = reflect101(x + offset, width);
            sum += src[(y * width + source_x) * channels + channel] *
                weights[offset + radius];
        }
        temporary[(y * width + x) * channels + channel] = sum;
    }
}

__global__ void gaussian_vertical_kernel(
    const float* temporary,
    uint8_t* dst,
    int width,
    int height,
    int channels,
    const float* weights,
    int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int channel = 0; channel < channels; ++channel) {
        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int source_y = reflect101(y + offset, height);
            sum += temporary[(source_y * width + x) * channels + channel] *
                weights[offset + radius];
        }
        dst[(y * width + x) * channels + channel] =
            static_cast<uint8_t>(
                fminf(255.0f, fmaxf(0.0f, sum + 0.5f)));
    }
}

__global__ void threshold_kernel(
    const uint8_t* src,
    uint8_t* dst,
    std::size_t count,
    int threshold,
    int max_value,
    int invert) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const bool high = src[index] > threshold;
    dst[index] = static_cast<uint8_t>(
        (invert ? !high : high) ? max_value : 0);
}

__global__ void integral_rows_kernel(
    const uint8_t* src,
    std::uint64_t* integral,
    int width,
    int height) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    std::uint64_t running = 0;
    const std::size_t row_offset =
        static_cast<std::size_t>(y) * width;
    for (int x = 0; x < width; ++x) {
        running += src[row_offset + x];
        integral[row_offset + x] = running;
    }
}

__global__ void integral_columns_kernel(
    std::uint64_t* integral,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    std::uint64_t running = 0;
    for (int y = 0; y < height; ++y) {
        const std::size_t index =
            static_cast<std::size_t>(y) * width + x;
        running += integral[index];
        integral[index] = running;
    }
}

__device__ std::uint64_t integral_value(
    const std::uint64_t* integral,
    int width,
    int x,
    int y) {
    if (x < 0 || y < 0) return 0;
    return integral[static_cast<std::size_t>(y) * width + x];
}

__device__ std::uint64_t rectangle_sum(
    const std::uint64_t* integral,
    int width,
    int x0,
    int y0,
    int x1,
    int y1) {
    if (x0 > x1 || y0 > y1) return 0;
    return integral_value(integral, width, x1, y1) -
        integral_value(integral, width, x0 - 1, y1) -
        integral_value(integral, width, x1, y0 - 1) +
        integral_value(integral, width, x0 - 1, y0 - 1);
}

__global__ void adaptive_integral_kernel(
    const uint8_t* src,
    const std::uint64_t* integral,
    uint8_t* dst,
    int width,
    int height,
    int radius,
    int idelta,
    int max_value,
    int invert) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int raw_x0 = x - radius;
    const int raw_y0 = y - radius;
    const int raw_x1 = x + radius;
    const int raw_y1 = y + radius;
    const int x0 = max(0, raw_x0);
    const int y0 = max(0, raw_y0);
    const int x1 = min(width - 1, raw_x1);
    const int y1 = min(height - 1, raw_y1);

    const int left = x0 - raw_x0;
    const int right = raw_x1 - x1;
    const int top = y0 - raw_y0;
    const int bottom = raw_y1 - y1;

    std::uint64_t sum = rectangle_sum(
        integral, width, x0, y0, x1, y1);

    if (left > 0) {
        sum += static_cast<std::uint64_t>(left) *
            rectangle_sum(integral, width, 0, y0, 0, y1);
    }
    if (right > 0) {
        sum += static_cast<std::uint64_t>(right) *
            rectangle_sum(
                integral, width, width - 1, y0, width - 1, y1);
    }
    if (top > 0) {
        sum += static_cast<std::uint64_t>(top) *
            rectangle_sum(integral, width, x0, 0, x1, 0);
    }
    if (bottom > 0) {
        sum += static_cast<std::uint64_t>(bottom) *
            rectangle_sum(
                integral, width, x0, height - 1, x1, height - 1);
    }

    if (top > 0 && left > 0) {
        sum += static_cast<std::uint64_t>(top) * left * src[0];
    }
    if (top > 0 && right > 0) {
        sum += static_cast<std::uint64_t>(top) * right *
            src[width - 1];
    }
    if (bottom > 0 && left > 0) {
        sum += static_cast<std::uint64_t>(bottom) * left *
            src[static_cast<std::size_t>(height - 1) * width];
    }
    if (bottom > 0 && right > 0) {
        sum += static_cast<std::uint64_t>(bottom) * right *
            src[static_cast<std::size_t>(height) * width - 1];
    }

    const std::uint64_t block_size =
        static_cast<std::uint64_t>(radius * 2 + 1);
    const std::uint64_t area = block_size * block_size;
    const int mean = static_cast<int>((sum + area / 2) / area);
    const int difference =
        static_cast<int>(src[static_cast<std::size_t>(y) * width + x]) - mean;
    const bool selected = invert
        ? difference <= -idelta
        : difference > -idelta;
    dst[static_cast<std::size_t>(y) * width + x] =
        static_cast<uint8_t>(selected ? max_value : 0);
}

__global__ void morphology_kernel(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    int channels,
    int radius,
    int dilate) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int channel = 0; channel < channels; ++channel) {
        int value = dilate ? 0 : 255;
        for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
            for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
                const int source_x = x + offset_x;
                const int source_y = y + offset_y;
                const int sample =
                    (source_x < 0 || source_x >= width ||
                     source_y < 0 || source_y >= height)
                    ? (dilate ? 0 : 255)
                    : src[(source_y * width + source_x) * channels + channel];
                value = dilate
                    ? max(value, sample)
                    : min(value, sample);
            }
        }
        dst[(y * width + x) * channels + channel] =
            static_cast<uint8_t>(value);
    }
}

dim3 grid2d(int width, int height) {
    return dim3(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y);
}

int check_launch() {
    return cuda_result(cudaGetLastError());
}
}  // namespace

VF_CUDA_API int vf_gpu_abi_version() {
    return VF_CUDA_ABI_VERSION;
}

VF_CUDA_API int vf_gpu_device_count() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

VF_CUDA_API int vf_gpu_compute_capability() {
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, 0) == cudaSuccess
        ? properties.major * 10 + properties.minor
        : 0;
}

VF_CUDA_API int vf_gpu_device_name(char* output, int capacity) {
    if (output == nullptr || capacity <= 0) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess) return cuda_result(error);
    copy_text(output, capacity, properties.name);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_gpu_error_message(
    int error_code,
    char* output,
    int capacity) {
    if (output == nullptr || capacity <= 0) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    const char* message = "Unknown VisionFlow CUDA error";
    switch (error_code) {
        case VF_CUDA_OK:
            message = "Success";
            break;
        case VF_CUDA_INVALID_ARGUMENT:
            message = "Invalid argument";
            break;
        case VF_CUDA_ALLOCATION_FAILED:
            message = "Device allocation failed";
            break;
        case VF_CUDA_COPY_FAILED:
            message = "Host/device copy failed";
            break;
        case VF_CUDA_KERNEL_FAILED:
            message = "CUDA kernel failed";
            break;
        case VF_CUDA_DEVICE_UNAVAILABLE:
            message = "CUDA device unavailable";
            break;
        case VF_CUDA_ABI_MISMATCH:
            message = "CUDA DLL ABI mismatch";
            break;
        case VF_CUDA_INTERNAL_ERROR:
            message = "Internal CUDA DLL error";
            break;
        default:
            if (error_code >= VF_CUDA_RUNTIME_ERROR_BASE) {
                message = cudaGetErrorString(
                    static_cast<cudaError_t>(
                        error_code - VF_CUDA_RUNTIME_ERROR_BASE));
            }
            break;
    }
    copy_text(output, capacity, message);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_bgr_to_gray_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    if (src_channels != 3 || dst_channels != 1 ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(dst, width, height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, src_channels, &device_src);
    if (result != VF_CUDA_OK) return result;

    result = visionflow_cuda::allocate_bytes(
        &device_dst, static_cast<std::size_t>(width) * height);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(device_src);
        return result;
    }

    bgr_gray_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y)>>>(
        device_src, device_dst, width, height);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst, dst_stride, width, height, 1, device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_bgr_to_rgb_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    if (src_channels != 3 || dst_channels != 3 ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(dst, width, height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, src_channels, &device_src);
    if (result != VF_CUDA_OK) return result;

    result = visionflow_cuda::allocate_bytes(
        &device_dst,
        static_cast<std::size_t>(width) * height * 3);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(device_src);
        return result;
    }

    bgr_rgb_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y)>>>(
        device_src, device_dst, width, height);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst, dst_stride, width, height, 3, device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_crop_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height) {
    if (src_channels != dst_channels ||
        (src_channels != 1 && src_channels != 3) ||
        crop_x < 0 || crop_y < 0 ||
        crop_width <= 0 || crop_height <= 0 ||
        crop_x + crop_width > width ||
        crop_y + crop_height > height ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(
            dst, crop_width, crop_height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, src_channels, &device_src);
    if (result != VF_CUDA_OK) return result;

    result = visionflow_cuda::allocate_bytes(
        &device_dst,
        static_cast<std::size_t>(crop_width) * crop_height * src_channels);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(device_src);
        return result;
    }

    crop_kernel<<<
        grid2d(crop_width, crop_height), dim3(BLOCK_X, BLOCK_Y)>>>(
        device_src,
        device_dst,
        width,
        crop_x,
        crop_y,
        crop_width,
        crop_height,
        src_channels);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst,
            dst_stride,
            crop_width,
            crop_height,
            dst_channels,
            device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_resize_gray_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    int dst_width,
    int dst_height) {
    if (src_channels != 1 || dst_channels != 1 ||
        dst_width <= 0 || dst_height <= 0 ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(
            dst, dst_width, dst_height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, 1, &device_src);
    if (result != VF_CUDA_OK) return result;

    result = visionflow_cuda::allocate_bytes(
        &device_dst,
        static_cast<std::size_t>(dst_width) * dst_height);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(device_src);
        return result;
    }

    resize_gray_kernel<<<
        grid2d(dst_width, dst_height), dim3(BLOCK_X, BLOCK_Y)>>>(
        device_src,
        device_dst,
        width,
        height,
        dst_width,
        dst_height);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst,
            dst_stride,
            dst_width,
            dst_height,
            1,
            device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_gaussian_blur_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    int kernel_size) {
    if (src_channels != dst_channels ||
        (src_channels != 1 && src_channels != 3) ||
        kernel_size < 3 || kernel_size % 2 == 0 ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(
            dst, width, height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    float* device_temporary = nullptr;
    float* device_weights = nullptr;

    int result = alloc_copy(
        src, width, height, src_stride, src_channels, &device_src);
    if (result != VF_CUDA_OK) return result;

    const std::size_t sample_count =
        static_cast<std::size_t>(width) * height * src_channels;
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_temporary),
        sample_count * sizeof(float));
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_dst), sample_count);
    }
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(device_dst);
        visionflow_cuda::free_device(device_temporary);
        visionflow_cuda::free_device(device_src);
        return cuda_result(error);
    }

    const int radius = kernel_size / 2;
    std::vector<float> weights(kernel_size);
    if (kernel_size == 3) {
        weights = {0.25f, 0.5f, 0.25f};
    } else if (kernel_size == 5) {
        weights = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    } else if (kernel_size == 7) {
        weights = {
            0.03125f, 0.109375f, 0.21875f, 0.28125f,
            0.21875f, 0.109375f, 0.03125f};
    } else if (kernel_size == 9) {
        weights = {
            0.015625f, 0.05078125f, 0.1171875f, 0.19921875f,
            0.234375f,
            0.19921875f, 0.1171875f, 0.05078125f, 0.015625f};
    } else {
        const double sigma =
            0.3 * ((kernel_size - 1) * 0.5 - 1.0) + 0.8;
        float total = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            const float value = expf(
                -(offset * offset) /
                static_cast<float>(2.0 * sigma * sigma));
            weights[offset + radius] = value;
            total += value;
        }
        for (float& value : weights) value /= total;
    }

    error = cudaMalloc(
        reinterpret_cast<void**>(&device_weights),
        static_cast<std::size_t>(kernel_size) * sizeof(float));
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_weights,
            weights.data(),
            static_cast<std::size_t>(kernel_size) * sizeof(float),
            cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(device_weights);
        visionflow_cuda::free_device(device_dst);
        visionflow_cuda::free_device(device_temporary);
        visionflow_cuda::free_device(device_src);
        return cuda_result(error);
    }

    gaussian_horizontal_kernel<<<
        grid2d(width, height), dim3(BLOCK_X, BLOCK_Y)>>>(
        device_src,
        device_temporary,
        width,
        height,
        src_channels,
        device_weights,
        radius);
    result = check_launch();
    if (result == VF_CUDA_OK) {
        gaussian_vertical_kernel<<<
            grid2d(width, height), dim3(BLOCK_X, BLOCK_Y)>>>(
            device_temporary,
            device_dst,
            width,
            height,
            src_channels,
            device_weights,
            radius);
        result = visionflow_cuda::kernel_result();
    }

    visionflow_cuda::free_device(device_weights);
    visionflow_cuda::free_device(device_temporary);
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst,
            dst_stride,
            width,
            height,
            dst_channels,
            device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_threshold_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    int threshold,
    int max_value,
    int invert) {
    if (src_channels != 1 || dst_channels != 1 ||
        threshold < 0 || threshold > 255 ||
        max_value < 0 || max_value > 255 ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(dst, width, height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, 1, &device_src);
    if (result != VF_CUDA_OK) return result;

    const std::size_t count =
        static_cast<std::size_t>(width) * height;
    result = visionflow_cuda::allocate_bytes(&device_dst, count);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(device_src);
        return result;
    }

    const unsigned int blocks = static_cast<unsigned int>(
        (count + LINEAR_BLOCK - 1) / LINEAR_BLOCK);
    threshold_kernel<<<blocks, LINEAR_BLOCK>>>(
        device_src,
        device_dst,
        count,
        threshold,
        max_value,
        invert != 0);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst, dst_stride, width, height, 1, device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_adaptive_mean_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    int block_size,
    float c,
    int max_value,
    int invert) {
    if (src_channels != 1 || dst_channels != 1 ||
        block_size < 3 || block_size % 2 == 0 ||
        max_value < 0 || max_value > 255 ||
        !std::isfinite(c) ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(dst, width, height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    const std::size_t sample_count =
        static_cast<std::size_t>(width) * height;
    if (sample_count >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* device_src = nullptr;
    uint8_t* device_dst = nullptr;
    std::uint64_t* device_integral = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, 1, &device_src);
    if (result != VF_CUDA_OK) return result;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_integral),
        sample_count * sizeof(std::uint64_t));
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_dst), sample_count);
    }
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(device_dst);
        visionflow_cuda::free_device(device_integral);
        visionflow_cuda::free_device(device_src);
        return cuda_result(error);
    }

    integral_rows_kernel<<<
        (height + LINEAR_BLOCK - 1) / LINEAR_BLOCK,
        LINEAR_BLOCK>>>(
        device_src, device_integral, width, height);
    result = check_launch();

    if (result == VF_CUDA_OK) {
        integral_columns_kernel<<<
            (width + LINEAR_BLOCK - 1) / LINEAR_BLOCK,
            LINEAR_BLOCK>>>(
            device_integral, width, height);
        result = check_launch();
    }

    if (result == VF_CUDA_OK) {
        const int idelta = invert != 0
            ? static_cast<int>(floorf(c))
            : static_cast<int>(ceilf(c));
        adaptive_integral_kernel<<<
            grid2d(width, height), dim3(BLOCK_X, BLOCK_Y)>>>(
            device_src,
            device_integral,
            device_dst,
            width,
            height,
            block_size / 2,
            idelta,
            max_value,
            invert != 0);
        result = visionflow_cuda::kernel_result();
    }

    visionflow_cuda::free_device(device_integral);
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst, dst_stride, width, height, 1, device_dst);
    } else {
        visionflow_cuda::free_device(device_dst);
    }
    visionflow_cuda::free_device(device_src);
    return result;
}

VF_CUDA_API int vf_morphology_rect_u8(
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    int operation,
    int kernel_size,
    int iterations) {
    if (src_channels != dst_channels ||
        (src_channels != 1 && src_channels != 3) ||
        kernel_size < 3 || kernel_size % 2 == 0 ||
        iterations < 1 ||
        operation < VF_MORPH_OPEN || operation > VF_MORPH_ERODE ||
        !visionflow_cuda::valid_image(
            src, width, height, src_stride, src_channels) ||
        !valid_output(dst, width, height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }

    uint8_t* first = nullptr;
    uint8_t* second = nullptr;
    int result = alloc_copy(
        src, width, height, src_stride, src_channels, &first);
    if (result != VF_CUDA_OK) return result;

    result = visionflow_cuda::allocate_bytes(
        &second,
        static_cast<std::size_t>(width) * height * src_channels);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(first);
        return result;
    }

    const int radius = kernel_size / 2;
    auto pass = [&](int dilate) -> int {
        morphology_kernel<<<
            grid2d(width, height), dim3(BLOCK_X, BLOCK_Y)>>>(
            first,
            second,
            width,
            height,
            src_channels,
            radius,
            dilate);
        const int launch_result = check_launch();
        if (launch_result != VF_CUDA_OK) return launch_result;
        std::swap(first, second);
        return VF_CUDA_OK;
    };

    if (operation == VF_MORPH_OPEN) {
        for (int index = 0; index < iterations && result == VF_CUDA_OK; ++index) {
            result = pass(0);
        }
        for (int index = 0; index < iterations && result == VF_CUDA_OK; ++index) {
            result = pass(1);
        }
    } else if (operation == VF_MORPH_CLOSE) {
        for (int index = 0; index < iterations && result == VF_CUDA_OK; ++index) {
            result = pass(1);
        }
        for (int index = 0; index < iterations && result == VF_CUDA_OK; ++index) {
            result = pass(0);
        }
    } else {
        const int dilate = operation == VF_MORPH_DILATE;
        for (int index = 0; index < iterations && result == VF_CUDA_OK; ++index) {
            result = pass(dilate);
        }
    }

    if (result == VF_CUDA_OK) {
        result = cuda_result(cudaDeviceSynchronize());
    }
    if (result == VF_CUDA_OK) {
        result = copy_back_free(
            dst,
            dst_stride,
            width,
            height,
            dst_channels,
            first);
    } else {
        visionflow_cuda::free_device(first);
    }
    visionflow_cuda::free_device(second);
    return result;
}
