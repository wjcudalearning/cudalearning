#include "05_template_matching.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>
#include <vector>

namespace {

__global__ void template_ssd_kernel(
    const unsigned char* image,
    const unsigned char* templ,
    float* scores,
    int image_width,
    int template_width,
    int template_height,
    int result_width,
    int result_height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= result_width || y >= result_height) {
        return;
    }

    float sum = 0.0f;
    for (int template_y = 0; template_y < template_height; ++template_y) {
        for (int template_x = 0; template_x < template_width; ++template_x) {
            const float difference =
                static_cast<float>(
                    image[(y + template_y) * image_width + x + template_x]
                ) -
                static_cast<float>(
                    templ[template_y * template_width + template_x]
                );
            sum += difference * difference;
        }
    }

    scores[y * result_width + x] =
        sum / static_cast<float>(template_width * template_height);
}

}  // namespace

int cuda_template_match_best_ssd_u8(
    const unsigned char* image,
    int image_width,
    int image_height,
    const unsigned char* templ,
    int template_width,
    int template_height,
    int* best_x,
    int* best_y,
    float* best_mean_ssd
) {
    if (
        image == nullptr || templ == nullptr ||
        best_x == nullptr || best_y == nullptr || best_mean_ssd == nullptr ||
        image_width <= 0 || image_height <= 0 ||
        template_width <= 0 || template_height <= 0 ||
        template_width > image_width || template_height > image_height
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int result_width = image_width - template_width + 1;
    const int result_height = image_height - template_height + 1;
    const std::size_t image_bytes =
        static_cast<std::size_t>(image_width) * image_height;
    const std::size_t template_bytes =
        static_cast<std::size_t>(template_width) * template_height;
    const std::size_t score_count =
        static_cast<std::size_t>(result_width) * result_height;
    const std::size_t score_bytes = score_count * sizeof(float);

    unsigned char* device_image = nullptr;
    unsigned char* device_template = nullptr;
    float* device_scores = nullptr;
    std::vector<float> scores(score_count);

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_image), image_bytes
    );
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_template), template_bytes
        );
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_scores), score_bytes
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_image, image, image_bytes, cudaMemcpyHostToDevice
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            device_template, templ, template_bytes, cudaMemcpyHostToDevice
        );
    }
    if (error == cudaSuccess) {
        const dim3 threads(16, 16);
        const dim3 blocks(
            (static_cast<unsigned int>(result_width) + threads.x - 1) / threads.x,
            (static_cast<unsigned int>(result_height) + threads.y - 1) / threads.y
        );
        template_ssd_kernel<<<blocks, threads>>>(
            device_image,
            device_template,
            device_scores,
            image_width,
            template_width,
            template_height,
            result_width,
            result_height
        );
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            scores.data(), device_scores, score_bytes, cudaMemcpyDeviceToHost
        );
    }

    if (error == cudaSuccess) {
        float best_score = FLT_MAX;
        std::size_t best_index = 0;
        for (std::size_t index = 0; index < score_count; ++index) {
            if (scores[index] < best_score) {
                best_score = scores[index];
                best_index = index;
            }
        }
        *best_x = static_cast<int>(best_index % result_width);
        *best_y = static_cast<int>(best_index / result_width);
        *best_mean_ssd = best_score;
    }

    cudaFree(device_scores);
    cudaFree(device_template);
    cudaFree(device_image);
    return static_cast<int>(error);
}

const char* cuda_error_string(int error_code) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
