#include "cuda_pattern_match.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace {

constexpr int kSuccess = 0;
constexpr int kInvalidArgument = -1;
constexpr int kTemplateLargerThanImage = -2;
constexpr int kFlatTemplate = -3;
constexpr int kHostAllocationFailure = -4;
constexpr int kNoCudaDevice = -5;
constexpr int kDeviceAllocationTooLarge = -6;

constexpr int kMinCoarseTemplateDimension = 8;
constexpr int kMaximumPyramidFactor = 32;
constexpr long long kTargetRefinePixelsPerLaunch = 600000000LL;
constexpr long long kMinimumRefineTasksPerLaunch = 4096LL;

struct DeviceCandidate {
    int x;
    int y;
    float score;
};

struct HostTemplateLevel {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
    std::vector<float> centered;
    float energy = 0.0f;
};

struct DeviceLevel {
    int image_width = 0;
    int image_height = 0;
    int template_width = 0;
    int template_height = 0;
    unsigned char* image = nullptr;
    float* centered_template = nullptr;
    float template_energy = 0.0f;
};

__global__ void downsample2_u8_kernel(
    const unsigned char* source,
    int source_width,
    int source_height,
    unsigned char* destination,
    int destination_width,
    int destination_height
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= destination_width || y >= destination_height) {
        return;
    }

    const int source_x = x * 2;
    const int source_y = y * 2;

    unsigned int sum = 0;
    unsigned int count = 0;

    for (int dy = 0; dy < 2; ++dy) {
        const int yy = source_y + dy;
        if (yy >= source_height) {
            continue;
        }
        for (int dx = 0; dx < 2; ++dx) {
            const int xx = source_x + dx;
            if (xx >= source_width) {
                continue;
            }
            sum += source[static_cast<std::size_t>(yy) * source_width + xx];
            ++count;
        }
    }

    destination[static_cast<std::size_t>(y) * destination_width + x] =
        static_cast<unsigned char>((sum + count / 2U) / count);
}

__global__ void coarse_zncc_score_kernel(
    const unsigned char* image,
    int image_width,
    const float* centered_template,
    int template_width,
    int template_height,
    float template_energy,
    int positions_x,
    int positions_y,
    float* scores
) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long position_count =
        static_cast<long long>(positions_x) * positions_y;

    if (index >= position_count) {
        return;
    }

    const int origin_x = static_cast<int>(index % positions_x);
    const int origin_y = static_cast<int>(index / positions_x);

    float image_sum = 0.0f;
    float image_square_sum = 0.0f;
    float numerator = 0.0f;

    for (int template_y = 0; template_y < template_height; ++template_y) {
        const unsigned char* image_row =
            image + static_cast<std::size_t>(origin_y + template_y) *
                        image_width +
            origin_x;
        const float* template_row =
            centered_template +
            static_cast<std::size_t>(template_y) * template_width;

        for (int template_x = 0; template_x < template_width; ++template_x) {
            const float image_value = static_cast<float>(image_row[template_x]);
            image_sum += image_value;
            image_square_sum = fmaf(image_value, image_value, image_square_sum);
            numerator = fmaf(image_value, template_row[template_x], numerator);
        }
    }

    const float sample_count =
        static_cast<float>(template_width * template_height);
    const float image_energy =
        image_square_sum - (image_sum * image_sum) / sample_count;

    float score = -1.0f;
    if (image_energy > 1.0e-5f && template_energy > 1.0e-9f) {
        const float denominator = sqrtf(image_energy * template_energy);
        if (denominator > 1.0e-12f) {
            score = numerator / denominator;
            score = fminf(1.0f, fmaxf(-1.0f, score));
        }
    }

    scores[index] = score;
}

__global__ void local_maxima_kernel(
    const float* scores,
    int width,
    int height,
    float threshold,
    DeviceCandidate* candidates,
    int max_candidates,
    int* candidate_count
) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(width) * height;

    if (index >= count) {
        return;
    }

    const float score = scores[index];
    if (score < threshold) {
        return;
    }

    const int x = static_cast<int>(index % width);
    const int y = static_cast<int>(index / width);

    for (int dy = -1; dy <= 1; ++dy) {
        const int neighbor_y = y + dy;
        if (neighbor_y < 0 || neighbor_y >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int neighbor_x = x + dx;
            if ((dx == 0 && dy == 0) || neighbor_x < 0 || neighbor_x >= width) {
                continue;
            }

            const long long neighbor_index =
                static_cast<long long>(neighbor_y) * width + neighbor_x;
            const float neighbor_score = scores[neighbor_index];

            // Tie break by index so a flat plateau emits only one point.
            if (
                neighbor_score > score ||
                (neighbor_score == score && neighbor_index < index)
            ) {
                return;
            }
        }
    }

    const int output_index = atomicAdd(candidate_count, 1);
    if (output_index < max_candidates) {
        candidates[output_index] = DeviceCandidate{x, y, score};
    }
}

__global__ void refine_score_blocks_kernel(
    const unsigned char* image,
    int image_width,
    int image_height,
    const float* centered_template,
    int template_width,
    int template_height,
    float template_energy,
    const DeviceCandidate* coarse_candidates,
    int candidate_count,
    int refine_radius,
    long long task_offset,
    long long task_count,
    float* refinement_scores
) {
    const long long local_task = blockIdx.x;
    if (local_task >= task_count) {
        return;
    }

    const long long task = task_offset + local_task;
    const int side = refine_radius * 2 + 1;
    const int positions_per_candidate = side * side;
    const int candidate_index =
        static_cast<int>(task / positions_per_candidate);
    const int offset_index =
        static_cast<int>(task % positions_per_candidate);

    if (candidate_index >= candidate_count) {
        return;
    }

    const int offset_x = offset_index % side - refine_radius;
    const int offset_y = offset_index / side - refine_radius;
    const DeviceCandidate source = coarse_candidates[candidate_index];

    const int origin_x = source.x * 2 + offset_x;
    const int origin_y = source.y * 2 + offset_y;
    const int positions_x = image_width - template_width + 1;
    const int positions_y = image_height - template_height + 1;

    if (
        origin_x < 0 || origin_y < 0 ||
        origin_x >= positions_x || origin_y >= positions_y
    ) {
        if (threadIdx.x == 0) {
            refinement_scores[task] = -2.0f;
        }
        return;
    }

    float local_sum = 0.0f;
    float local_square_sum = 0.0f;
    float local_numerator = 0.0f;
    const long long template_pixel_count =
        static_cast<long long>(template_width) * template_height;

    for (
        long long pixel_index = threadIdx.x;
        pixel_index < template_pixel_count;
        pixel_index += blockDim.x
    ) {
        const int template_x = static_cast<int>(pixel_index % template_width);
        const int template_y = static_cast<int>(pixel_index / template_width);
        const float image_value = static_cast<float>(
            image[static_cast<std::size_t>(origin_y + template_y) *
                      image_width +
                  origin_x + template_x]
        );
        const float template_value = centered_template[pixel_index];

        local_sum += image_value;
        local_square_sum = fmaf(image_value, image_value, local_square_sum);
        local_numerator = fmaf(image_value, template_value, local_numerator);
    }

    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_square_sum = shared + blockDim.x;
    float* shared_numerator = shared + blockDim.x * 2;

    shared_sum[threadIdx.x] = local_sum;
    shared_square_sum[threadIdx.x] = local_square_sum;
    shared_numerator[threadIdx.x] = local_numerator;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_square_sum[threadIdx.x] +=
                shared_square_sum[threadIdx.x + stride];
            shared_numerator[threadIdx.x] +=
                shared_numerator[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float sample_count = static_cast<float>(template_pixel_count);
        const float image_energy =
            shared_square_sum[0] -
            (shared_sum[0] * shared_sum[0]) / sample_count;

        float score = -1.0f;
        if (image_energy > 1.0e-5f && template_energy > 1.0e-9f) {
            const float denominator = sqrtf(image_energy * template_energy);
            if (denominator > 1.0e-12f) {
                score = shared_numerator[0] / denominator;
                score = fminf(1.0f, fmaxf(-1.0f, score));
            }
        }
        refinement_scores[task] = score;
    }
}

__global__ void select_refinement_kernel(
    const DeviceCandidate* coarse_candidates,
    int candidate_count,
    const float* refinement_scores,
    int refine_radius,
    DeviceCandidate* refined_candidates
) {
    const int candidate_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (candidate_index >= candidate_count) {
        return;
    }

    const int side = refine_radius * 2 + 1;
    const int positions_per_candidate = side * side;
    const long long score_base =
        static_cast<long long>(candidate_index) * positions_per_candidate;

    float best_score = -2.0f;
    int best_offset_index = refine_radius * side + refine_radius;

    for (int offset_index = 0; offset_index < positions_per_candidate; ++offset_index) {
        const float score = refinement_scores[score_base + offset_index];
        if (score > best_score) {
            best_score = score;
            best_offset_index = offset_index;
        }
    }

    const int best_offset_x =
        best_offset_index % side - refine_radius;
    const int best_offset_y =
        best_offset_index / side - refine_radius;
    const DeviceCandidate source = coarse_candidates[candidate_index];

    refined_candidates[candidate_index] = DeviceCandidate{
        source.x * 2 + best_offset_x,
        source.y * 2 + best_offset_y,
        best_score,
    };
}

float intersection_over_union(
    const DeviceCandidate& a,
    const DeviceCandidate& b,
    int width,
    int height
) {
    const int left = std::max(a.x, b.x);
    const int top = std::max(a.y, b.y);
    const int right = std::min(a.x + width, b.x + width);
    const int bottom = std::min(a.y + height, b.y + height);

    const int intersection_width = std::max(0, right - left);
    const int intersection_height = std::max(0, bottom - top);
    const float intersection_area = static_cast<float>(
        intersection_width * intersection_height
    );

    if (intersection_area <= 0.0f) {
        return 0.0f;
    }

    const float box_area = static_cast<float>(width) * height;
    const float union_area = box_area + box_area - intersection_area;
    return union_area > 0.0f ? intersection_area / union_area : 0.0f;
}

void reset_timing(CudaPatternMatchTiming* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(CudaPatternMatchTiming));
    }
}

int cuda_status(cudaError_t error) {
    return error == cudaSuccess ? kSuccess : static_cast<int>(error);
}

int normalize_pyramid_factor(int requested) {
    requested = std::clamp(requested, 1, kMaximumPyramidFactor);
    int factor = 1;
    while (factor * 2 <= requested) {
        factor *= 2;
    }
    return factor;
}

int normalize_block_threads(int requested, int maximum) {
    if (maximum < 32) {
        return maximum;
    }

    if (requested <= 0) {
        return std::min(256, maximum);
    }

    int normalized = 32;
    while (normalized * 2 <= requested && normalized * 2 <= maximum) {
        normalized *= 2;
    }
    return normalized;
}

std::vector<unsigned char> copy_strided_image(
    const unsigned char* source,
    int width,
    int height,
    int stride
) {
    std::vector<unsigned char> output(
        static_cast<std::size_t>(width) * height
    );
    for (int y = 0; y < height; ++y) {
        std::memcpy(
            output.data() + static_cast<std::size_t>(y) * width,
            source + static_cast<std::size_t>(y) * stride,
            static_cast<std::size_t>(width)
        );
    }
    return output;
}

HostTemplateLevel downsample_template2(const HostTemplateLevel& source) {
    HostTemplateLevel destination;
    destination.width = (source.width + 1) / 2;
    destination.height = (source.height + 1) / 2;
    destination.pixels.resize(
        static_cast<std::size_t>(destination.width) * destination.height
    );

    for (int y = 0; y < destination.height; ++y) {
        for (int x = 0; x < destination.width; ++x) {
            unsigned int sum = 0;
            unsigned int count = 0;
            for (int dy = 0; dy < 2; ++dy) {
                const int source_y = y * 2 + dy;
                if (source_y >= source.height) {
                    continue;
                }
                for (int dx = 0; dx < 2; ++dx) {
                    const int source_x = x * 2 + dx;
                    if (source_x >= source.width) {
                        continue;
                    }
                    sum += source.pixels[
                        static_cast<std::size_t>(source_y) * source.width +
                        source_x
                    ];
                    ++count;
                }
            }
            destination.pixels[
                static_cast<std::size_t>(y) * destination.width + x
            ] = static_cast<unsigned char>((sum + count / 2U) / count);
        }
    }
    return destination;
}

bool center_template(HostTemplateLevel& level) {
    const std::size_t count = level.pixels.size();
    if (count == 0) {
        return false;
    }

    double sum = 0.0;
    for (unsigned char value : level.pixels) {
        sum += static_cast<double>(value);
    }
    const double mean = sum / static_cast<double>(count);

    level.centered.resize(count);
    double energy = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const float centered = static_cast<float>(
            static_cast<double>(level.pixels[index]) - mean
        );
        level.centered[index] = centered;
        energy += static_cast<double>(centered) * centered;
    }

    if (energy <= 1.0e-9) {
        return false;
    }
    level.energy = static_cast<float>(energy);
    return true;
}

int build_template_pyramid(
    const unsigned char* templ,
    int template_width,
    int template_height,
    int template_stride_bytes,
    int requested_factor,
    int image_width,
    int image_height,
    std::vector<HostTemplateLevel>& levels
) {
    HostTemplateLevel first;
    first.width = template_width;
    first.height = template_height;
    first.pixels = copy_strided_image(
        templ,
        template_width,
        template_height,
        template_stride_bytes
    );
    levels.push_back(std::move(first));

    int current_factor = 1;
    while (current_factor < requested_factor) {
        const HostTemplateLevel& current = levels.back();
        const int next_template_width = (current.width + 1) / 2;
        const int next_template_height = (current.height + 1) / 2;
        const int next_image_width =
            (image_width + current_factor * 2 - 1) / (current_factor * 2);
        const int next_image_height =
            (image_height + current_factor * 2 - 1) / (current_factor * 2);

        if (
            next_template_width < kMinCoarseTemplateDimension ||
            next_template_height < kMinCoarseTemplateDimension ||
            next_template_width > next_image_width ||
            next_template_height > next_image_height
        ) {
            break;
        }

        levels.push_back(downsample_template2(current));
        current_factor *= 2;
    }

    for (HostTemplateLevel& level : levels) {
        if (!center_template(level)) {
            return kFlatTemplate;
        }
    }
    return kSuccess;
}

}  // namespace

int cuda_pattern_match_pyramid_zncc_u8(
    const unsigned char* image,
    int image_width,
    int image_height,
    int image_stride_bytes,
    const unsigned char* templ,
    int template_width,
    int template_height,
    int template_stride_bytes,
    float final_score_threshold,
    float coarse_score_threshold,
    int pyramid_factor,
    int refine_radius,
    float nms_iou_threshold,
    int max_results,
    int max_candidates,
    int block_threads,
    CudaPatternMatchResult* results,
    int* result_count,
    CudaPatternMatchTiming* timing
) {
    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    reset_timing(timing);

    if (result_count != nullptr) {
        *result_count = 0;
    }

    if (
        image == nullptr || templ == nullptr || results == nullptr ||
        result_count == nullptr || image_width <= 0 || image_height <= 0 ||
        template_width <= 0 || template_height <= 0 ||
        image_stride_bytes < image_width ||
        template_stride_bytes < template_width ||
        !std::isfinite(final_score_threshold) ||
        !std::isfinite(coarse_score_threshold) ||
        final_score_threshold < -1.0f || final_score_threshold > 1.0f ||
        coarse_score_threshold < -1.0f || coarse_score_threshold > 1.0f ||
        refine_radius < 1 || refine_radius > 16 ||
        !std::isfinite(nms_iou_threshold) ||
        nms_iou_threshold < 0.0f || nms_iou_threshold > 1.0f ||
        max_results <= 0 || max_candidates <= 0
    ) {
        return kInvalidArgument;
    }

    if (template_width > image_width || template_height > image_height) {
        return kTemplateLargerThanImage;
    }

    const bool automatic_pyramid = pyramid_factor <= 0;
    const int normalized_requested_factor = automatic_pyramid
        ? kMaximumPyramidFactor
        : normalize_pyramid_factor(pyramid_factor);
    if (timing != nullptr) {
        timing->requested_pyramid_factor = automatic_pyramid
            ? 0
            : normalized_requested_factor;
    }

    std::vector<HostTemplateLevel> host_template_levels;
    std::vector<DeviceLevel> device_levels;
    float* device_coarse_scores = nullptr;
    DeviceCandidate* device_candidates_a = nullptr;
    DeviceCandidate* device_candidates_b = nullptr;
    float* device_refinement_scores = nullptr;
    int* device_candidate_count = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;

    std::size_t free_vram_before = 0;
    std::size_t total_vram = 0;
    std::size_t minimum_free_vram = std::numeric_limits<std::size_t>::max();

    const auto update_vram_low_watermark = [&]() {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            minimum_free_vram = std::min(minimum_free_vram, free_bytes);
        }
    };

    const auto cleanup = [&]() {
        if (device_candidate_count != nullptr) {
            cudaFree(device_candidate_count);
            device_candidate_count = nullptr;
        }
        if (device_refinement_scores != nullptr) {
            cudaFree(device_refinement_scores);
            device_refinement_scores = nullptr;
        }
        if (device_candidates_b != nullptr) {
            cudaFree(device_candidates_b);
            device_candidates_b = nullptr;
        }
        if (device_candidates_a != nullptr) {
            cudaFree(device_candidates_a);
            device_candidates_a = nullptr;
        }
        if (device_coarse_scores != nullptr) {
            cudaFree(device_coarse_scores);
            device_coarse_scores = nullptr;
        }
        for (DeviceLevel& level : device_levels) {
            if (level.centered_template != nullptr) {
                cudaFree(level.centered_template);
                level.centered_template = nullptr;
            }
            if (level.image != nullptr) {
                cudaFree(level.image);
                level.image = nullptr;
            }
        }
        if (event_stop != nullptr) {
            cudaEventDestroy(event_stop);
            event_stop = nullptr;
        }
        if (event_start != nullptr) {
            cudaEventDestroy(event_start);
            event_start = nullptr;
        }
    };

    const auto finish = [&](int code) {
        if (timing != nullptr) {
            timing->total_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - total_start
                ).count();
            constexpr double mib = 1024.0 * 1024.0;
            timing->device_total_vram_mib =
                static_cast<double>(total_vram) / mib;
            timing->device_free_vram_before_mib =
                static_cast<double>(free_vram_before) / mib;
            if (minimum_free_vram != std::numeric_limits<std::size_t>::max()) {
                timing->device_free_vram_after_alloc_mib =
                    static_cast<double>(minimum_free_vram) / mib;
                timing->estimated_vram_used_mib =
                    static_cast<double>(free_vram_before - minimum_free_vram) /
                    mib;
            }
        }
        cleanup();
        return code;
    };

    try {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (device_count <= 0) {
            return finish(kNoCudaDevice);
        }

        cudaDeviceProp properties{};
        error = cudaGetDeviceProperties(&properties, 0);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        const int compute_threads = normalize_block_threads(
            block_threads,
            properties.maxThreadsPerBlock
        );
        const int peak_threads = std::min(256, properties.maxThreadsPerBlock);

        if (timing != nullptr) {
            timing->coarse_threads_per_block = compute_threads;
            timing->peak_threads_per_block = peak_threads;
            timing->refine_threads_per_block = compute_threads;
        }

        const auto prepare_start = Clock::now();
        const int pyramid_status = build_template_pyramid(
            templ,
            template_width,
            template_height,
            template_stride_bytes,
            normalized_requested_factor,
            image_width,
            image_height,
            host_template_levels
        );
        if (pyramid_status != kSuccess) {
            return finish(pyramid_status);
        }

        const int level_count = static_cast<int>(host_template_levels.size());
        const int actual_factor = 1 << (level_count - 1);
        device_levels.resize(static_cast<std::size_t>(level_count));

        int level_image_width = image_width;
        int level_image_height = image_height;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            DeviceLevel& device_level = device_levels[level_index];
            const HostTemplateLevel& host_level =
                host_template_levels[level_index];
            device_level.image_width = level_image_width;
            device_level.image_height = level_image_height;
            device_level.template_width = host_level.width;
            device_level.template_height = host_level.height;
            device_level.template_energy = host_level.energy;
            level_image_width = (level_image_width + 1) / 2;
            level_image_height = (level_image_height + 1) / 2;
        }

        if (timing != nullptr) {
            timing->actual_pyramid_factor = actual_factor;
            timing->pyramid_level_count = level_count;
            timing->host_prepare_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - prepare_start
                ).count();
        }

        error = cudaMemGetInfo(&free_vram_before, &total_vram);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        minimum_free_vram = free_vram_before;

        error = cudaEventCreate(&event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        error = cudaEventCreate(&event_stop);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        // Allocate/copy the original image as a compact contiguous plane.
        {
            DeviceLevel& first_level = device_levels[0];
            const std::size_t image_bytes =
                static_cast<std::size_t>(first_level.image_width) *
                first_level.image_height;
            error = cudaMalloc(
                reinterpret_cast<void**>(&first_level.image),
                image_bytes
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
            error = cudaMemcpy2D(
                first_level.image,
                static_cast<std::size_t>(first_level.image_width),
                image,
                static_cast<std::size_t>(image_stride_bytes),
                static_cast<std::size_t>(image_width),
                static_cast<std::size_t>(image_height),
                cudaMemcpyHostToDevice
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
        }

        for (int level_index = 0; level_index < level_count; ++level_index) {
            DeviceLevel& device_level = device_levels[level_index];
            const HostTemplateLevel& host_level =
                host_template_levels[level_index];
            const std::size_t template_bytes =
                host_level.centered.size() * sizeof(float);
            error = cudaMalloc(
                reinterpret_cast<void**>(&device_level.centered_template),
                template_bytes
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
            error = cudaMemcpy(
                device_level.centered_template,
                host_level.centered.data(),
                template_bytes,
                cudaMemcpyHostToDevice
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
        }

        error = cudaEventRecord(event_stop);
        if (error == cudaSuccess) {
            error = cudaEventSynchronize(event_stop);
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (timing != nullptr) {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, event_start, event_stop);
            timing->host_to_device_ms = elapsed;
        }
        update_vram_low_watermark();

        // Build the image pyramid entirely on the GPU.
        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        int kernel_launch_count = 0;
        const dim3 downsample_threads(32, 8);
        for (int level_index = 1; level_index < level_count; ++level_index) {
            DeviceLevel& source_level = device_levels[level_index - 1];
            DeviceLevel& destination_level = device_levels[level_index];
            const std::size_t destination_bytes =
                static_cast<std::size_t>(destination_level.image_width) *
                destination_level.image_height;
            error = cudaMalloc(
                reinterpret_cast<void**>(&destination_level.image),
                destination_bytes
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }

            const dim3 blocks(
                (destination_level.image_width + downsample_threads.x - 1) /
                    downsample_threads.x,
                (destination_level.image_height + downsample_threads.y - 1) /
                    downsample_threads.y
            );
            downsample2_u8_kernel<<<blocks, downsample_threads>>>(
                source_level.image,
                source_level.image_width,
                source_level.image_height,
                destination_level.image,
                destination_level.image_width,
                destination_level.image_height
            );
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
            ++kernel_launch_count;
            update_vram_low_watermark();
        }

        error = cudaEventRecord(event_stop);
        if (error == cudaSuccess) {
            error = cudaEventSynchronize(event_stop);
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (timing != nullptr) {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, event_start, event_stop);
            timing->pyramid_build_ms = elapsed;
        }

        DeviceLevel& coarse_level = device_levels.back();
        const int coarse_positions_x =
            coarse_level.image_width - coarse_level.template_width + 1;
        const int coarse_positions_y =
            coarse_level.image_height - coarse_level.template_height + 1;
        const long long coarse_position_count =
            static_cast<long long>(coarse_positions_x) * coarse_positions_y;

        if (coarse_positions_x <= 0 || coarse_positions_y <= 0) {
            return finish(kTemplateLargerThanImage);
        }
        if (
            static_cast<unsigned long long>(coarse_position_count) >
            std::numeric_limits<std::size_t>::max() / sizeof(float)
        ) {
            return finish(kDeviceAllocationTooLarge);
        }

        if (timing != nullptr) {
            timing->coarse_evaluated_positions = coarse_position_count;
            timing->coarse_image_width = coarse_level.image_width;
            timing->coarse_image_height = coarse_level.image_height;
            timing->coarse_template_width = coarse_level.template_width;
            timing->coarse_template_height = coarse_level.template_height;
        }

        const std::size_t coarse_score_bytes =
            static_cast<std::size_t>(coarse_position_count) * sizeof(float);
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_coarse_scores),
            coarse_score_bytes
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        const std::size_t candidate_bytes =
            static_cast<std::size_t>(max_candidates) * sizeof(DeviceCandidate);
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_candidates_a),
            candidate_bytes
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_candidates_b),
            candidate_bytes
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        error = cudaMalloc(
            reinterpret_cast<void**>(&device_candidate_count),
            sizeof(int)
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        update_vram_low_watermark();

        // Coarse exhaustive score map. At factor 8 this is ~1/4096 of the
        // original direct-ZNCC arithmetic for the same full-resolution template.
        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        const long long coarse_blocks =
            (coarse_position_count + compute_threads - 1) / compute_threads;
        if (coarse_blocks > properties.maxGridSize[0]) {
            return finish(kDeviceAllocationTooLarge);
        }
        coarse_zncc_score_kernel<<<
            static_cast<unsigned int>(coarse_blocks),
            compute_threads
        >>>(
            coarse_level.image,
            coarse_level.image_width,
            coarse_level.centered_template,
            coarse_level.template_width,
            coarse_level.template_height,
            coarse_level.template_energy,
            coarse_positions_x,
            coarse_positions_y,
            device_coarse_scores
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        ++kernel_launch_count;

        error = cudaEventRecord(event_stop);
        if (error == cudaSuccess) {
            error = cudaEventSynchronize(event_stop);
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (timing != nullptr) {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, event_start, event_stop);
            timing->coarse_score_ms = elapsed;
        }

        // Local maxima extraction keeps only one point per coarse peak.
        error = cudaMemset(device_candidate_count, 0, sizeof(int));
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        const long long peak_blocks =
            (coarse_position_count + peak_threads - 1) / peak_threads;
        if (peak_blocks > properties.maxGridSize[0]) {
            return finish(kDeviceAllocationTooLarge);
        }
        local_maxima_kernel<<<
            static_cast<unsigned int>(peak_blocks),
            peak_threads
        >>>(
            device_coarse_scores,
            coarse_positions_x,
            coarse_positions_y,
            coarse_score_threshold,
            device_candidates_a,
            max_candidates,
            device_candidate_count
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        ++kernel_launch_count;

        error = cudaEventRecord(event_stop);
        if (error == cudaSuccess) {
            error = cudaEventSynchronize(event_stop);
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (timing != nullptr) {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, event_start, event_stop);
            timing->coarse_peak_ms = elapsed;
        }

        int raw_candidate_count = 0;
        error = cudaMemcpy(
            &raw_candidate_count,
            device_candidate_count,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        int stored_candidate_count =
            std::min(raw_candidate_count, max_candidates);

        if (timing != nullptr) {
            timing->raw_candidate_count = raw_candidate_count;
            timing->stored_candidate_count = stored_candidate_count;
            timing->candidate_overflow =
                raw_candidate_count > max_candidates ? 1 : 0;
        }

        // Coarse score map is no longer needed. Free it before full-resolution
        // refinement so VRAM is available to the refinement scratch buffer.
        cudaFree(device_coarse_scores);
        device_coarse_scores = nullptr;

        DeviceCandidate* current_candidates = device_candidates_a;
        DeviceCandidate* next_candidates = device_candidates_b;
        const int side = refine_radius * 2 + 1;
        const int positions_per_candidate = side * side;
        const long long maximum_refinement_tasks =
            static_cast<long long>(stored_candidate_count) *
            positions_per_candidate;

        if (
            maximum_refinement_tasks > 0 &&
            static_cast<unsigned long long>(maximum_refinement_tasks) >
                std::numeric_limits<std::size_t>::max() / sizeof(float)
        ) {
            return finish(kDeviceAllocationTooLarge);
        }

        if (maximum_refinement_tasks > 0 && level_count > 1) {
            error = cudaMalloc(
                reinterpret_cast<void**>(&device_refinement_scores),
                static_cast<std::size_t>(maximum_refinement_tasks) *
                    sizeof(float)
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
            update_vram_low_watermark();
        }

        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        long long refine_evaluated_positions = 0;
        for (
            int level_index = level_count - 2;
            level_index >= 0 && stored_candidate_count > 0;
            --level_index
        ) {
            const DeviceLevel& fine_level = device_levels[level_index];
            const long long total_tasks =
                static_cast<long long>(stored_candidate_count) *
                positions_per_candidate;
            refine_evaluated_positions += total_tasks;

            const long long template_pixels =
                static_cast<long long>(fine_level.template_width) *
                fine_level.template_height;
            const long long tasks_per_launch = std::max(
                kMinimumRefineTasksPerLaunch,
                std::min(
                    total_tasks,
                    kTargetRefinePixelsPerLaunch /
                        std::max(1LL, template_pixels)
                )
            );
            const std::size_t shared_bytes =
                static_cast<std::size_t>(compute_threads) * 3 * sizeof(float);

            for (
                long long task_offset = 0;
                task_offset < total_tasks;
                task_offset += tasks_per_launch
            ) {
                const long long tasks_this_launch = std::min(
                    tasks_per_launch,
                    total_tasks - task_offset
                );
                if (tasks_this_launch > properties.maxGridSize[0]) {
                    return finish(kDeviceAllocationTooLarge);
                }
                refine_score_blocks_kernel<<<
                    static_cast<unsigned int>(tasks_this_launch),
                    compute_threads,
                    shared_bytes
                >>>(
                    fine_level.image,
                    fine_level.image_width,
                    fine_level.image_height,
                    fine_level.centered_template,
                    fine_level.template_width,
                    fine_level.template_height,
                    fine_level.template_energy,
                    current_candidates,
                    stored_candidate_count,
                    refine_radius,
                    task_offset,
                    tasks_this_launch,
                    device_refinement_scores
                );
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return finish(cuda_status(error));
                }
                ++kernel_launch_count;
            }

            const int select_blocks =
                (stored_candidate_count + peak_threads - 1) / peak_threads;
            select_refinement_kernel<<<select_blocks, peak_threads>>>(
                current_candidates,
                stored_candidate_count,
                device_refinement_scores,
                refine_radius,
                next_candidates
            );
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
            ++kernel_launch_count;
            std::swap(current_candidates, next_candidates);
        }

        error = cudaEventRecord(event_stop);
        if (error == cudaSuccess) {
            error = cudaEventSynchronize(event_stop);
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (timing != nullptr) {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, event_start, event_stop);
            timing->refine_ms = elapsed;
            timing->refine_evaluated_positions = refine_evaluated_positions;
            timing->kernel_launch_count = kernel_launch_count;
        }

        // D2H only transfers the refined candidate list, not a full-resolution
        // score map.
        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        std::vector<DeviceCandidate> host_candidates(
            static_cast<std::size_t>(stored_candidate_count)
        );
        if (stored_candidate_count > 0) {
            error = cudaMemcpy(
                host_candidates.data(),
                current_candidates,
                static_cast<std::size_t>(stored_candidate_count) *
                    sizeof(DeviceCandidate),
                cudaMemcpyDeviceToHost
            );
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }
        }
        error = cudaEventRecord(event_stop);
        if (error == cudaSuccess) {
            error = cudaEventSynchronize(event_stop);
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }
        if (timing != nullptr) {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, event_start, event_stop);
            timing->device_to_host_ms = elapsed;
        }

        const auto nms_start = Clock::now();
        host_candidates.erase(
            std::remove_if(
                host_candidates.begin(),
                host_candidates.end(),
                [&](const DeviceCandidate& candidate) {
                    return !std::isfinite(candidate.score) ||
                           candidate.score < final_score_threshold;
                }
            ),
            host_candidates.end()
        );

        std::sort(
            host_candidates.begin(),
            host_candidates.end(),
            [](const DeviceCandidate& left, const DeviceCandidate& right) {
                return left.score > right.score;
            }
        );

        std::vector<DeviceCandidate> selected;
        selected.reserve(static_cast<std::size_t>(max_results));
        for (const DeviceCandidate& candidate : host_candidates) {
            bool suppressed = false;
            for (const DeviceCandidate& accepted : selected) {
                if (
                    intersection_over_union(
                        candidate,
                        accepted,
                        template_width,
                        template_height
                    ) > nms_iou_threshold
                ) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                selected.push_back(candidate);
                if (static_cast<int>(selected.size()) >= max_results) {
                    break;
                }
            }
        }

        for (std::size_t index = 0; index < selected.size(); ++index) {
            results[index] = CudaPatternMatchResult{
                selected[index].x,
                selected[index].y,
                template_width,
                template_height,
                selected[index].score,
            };
        }
        *result_count = static_cast<int>(selected.size());

        if (timing != nullptr) {
            timing->sort_nms_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - nms_start
                ).count();
            timing->result_count = *result_count;
        }

        return finish(kSuccess);
    } catch (const std::bad_alloc&) {
        return finish(kHostAllocationFailure);
    } catch (...) {
        return finish(kHostAllocationFailure);
    }
}

int cuda_pattern_match_zncc_u8(
    const unsigned char* image,
    int image_width,
    int image_height,
    int image_stride_bytes,
    const unsigned char* templ,
    int template_width,
    int template_height,
    int template_stride_bytes,
    float score_threshold,
    int search_step,
    float nms_iou_threshold,
    int max_results,
    int max_candidates,
    CudaPatternMatchResult* results,
    int* result_count,
    CudaPatternMatchTiming* timing
) {
    const int factor = normalize_pyramid_factor(search_step);
    const float coarse_threshold = std::max(-1.0f, score_threshold - 0.15f);
    return cuda_pattern_match_pyramid_zncc_u8(
        image,
        image_width,
        image_height,
        image_stride_bytes,
        templ,
        template_width,
        template_height,
        template_stride_bytes,
        score_threshold,
        coarse_threshold,
        factor,
        2,
        nms_iou_threshold,
        max_results,
        max_candidates,
        0,
        results,
        result_count,
        timing
    );
}

const char* cuda_pattern_match_error_string(int error_code) {
    switch (error_code) {
        case kSuccess:
            return "Success";
        case kInvalidArgument:
            return "Invalid argument";
        case kTemplateLargerThanImage:
            return "Template is larger than the source image";
        case kFlatTemplate:
            return "Template has no usable intensity variation";
        case kHostAllocationFailure:
            return "Host memory allocation failed";
        case kNoCudaDevice:
            return "No CUDA device is available";
        case kDeviceAllocationTooLarge:
            return "Requested CUDA grid or allocation is too large";
        default:
            return cudaGetErrorString(static_cast<cudaError_t>(error_code));
    }
}
