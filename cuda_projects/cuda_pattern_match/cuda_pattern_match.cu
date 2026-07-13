#include "cuda_pattern_match.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr int kSuccess = 0;
constexpr int kInvalidArgument = -1;
constexpr int kTemplateLargerThanImage = -2;
constexpr int kFlatTemplate = -3;
constexpr int kHostAllocationFailure = -4;

struct DeviceCandidate {
    int x;
    int y;
    float score;
};

__global__ void zncc_match_kernel(
    const unsigned char* image,
    int image_stride_bytes,
    const float* centered_template,
    int template_width,
    int template_height,
    float template_energy,
    float score_threshold,
    int search_step,
    int positions_x,
    long long position_offset,
    long long positions_this_launch,
    DeviceCandidate* candidates,
    int max_candidates,
    int* candidate_count
) {
    const long long local_index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (local_index >= positions_this_launch) {
        return;
    }

    const long long linear_index = position_offset + local_index;

    const int position_index_x =
        static_cast<int>(linear_index % positions_x);
    const int position_index_y =
        static_cast<int>(linear_index / positions_x);

    const int origin_x = position_index_x * search_step;
    const int origin_y = position_index_y * search_step;

    float image_sum = 0.0f;
    float image_square_sum = 0.0f;
    float numerator = 0.0f;

    for (int template_y = 0; template_y < template_height; ++template_y) {
        const unsigned char* image_row =
            image + static_cast<std::size_t>(origin_y + template_y) *
                        image_stride_bytes +
            origin_x;

        const float* template_row =
            centered_template +
            static_cast<std::size_t>(template_y) * template_width;

        for (int template_x = 0; template_x < template_width; ++template_x) {
            const float image_value =
                static_cast<float>(image_row[template_x]);

            image_sum += image_value;
            image_square_sum += image_value * image_value;
            numerator += image_value * template_row[template_x];
        }
    }

    const float sample_count =
        static_cast<float>(template_width * template_height);
    const float image_energy =
        image_square_sum - (image_sum * image_sum) / sample_count;

    if (image_energy <= 1.0e-6f) {
        return;
    }

    const float denominator = sqrtf(image_energy * template_energy);

    if (denominator <= 1.0e-12f) {
        return;
    }

    float score = numerator / denominator;
    score = fminf(1.0f, fmaxf(-1.0f, score));

    if (score < score_threshold) {
        return;
    }

    const int candidate_index = atomicAdd(candidate_count, 1);

    if (candidate_index < max_candidates) {
        candidates[candidate_index] = DeviceCandidate{
            origin_x,
            origin_y,
            score,
        };
    }
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

    const float box_area = static_cast<float>(width * height);
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

}  // namespace

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
    using Clock = std::chrono::steady_clock;

    const auto total_start = Clock::now();
    reset_timing(timing);

    if (result_count != nullptr) {
        *result_count = 0;
    }

    if (
        image == nullptr ||
        templ == nullptr ||
        results == nullptr ||
        result_count == nullptr ||
        image_width <= 0 ||
        image_height <= 0 ||
        template_width <= 0 ||
        template_height <= 0 ||
        image_stride_bytes < image_width ||
        template_stride_bytes < template_width ||
        !std::isfinite(score_threshold) ||
        score_threshold < -1.0f ||
        score_threshold > 1.0f ||
        search_step <= 0 ||
        !std::isfinite(nms_iou_threshold) ||
        nms_iou_threshold < 0.0f ||
        nms_iou_threshold > 1.0f ||
        max_results <= 0 ||
        max_candidates <= 0
    ) {
        return kInvalidArgument;
    }

    if (
        template_width > image_width ||
        template_height > image_height
    ) {
        return kTemplateLargerThanImage;
    }

    unsigned char* device_image = nullptr;
    float* device_template = nullptr;
    DeviceCandidate* device_candidates = nullptr;
    int* device_candidate_count = nullptr;

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;

    const auto cleanup = [&]() {
        if (device_candidate_count != nullptr) {
            cudaFree(device_candidate_count);
            device_candidate_count = nullptr;
        }
        if (device_candidates != nullptr) {
            cudaFree(device_candidates);
            device_candidates = nullptr;
        }
        if (device_template != nullptr) {
            cudaFree(device_template);
            device_template = nullptr;
        }
        if (device_image != nullptr) {
            cudaFree(device_image);
            device_image = nullptr;
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
        cleanup();
        const auto total_stop = Clock::now();
        if (timing != nullptr) {
            timing->total_ms =
                std::chrono::duration<double, std::milli>(
                    total_stop - total_start
                ).count();
        }
        return code;
    };

    try {
        const auto prepare_start = Clock::now();

        const std::size_t template_pixel_count =
            static_cast<std::size_t>(template_width) * template_height;
        std::vector<float> centered_template(template_pixel_count);

        double template_sum = 0.0;

        for (int y = 0; y < template_height; ++y) {
            const unsigned char* row =
                templ + static_cast<std::size_t>(y) * template_stride_bytes;

            for (int x = 0; x < template_width; ++x) {
                template_sum += static_cast<double>(row[x]);
            }
        }

        const double template_mean =
            template_sum / static_cast<double>(template_pixel_count);
        double template_energy_double = 0.0;

        for (int y = 0; y < template_height; ++y) {
            const unsigned char* row =
                templ + static_cast<std::size_t>(y) * template_stride_bytes;

            for (int x = 0; x < template_width; ++x) {
                const float centered = static_cast<float>(
                    static_cast<double>(row[x]) - template_mean
                );
                centered_template[
                    static_cast<std::size_t>(y) * template_width + x
                ] = centered;
                template_energy_double +=
                    static_cast<double>(centered) * centered;
            }
        }

        if (template_energy_double <= 1.0e-9) {
            return finish(kFlatTemplate);
        }

        const float template_energy =
            static_cast<float>(template_energy_double);

        const int positions_x =
            (image_width - template_width) / search_step + 1;
        const int positions_y =
            (image_height - template_height) / search_step + 1;
        const long long total_positions =
            static_cast<long long>(positions_x) * positions_y;

        if (timing != nullptr) {
            timing->evaluated_positions = total_positions;
        }

        const auto prepare_stop = Clock::now();
        if (timing != nullptr) {
            timing->host_prepare_ms =
                std::chrono::duration<double, std::milli>(
                    prepare_stop - prepare_start
                ).count();
        }

        const std::size_t image_bytes =
            static_cast<std::size_t>(image_stride_bytes) * image_height;
        const std::size_t template_bytes =
            template_pixel_count * sizeof(float);
        const std::size_t candidate_bytes =
            static_cast<std::size_t>(max_candidates) *
            sizeof(DeviceCandidate);

        cudaError_t error = cudaEventCreate(&event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaEventCreate(&event_stop);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaMalloc(
            reinterpret_cast<void**>(&device_image),
            image_bytes
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaMalloc(
            reinterpret_cast<void**>(&device_template),
            template_bytes
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaMalloc(
            reinterpret_cast<void**>(&device_candidates),
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

        error = cudaMemset(device_candidate_count, 0, sizeof(int));
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaMemcpy(
            device_image,
            image,
            image_bytes,
            cudaMemcpyHostToDevice
        );
        if (error == cudaSuccess) {
            error = cudaMemcpy(
                device_template,
                centered_template.data(),
                template_bytes,
                cudaMemcpyHostToDevice
            );
        }
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
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

        constexpr int threads = 128;
        constexpr long long target_template_comparisons_per_launch =
            100000000LL;
        constexpr long long maximum_positions_per_launch = 1000000LL;

        const long long template_pixels_for_batch =
            static_cast<long long>(template_pixel_count);
        const long long positions_per_launch = std::max(
            1LL,
            std::min(
                maximum_positions_per_launch,
                target_template_comparisons_per_launch /
                    std::max(1LL, template_pixels_for_batch)
            )
        );

        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        int kernel_launch_count = 0;

        for (
            long long position_offset = 0;
            position_offset < total_positions;
            position_offset += positions_per_launch
        ) {
            const long long positions_this_launch = std::min(
                positions_per_launch,
                total_positions - position_offset
            );
            const long long blocks_long =
                (positions_this_launch + threads - 1) / threads;

            if (
                blocks_long <= 0 ||
                blocks_long > std::numeric_limits<int>::max()
            ) {
                return finish(kInvalidArgument);
            }

            zncc_match_kernel<<<static_cast<int>(blocks_long), threads>>>(
                device_image,
                image_stride_bytes,
                device_template,
                template_width,
                template_height,
                template_energy,
                score_threshold,
                search_step,
                positions_x,
                position_offset,
                positions_this_launch,
                device_candidates,
                max_candidates,
                device_candidate_count
            );

            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return finish(cuda_status(error));
            }

            ++kernel_launch_count;
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
            timing->kernel_ms = elapsed;
            timing->kernel_launch_count = kernel_launch_count;
        }

        int raw_candidate_count = 0;

        error = cudaEventRecord(event_start);
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        error = cudaMemcpy(
            &raw_candidate_count,
            device_candidate_count,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
        if (error != cudaSuccess) {
            return finish(cuda_status(error));
        }

        const int stored_candidate_count =
            std::min(raw_candidate_count, max_candidates);
        std::vector<DeviceCandidate> host_candidates(
            static_cast<std::size_t>(stored_candidate_count)
        );

        if (stored_candidate_count > 0) {
            error = cudaMemcpy(
                host_candidates.data(),
                device_candidates,
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
            timing->raw_candidate_count = raw_candidate_count;
            timing->stored_candidate_count = stored_candidate_count;
            timing->candidate_overflow =
                raw_candidate_count > max_candidates ? 1 : 0;
        }

        const auto nms_start = Clock::now();

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

        const auto nms_stop = Clock::now();
        if (timing != nullptr) {
            timing->sort_nms_ms =
                std::chrono::duration<double, std::milli>(
                    nms_stop - nms_start
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
        default:
            return cudaGetErrorString(static_cast<cudaError_t>(error_code));
    }
}
