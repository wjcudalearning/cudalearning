#include "visionflow_cuda.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
bool check_result(int result, const char* operation) {
    if (result == VF_CUDA_OK) return true;
    char message[256]{};
    vf_gpu_error_message(result, message, static_cast<int>(sizeof(message)));
    std::cerr << operation << " failed: " << message
              << " (code=" << result << ")\n";
    return false;
}

bool all_equal(const std::vector<uint8_t>& values, uint8_t expected) {
    return std::all_of(
        values.begin(), values.end(),
        [expected](uint8_t value) { return value == expected; });
}
}  // namespace

int main() {
    std::cout << "VisionFlow CUDA ABI: " << vf_gpu_abi_version() << "\n";
    if (vf_gpu_abi_version() != VF_CUDA_ABI_VERSION) {
        std::cerr << "ABI mismatch\n";
        return 2;
    }

    const int device_count = vf_gpu_device_count();
    std::cout << "CUDA device count: " << device_count << "\n";
    if (device_count <= 0) {
        std::cerr << "No CUDA device. Run this executable on an NVIDIA GPU machine.\n";
        return 3;
    }

    char device_name[256]{};
    if (!check_result(
            vf_gpu_device_name(
                device_name, static_cast<int>(sizeof(device_name))),
            "vf_gpu_device_name")) {
        return 4;
    }

    const int capability = vf_gpu_compute_capability();
    std::cout << "Device: " << device_name << "\n";
    std::cout << "Compute capability: "
              << capability / 10 << "." << capability % 10 << "\n";

    constexpr int width = 8;
    constexpr int height = 8;
    std::vector<uint8_t> bgr(width * height * 3, 128);
    std::vector<uint8_t> gray(width * height, 0);

    if (!check_result(
            vf_bgr_to_gray_u8(
                bgr.data(), width, height, width * 3, 3,
                gray.data(), width, 1),
            "vf_bgr_to_gray_u8")) {
        return 5;
    }
    if (!all_equal(gray, 128)) {
        std::cerr << "Grayscale value verification failed\n";
        return 6;
    }

    std::vector<uint8_t> blurred(width * height, 0);
    if (!check_result(
            vf_gaussian_blur_u8(
                gray.data(), width, height, width, 1,
                blurred.data(), width, 1, 5),
            "vf_gaussian_blur_u8")) {
        return 7;
    }
    if (!all_equal(blurred, 128)) {
        std::cerr << "Gaussian uniform-image verification failed\n";
        return 8;
    }

    std::vector<uint8_t> binary(width * height, 0);
    if (!check_result(
            vf_threshold_u8(
                gray.data(), width, height, width, 1,
                binary.data(), width, 1, 127, 255, 0),
            "vf_threshold_u8")) {
        return 9;
    }
    if (!all_equal(binary, 255)) {
        std::cerr << "Global threshold verification failed\n";
        return 10;
    }

    std::fill(binary.begin(), binary.end(), 0);
    if (!check_result(
            vf_adaptive_mean_u8(
                gray.data(), width, height, width, 1,
                binary.data(), width, 1, 3, 2.0f, 255, 0),
            "vf_adaptive_mean_u8")) {
        return 11;
    }
    if (!all_equal(binary, 255)) {
        std::cerr << "Adaptive threshold verification failed\n";
        return 12;
    }

    std::vector<uint8_t> impulse(width * height, 0);
    impulse[(height / 2) * width + width / 2] = 255;
    std::vector<uint8_t> dilated(width * height, 0);
    if (!check_result(
            vf_morphology_rect_u8(
                impulse.data(), width, height, width, 1,
                dilated.data(), width, 1,
                VF_MORPH_DILATE, 3, 1),
            "vf_morphology_rect_u8")) {
        return 13;
    }
    const int white_count = static_cast<int>(std::count(
        dilated.begin(), dilated.end(), static_cast<uint8_t>(255)));
    if (white_count != 9) {
        std::cerr << "Morphology verification failed: expected 9 white pixels, got "
                  << white_count << "\n";
        return 14;
    }

    std::cout << "PASS: ABI, grayscale, separable Gaussian, integral adaptive "
                 "threshold, global threshold and morphology smoke tests\n";
    return 0;
}
