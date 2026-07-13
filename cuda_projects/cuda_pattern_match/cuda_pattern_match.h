#pragma once

#ifdef _WIN32
    #ifdef CUDA_DLL_EXPORTS
        #define CUDA_DLL_API extern "C" __declspec(dllexport)
    #else
        #define CUDA_DLL_API extern "C" __declspec(dllimport)
    #endif
#else
    #define CUDA_DLL_API extern "C"
#endif

struct CudaPatternMatchResult {
    int x;
    int y;
    int width;
    int height;
    float score;
};

struct CudaPatternMatchTiming {
    double host_prepare_ms;
    double host_to_device_ms;
    double kernel_ms;
    double device_to_host_ms;
    double sort_nms_ms;
    double total_ms;

    long long evaluated_positions;
    int raw_candidate_count;
    int stored_candidate_count;
    int result_count;
    int candidate_overflow;
    int kernel_launch_count;
};

// 對 8-bit 單通道影像執行 ZNCC template matching。
//
// image_stride_bytes/template_stride_bytes 是每列位元組數。
// search_step=1 代表逐像素搜尋；較大的值可換取更快速度。
// nms_iou_threshold 越小，重疊框抑制越強。
// max_candidates 是 GPU 暫存候選點容量；太低時 timing.candidate_overflow 會是 1。
CUDA_DLL_API int cuda_pattern_match_zncc_u8(
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
);

CUDA_DLL_API const char* cuda_pattern_match_error_string(int error_code);
