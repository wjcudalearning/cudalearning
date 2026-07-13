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

// v2 timing layout. Python ctypes must use the same field order.
struct CudaPatternMatchTiming {
    double host_prepare_ms;
    double host_to_device_ms;
    double pyramid_build_ms;
    double coarse_score_ms;
    double coarse_peak_ms;
    double refine_ms;
    double device_to_host_ms;
    double sort_nms_ms;
    double total_ms;

    long long coarse_evaluated_positions;
    long long refine_evaluated_positions;

    int raw_candidate_count;
    int stored_candidate_count;
    int result_count;
    int candidate_overflow;
    int kernel_launch_count;

    int requested_pyramid_factor;
    int actual_pyramid_factor;
    int pyramid_level_count;
    int coarse_image_width;
    int coarse_image_height;
    int coarse_template_width;
    int coarse_template_height;

    int coarse_threads_per_block;
    int peak_threads_per_block;
    int refine_threads_per_block;

    double device_total_vram_mib;
    double device_free_vram_before_mib;
    double device_free_vram_after_alloc_mib;
    double estimated_vram_used_mib;
};

// 正式版：金字塔 ZNCC。
//
// 流程：
// 1. GPU 建立 2x 影像金字塔。
// 2. 在最粗層逐像素計算 ZNCC score map。
// 3. GPU 取局部極大值候選。
// 4. 每一層以 refine_radius 在候選周圍精搜，最後回到原圖像素座標。
// 5. CPU 只做最後排序與 IoU NMS。
//
// pyramid_factor 建議 4/8/16，DLL 會自動降到 Template 仍可辨識的倍率。
// coarse_score_threshold 應略低於 final_score_threshold，避免下採樣後漏檢。
// block_threads=0 代表自動；也可指定 128/256/512/1024。
CUDA_DLL_API int cuda_pattern_match_pyramid_zncc_u8(
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
);

// 舊介面相容包裝。search_step 會被正規化為 1/2/4/8/16/32 的金字塔倍率。
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
