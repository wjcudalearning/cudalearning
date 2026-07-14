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

CUDA_DLL_API int cuda_roi_batch_threshold_u8(
    const unsigned char* rois,
    unsigned char* output,
    int roi_width,
    int roi_height,
    int roi_count,
    unsigned char threshold
);

CUDA_DLL_API const char* cuda_error_string(int error_code);
