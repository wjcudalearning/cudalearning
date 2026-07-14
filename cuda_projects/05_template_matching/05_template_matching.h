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

CUDA_DLL_API int cuda_template_match_best_ssd_u8(
    const unsigned char* image,
    int image_width,
    int image_height,
    const unsigned char* templ,
    int template_width,
    int template_height,
    int* best_x,
    int* best_y,
    float* best_mean_ssd
);

CUDA_DLL_API const char* cuda_error_string(int error_code);
