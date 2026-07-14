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

CUDA_DLL_API int cuda_bgr_to_gray_u8(
    const unsigned char* bgr,
    unsigned char* gray,
    int width,
    int height
);

CUDA_DLL_API const char* cuda_error_string(int error_code);
