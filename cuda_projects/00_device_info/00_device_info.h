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

struct CudaDeviceInfo {
    char name[256];
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    unsigned long long total_global_memory_bytes;
};

CUDA_DLL_API int cuda_get_device_count(int* count);
CUDA_DLL_API int cuda_get_device_info(int device_index, CudaDeviceInfo* info);
CUDA_DLL_API const char* cuda_error_string(int error_code);
