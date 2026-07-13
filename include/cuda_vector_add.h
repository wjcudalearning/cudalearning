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

// 回傳 CUDA error code；0 代表 cudaSuccess。
CUDA_DLL_API int cuda_get_device_count(int* count);

// a、b、out 都是 CPU 端 float 陣列，長度為 n。
// DLL 內部會配置 GPU 記憶體、執行 kernel，再將結果複製回 out。
CUDA_DLL_API int cuda_vector_add_f32(
    const float* a,
    const float* b,
    float* out,
    int n
);

// 將 CUDA error code 轉成可讀文字。
// 回傳字串由 CUDA runtime 管理，不需要呼叫端釋放。
CUDA_DLL_API const char* cuda_error_string(int error_code);
