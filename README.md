# CUDA 多子專案 GitHub Actions

固定根目錄：

```text
cuda_projects/
```

每個第一層子資料夾都是一個獨立 DLL 專案：

```text
cuda_projects/
├─ cuda_vector_add/
│  ├─ cuda_vector_add.cu
│  ├─ cuda_vector_add.h
│  └─ test.py
└─ cuda_wic_image/
   ├─ cuda_wic_image.cu
   ├─ cuda_wic_image.h
   └─ test.py
```

子資料夾名稱會直接成為 DLL 名稱：

```text
cuda_vector_add/ → cuda_vector_add.dll
cuda_wic_image/  → cuda_wic_image.dll
```

## Push 時會發生什麼

如果這次 push 只新增或修改：

```text
cuda_projects/cuda_wic_image/
```

Action 只會執行：

```text
Build cuda_wic_image
```

不會重新編譯其他子專案。

如果同一次 push 修改三個子專案，就會建立三個 matrix jobs，
分別編譯並上傳三個 artifacts。

## 新增專案

建立：

```text
cuda_projects/my_filter/
```

放入：

```text
cuda_projects/my_filter/my_filter.cu
cuda_projects/my_filter/my_filter.h
cuda_projects/my_filter/test.py
```

然後：

```powershell
git add cuda_projects/my_filter
git commit -m "add my_filter CUDA project"
git push
```

編譯成功後，Actions 會產生：

```text
my_filter-windows-x64-sm86
```

其中包含：

```text
my_filter.dll
my_filter.lib
my_filter.h
test.py
```

## 手動執行

在 GitHub：

```text
Actions
→ Build Changed CUDA Projects
→ Run workflow
```

手動執行會編譯 `cuda_projects/` 下的所有子專案。

## 統一匯出巨集

每個 `.h` 建議使用：

```cpp
#ifdef _WIN32
    #ifdef CUDA_DLL_EXPORTS
        #define CUDA_DLL_API extern "C" __declspec(dllexport)
    #else
        #define CUDA_DLL_API extern "C" __declspec(dllimport)
    #endif
#else
    #define CUDA_DLL_API extern "C"
#endif
```

workflow 已經自動加入：

```text
-DCUDA_DLL_EXPORTS
```

## 目前固定設定

```text
Windows runner: windows-2022
CUDA Toolkit: 12.6.3
GPU architecture: sm_86
C++ standard: C++17
```

`sm_86` 適用 RTX 3090。

## 專案自訂 build script

若 `cuda_projects/<project>/build_cuda_dll.ps1` 存在，workflow 會優先執行該腳本，並傳入：

```powershell
-Architecture sm_86 -OutputDirectory build\<project>
```

這適合 DLL source 與測試 `.cu` 必須分開編譯的專案，例如 `visionflow_cuda`。沒有自訂腳本的舊專案仍走原本的通用 DLL build。
