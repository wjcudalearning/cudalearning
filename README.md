# GitHub Actions：將 CUDA `.cu` 編譯成 Windows `.dll`

這是一個最小可執行範例：

1. GitHub Actions 使用 `windows-2022` runner。
2. 安裝 CUDA Toolkit 12.6.3。
3. 使用 `nvcc` 將 `src/cuda_vector_add.cu` 編譯成 `cuda_vector_add.dll`。
4. 使用 `dumpbin /exports` 驗證匯出函式。
5. 將 DLL、標頭檔與 Python 測試檔上傳為 GitHub Actions artifact。
6. 下載 artifact 到有 NVIDIA GPU 的 Windows 電腦，使用 Python `ctypes` 執行測試。

## 專案結構

```text
.
├─ .github/
│  └─ workflows/
│     └─ build-cuda-dll.yml
├─ include/
│  └─ cuda_vector_add.h
├─ src/
│  └─ cuda_vector_add.cu
├─ python/
│  └─ test_cuda_dll.py
├─ .gitignore
└─ README.md
```

## 在 GitHub 執行

1. 建立一個 GitHub repository。
2. 將此專案所有檔案上傳並 push。
3. 進入 repository 的 **Actions** 頁面。
4. 選擇 **Build CUDA DLL**。
5. 按下 **Run workflow**。
6. Workflow 成功後，在頁面下方下載：
   `cuda-vector-add-windows-x64-sm86`

## 在本機測試

解壓縮 artifact，確認以下兩個檔案在同一資料夾：

```text
cuda_vector_add.dll
test_cuda_dll.py
```

執行：

```powershell
python test_cuda_dll.py
```

預期輸出大致如下：

```text
偵測到 CUDA GPU 數量：1
輸入 A： [1.0, 2.0, 3.0, 4.0]
輸入 B： [10.0, 20.0, 30.0, 40.0]
GPU 結果： [11.0, 22.0, 33.0, 44.0]
測試成功：.cu → .dll → Python ctypes → CUDA kernel
```

## 注意

- `-arch=sm_86` 是針對 RTX 3090/Ampere。
- 一般 GitHub-hosted Windows runner 用來編譯，不保證有可用 NVIDIA GPU，因此 workflow 不直接執行 kernel。
- DLL 採用 C ABI (`extern "C"`) 匯出，較容易從 Python、C#、C++ 呼叫。
- `--cudart static` 可降低對額外 CUDA runtime DLL 的依賴；執行端仍需 NVIDIA 顯示卡驅動。
