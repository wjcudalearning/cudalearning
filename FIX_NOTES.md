# 修正版說明

`detect-projects` 已由 Ubuntu PowerShell 改為 Bash。

主要變更：

- 不再呼叫 `.NET JsonSerializer.Serialize()`。
- 使用 `jq` 固定輸出 JSON 陣列。
- 沒有任何異動專案時輸出 `[]`。
- 手動執行、首次 push 或 workflow 本身異動時編譯全部專案。
- 一般 push 只編譯有異動的 `cuda_projects/<專案>/`。
- 已刪除的專案資料夾會被忽略。

請用此檔覆蓋：

```text
.github/workflows/build-changed-cuda-projects.yml
```

然後 commit 與 push。

## VisionFlow custom-build support

- Added `cuda_projects/visionflow_cuda/`.
- Projects containing `build_cuda_dll.ps1` now use that script instead of the generic `*.cu` DLL build.
- This prevents `test_cuda_api.cu` from being linked into `visionflow_cuda.dll`.
- Project-local `include/` folders and native smoke artifacts are packaged recursively.
