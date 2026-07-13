# cuda_pattern_match

不使用 OpenCV 的 CUDA Pattern Match DLL 與 PySide6 GUI。

## 功能

- 自製 CUDA ZNCC（Zero-mean Normalized Cross-Correlation）
- 8-bit 灰階大圖與 Template
- 分數門檻篩選多個候選位置
- CPU 排序與 IoU NMS，輸出多個不重複框
- 依 Template 大小自動分批 launch，降低 Windows TDR 風險
- PySide6 GUI 選擇 DLL、大圖、Template
- 大圖上顯示結果框、排名與分數
- 結果座標表
- Python、H2D、CUDA kernel、D2H、排序/NMS、GUI 繪圖等分段耗時
- 可另存標註結果

## 安裝 GUI 套件

```powershell
py -m pip install -r requirements.txt
```

## 執行

把 GitHub Actions artifact 解壓縮後，確認以下檔案在同一資料夾：

```text
cuda_pattern_match.dll
cuda_pattern_match.h
test.py
requirements.txt
```

執行：

```powershell
py test.py
```

## 產生測試圖片

```powershell
py make_demo_images.py
```

會在 `demo_data/` 產生一張大圖與一張 Template，內含 5 個亮度略有差異的相同圖樣。

## 參數

- `分數門檻`：0~1，越高越嚴格，建議先從 `0.85` 開始。
- `搜尋步距`：`1` 是逐像素搜尋；大圖測試時可先用 `2~4`。
- `NMS IoU`：重疊框抑制門檻，建議 `0.2~0.4`。
- `最多結果數`：最後輸出的框數上限。
- `GPU 候選容量`：門檻以上位置的 GPU 暫存容量。顯示溢位時，提高門檻或容量。

## 目前版本限制

這是直接空間域 ZNCC：每個搜尋位置都會掃描完整 Template，運算量約為：

```text
搜尋位置數 × Template 像素數
```

因此大圖配大 Template 時，請先提高 `搜尋步距` 驗證流程。後續可再做 shared-memory、積分圖、多尺度 coarse-to-fine 或 FFT 版本。
