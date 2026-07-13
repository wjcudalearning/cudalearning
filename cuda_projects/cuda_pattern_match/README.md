# cuda_pattern_match 3.0 批次比對與切圖版

不使用 OpenCV 的 CUDA Pattern Match DLL 與 PySide6 GUI。

核心比對仍使用 GPU 金字塔 coarse-to-fine ZNCC；3.0 新增資料夾批次處理、
結果 CSV，以及 CPU/GPU 固定尺寸 ROI 批量切圖。

## 功能

### Pattern Match

- 自製 CUDA ZNCC，不依賴 OpenCV / cv2
- 8-bit 灰階大圖與 Template
- 支援 204,800,000 px 以上工業大圖
- GPU 影像金字塔粗搜、局部極大值、逐層精搜
- 多實例匹配與 IoU NMS
- `Auto / 128 / 256 / 512 / 1024 threads per block`
- PySide6 背景執行緒，GUI 不會在 CUDA 運算時凍結
- 大圖縮圖預覽、匹配框、排名、分數與詳細耗時

### 資料夾批次

- 單張圖片與資料夾批次兩種模式
- 可選擇是否包含子資料夾
- DLL 與 Template 在整批工作中只由 Python 載入一次
- 單張失敗不會中止整批；錯誤會記錄在批次總表
- 可在目前影像完成後停止批次
- 預設不逐張更新 4096 px 大圖預覽，避免預覽拖慢批次
- 批次完成後顯示最後一張成功影像

### 批量切小圖

- 每個匹配結果自動切出一張固定尺寸灰階小圖
- 切圖尺寸：`Template 尺寸 + 左右邊界 + 上下邊界`
- ROI 靠近大圖邊緣時以指定灰階值補齊，所有輸出尺寸一致
- 後端可選：
  - `Auto`：少量小 ROI 用 CPU，大量 ROI 才用 GPU
  - `強制 GPU`
  - `強制 CPU`
- GPU 版本一次上傳大圖，所有 ROI 使用同一個 CUDA kernel 平行擷取
- 輸出格式：PNG、TIFF LZW、JPEG
- 檔名包含來源影像、排名、座標與匹配分數

> 目前比對與切圖都使用 8-bit 灰階資料。若原圖是 RGB，輸出小圖也是灰階。

## GPU 切圖的限制

GPU 可以加速 ROI 像素擷取，但以下工作仍在 CPU：

1. Pillow 解碼來源 PNG/JPEG/TIFF。
2. PNG/JPEG/TIFF 壓縮編碼。
3. 寫入硬碟。

目前 GPU 切圖是獨立 DLL 呼叫，因此在 Pattern Match 完成後，需要為切圖再上傳一次完整大圖。
少量小 ROI 時，這個傳輸成本可能高於 CPU 直接切圖，所以 `Auto` 不會一律使用 GPU。
GUI 耗時表會把 ROI 擷取與編碼/寫入分開顯示。

## 輸出結構

假設輸出資料夾為 `output`，來源影像是 `line_001.tif`：

```text
output/
├─ batch_summary.csv
└─ line_001/
   ├─ matches.csv
   └─ crops/
      ├─ line_001__match_00001__x120_y450__s0.98321.png
      └─ line_001__match_00002__x860_y450__s0.97654.png
```

若啟用遞迴批次，輸出會保留來源的相對子資料夾結構。

### `matches.csv`

每張來源影像各有一份，包含：

- 來源影像
- 排名
- X / Y
- 寬 / 高
- 分數
- 切圖後端
- 切圖輸出路徑

### `batch_summary.csv`

包含每張影像的：

- 成功 / 失敗狀態
- 影像尺寸
- 匹配數與切圖數
- CPU / GPU 切圖後端
- 讀圖、Pattern Match、ROI 擷取、存檔及總耗時
- 錯誤訊息

## 安裝與執行

```powershell
py -m pip install -r requirements.txt
py test.py
```

GitHub Actions artifact 中應包含：

```text
cuda_pattern_match.dll
cuda_pattern_match.h
test.py
requirements.txt
README.md
VERSION.txt
```

## 建議起始參數

對 204,800,000 px 大圖：

```text
最終分數門檻       0.85
粗搜分數門檻       0.70
金字塔粗搜倍率     Auto
每層精搜半徑       2 px
CUDA threads/block Auto
NMS IoU            0.25
GPU 候選容量        100000
```

切圖建議先使用：

```text
擷取後端           Auto
左右額外邊界       0～50 px
上下額外邊界       0～50 px
邊界補值           0
格式                 PNG 或 TIFF
```

## 關於 threads/block 與 VRAM

VRAM 使用率低，不代表應直接把 block 設為 1024。threads/block 影響的是：

- SM occupancy
- register 壓力
- 同時可駐留的 block 數
- shared memory reduction

預設 `Auto` 使用 256 threads，通常比強制 1024 穩定。GPU ROI 切圖也會沿用此設定。

## 目前限制

- Pattern Match 只處理平移，不處理旋轉、縮放、透視或非剛性變形。
- 切圖輸出目前是 8-bit 灰階。
- 批次處理目前是逐張大圖執行，以控制 RAM 與 VRAM 峰值；不是同時將多張 200 MP 大圖放入 GPU。
- 停止按鈕會在目前影像完成後生效，無法中斷正在執行的單次 CUDA kernel。
