# cuda_pattern_match 正式版

不使用 OpenCV 的 CUDA Pattern Match DLL 與 PySide6 GUI。

## 核心改版

舊版是直接空間域暴力 ZNCC：每一個原圖搜尋位置都重新掃過完整 Template，
運算量約為：

```text
原圖搜尋位置數 × Template 像素數
```

正式版改成 GPU 金字塔 coarse-to-fine ZNCC：

1. GPU 建立 2× 影像金字塔。
2. 在最粗層逐像素建立完整 ZNCC score map。
3. GPU 只留下 3×3 局部極大值。
4. 候選點逐層放大，每層只搜尋附近的小範圍。
5. 回到原圖解析度後才套用最終分數門檻與 NMS。

若實際使用 8× 金字塔，粗搜階段的理論乘加工作量約是舊版的
`1 / 8^4 = 1 / 4096`。精搜只針對局部極大值，不再掃完整原圖。

## 功能

- 自製 CUDA ZNCC，不依賴 OpenCV / cv2
- 8-bit 灰階大圖與 Template
- 支援 204,800,000 px 以上工業大圖
- 多實例匹配、局部極大值、IoU NMS
- GPU 影像金字塔與逐層精搜
- 可選 `Auto / 128 / 256 / 512 / 1024 threads per block`
- 自動依 Template 尺寸降低金字塔倍率，避免 Template 縮得太小
- 精搜 kernel 自動分批，降低 Windows TDR 風險
- PySide6 背景執行緒，運算時 GUI 不凍結
- 大圖預覽、框選結果、排名與分數
- 可另存完整解析度標註圖
- 詳細耗時、搜尋位置數、kernel launch 次數與 VRAM 用量

## 安裝

```powershell
py -m pip install -r requirements.txt
```

## 執行

將 GitHub Actions artifact 解壓後，確認以下檔案在同一資料夾：

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

### 金字塔倍率

- `Auto`：建議。選擇 Template 仍至少約 8×8 px 的最大 2 次方倍率。
- `4×`：較保守，速度較慢但對細小紋理更穩。
- `8×`：一般大圖常用。
- `16× / 32×`：大型 Template 或極大影像使用。

若 Template 太小，DLL 會自動降低實際倍率；GUI 耗時表會顯示要求倍率與實際倍率。

### 粗搜門檻

粗搜影像經過下採樣，分數可能比原圖略低，因此粗搜門檻應低於最終門檻。
一般可先設：

```text
粗搜門檻 = 最終門檻 - 0.10 ~ 0.20
```

粗搜門檻太高可能漏檢；太低會增加精搜候選數。

### threads per block

VRAM 使用率低不代表 block 應直接開到 1024。threads/block 主要影響：

- SM occupancy
- register 壓力
- block 排程數量
- reduction shared memory

它與「24 GB VRAM 是否用滿」沒有直接比例關係。`Auto` 目前採用 256 threads，
通常比強制 1024 更穩定。GUI 仍保留手動選項，可用耗時表實測 256 / 512 / 1024。

## 耗時表

GUI 會列出：

- Template 金字塔預處理
- Host → Device
- GPU 建立影像金字塔
- 粗層 ZNCC score map
- 粗層局部極大值
- GPU 逐層精搜
- Device → Host
- CPU 排序與 NMS
- DLL 總耗時
- 粗搜／精搜位置數
- 實際金字塔倍率
- 各 kernel threads/block
- kernel launch 次數
- DLL 估計峰值 VRAM

## 候選溢位

若「候選容量溢位」顯示為「是」，粗層局部極大值數量超過 GPU 候選容量。
可採取：

1. 提高粗搜分數門檻。
2. 提高 GPU 候選容量。
3. 提高金字塔倍率。

## 精度限制

目前只處理平移，不處理旋轉與縮放差異。Template 與目標若存在明顯旋轉、比例改變、
透視變形或非剛性變形，需要再加入多角度／多尺度 Template 金字塔。
