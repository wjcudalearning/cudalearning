from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "06_roi_batch.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    dll.cuda_roi_batch_threshold_u8.argtypes = [
        u8_p,
        u8_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ubyte,
    ]
    dll.cuda_roi_batch_threshold_u8.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    roi_width, roi_height, roi_count = 8, 8, 4
    roi_pixels = roi_width * roi_height
    values = [
        roi * 60 + pixel % 16
        for roi in range(roi_count)
        for pixel in range(roi_pixels)
    ]
    expected = [255 if value > 128 else 0 for value in values]

    Array = ctypes.c_ubyte * len(values)
    source = Array(*values)
    output = Array()
    code = dll.cuda_roi_batch_threshold_u8(
        source, output, roi_width, roi_height, roi_count, 128
    )
    if code != 0:
        raw = dll.cuda_error_string(code)
        message = raw.decode("utf-8", errors="replace") if raw else "Unknown CUDA error"
        print(f"CUDA 錯誤 {code}：{message}")
        return 3

    result = list(output)
    if result != expected:
        print("批次 ROI 結果不正確。")
        return 4

    for roi in range(roi_count):
        chunk = result[roi * roi_pixels : (roi + 1) * roi_pixels]
        print(f"ROI {roi}：white pixels={chunk.count(255)}/{roi_pixels}")

    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
