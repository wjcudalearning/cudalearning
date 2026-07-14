from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "04_morphology.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    signature = [u8_p, u8_p, ctypes.c_int, ctypes.c_int]
    dll.cuda_dilate_3x3_u8.argtypes = signature
    dll.cuda_dilate_3x3_u8.restype = ctypes.c_int
    dll.cuda_erode_3x3_u8.argtypes = signature
    dll.cuda_erode_3x3_u8.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    width, height = 9, 7
    count = width * height
    Array = ctypes.c_ubyte * count

    dilation_input = [0] * count
    dilation_input[3 * width + 4] = 255
    dilation_output = Array()
    code = dll.cuda_dilate_3x3_u8(Array(*dilation_input), dilation_output, width, height)
    if code != 0:
        print("膨脹執行失敗。")
        return 3
    expected_dilation = [0] * count
    for y in range(2, 5):
        for x in range(3, 6):
            expected_dilation[y * width + x] = 255
    if list(dilation_output) != expected_dilation:
        print("膨脹結果不正確。")
        return 4

    erosion_input = [255] * count
    erosion_input[3 * width + 4] = 0
    erosion_output = Array()
    code = dll.cuda_erode_3x3_u8(Array(*erosion_input), erosion_output, width, height)
    if code != 0:
        print("侵蝕執行失敗。")
        return 5
    expected_erosion = [255] * count
    for y in range(2, 5):
        for x in range(3, 6):
            expected_erosion[y * width + x] = 0
    if list(erosion_output) != expected_erosion:
        print("侵蝕結果不正確。")
        return 6

    print("膨脹與侵蝕測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
