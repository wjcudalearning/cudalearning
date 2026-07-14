from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "05_template_matching.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    dll.cuda_template_match_best_ssd_u8.argtypes = [
        u8_p,
        ctypes.c_int,
        ctypes.c_int,
        u8_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.cuda_template_match_best_ssd_u8.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    image_width, image_height = 16, 12
    template_width, template_height = 3, 3
    expected_x, expected_y = 7, 5
    image = [10] * (image_width * image_height)
    templ = [0, 50, 0, 50, 255, 50, 0, 50, 0]
    for y in range(template_height):
        for x in range(template_width):
            image[(expected_y + y) * image_width + expected_x + x] = (
                templ[y * template_width + x]
            )

    ImageArray = ctypes.c_ubyte * len(image)
    TemplateArray = ctypes.c_ubyte * len(templ)
    best_x = ctypes.c_int()
    best_y = ctypes.c_int()
    best_score = ctypes.c_float()

    code = dll.cuda_template_match_best_ssd_u8(
        ImageArray(*image),
        image_width,
        image_height,
        TemplateArray(*templ),
        template_width,
        template_height,
        ctypes.byref(best_x),
        ctypes.byref(best_y),
        ctypes.byref(best_score),
    )
    if code != 0:
        raw = dll.cuda_error_string(code)
        message = raw.decode("utf-8", errors="replace") if raw else "Unknown CUDA error"
        print(f"CUDA 錯誤 {code}：{message}")
        return 3

    print(
        f"最佳位置：x={best_x.value}, y={best_y.value}, "
        f"mean SSD={best_score.value:.6f}"
    )
    if (best_x.value, best_y.value) != (expected_x, expected_y):
        return 4
    if abs(best_score.value) > 1e-6:
        return 5

    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
