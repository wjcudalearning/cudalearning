from __future__ import annotations

import csv
import ctypes
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

# 工業大圖可能遠超過 Pillow 預設的 decompression-bomb 像素上限。
# 此 GUI 只開啟使用者主動選取的本機影像，因此停用該限制。
Image.MAX_IMAGE_PIXELS = None

from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class MatchResult(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("score", ctypes.c_float),
    ]


class MatchTiming(ctypes.Structure):
    _fields_ = [
        ("host_prepare_ms", ctypes.c_double),
        ("host_to_device_ms", ctypes.c_double),
        ("pyramid_build_ms", ctypes.c_double),
        ("coarse_score_ms", ctypes.c_double),
        ("coarse_peak_ms", ctypes.c_double),
        ("refine_ms", ctypes.c_double),
        ("device_to_host_ms", ctypes.c_double),
        ("sort_nms_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("coarse_evaluated_positions", ctypes.c_longlong),
        ("refine_evaluated_positions", ctypes.c_longlong),
        ("raw_candidate_count", ctypes.c_int),
        ("stored_candidate_count", ctypes.c_int),
        ("result_count", ctypes.c_int),
        ("candidate_overflow", ctypes.c_int),
        ("kernel_launch_count", ctypes.c_int),
        ("requested_pyramid_factor", ctypes.c_int),
        ("actual_pyramid_factor", ctypes.c_int),
        ("pyramid_level_count", ctypes.c_int),
        ("coarse_image_width", ctypes.c_int),
        ("coarse_image_height", ctypes.c_int),
        ("coarse_template_width", ctypes.c_int),
        ("coarse_template_height", ctypes.c_int),
        ("coarse_threads_per_block", ctypes.c_int),
        ("peak_threads_per_block", ctypes.c_int),
        ("refine_threads_per_block", ctypes.c_int),
        ("device_total_vram_mib", ctypes.c_double),
        ("device_free_vram_before_mib", ctypes.c_double),
        ("device_free_vram_after_alloc_mib", ctypes.c_double),
        ("estimated_vram_used_mib", ctypes.c_double),
    ]


class CropRoi(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
    ]


class CropTiming(ctypes.Structure):
    _fields_ = [
        ("host_prepare_ms", ctypes.c_double),
        ("host_to_device_ms", ctypes.c_double),
        ("kernel_ms", ctypes.c_double),
        ("device_to_host_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("output_pixel_count", ctypes.c_longlong),
        ("roi_count", ctypes.c_int),
        ("crop_width", ctypes.c_int),
        ("crop_height", ctypes.c_int),
        ("threads_per_block", ctypes.c_int),
        ("kernel_launch_count", ctypes.c_int),
        ("device_total_vram_mib", ctypes.c_double),
        ("device_free_vram_before_mib", ctypes.c_double),
        ("device_free_vram_after_alloc_mib", ctypes.c_double),
        ("estimated_vram_used_mib", ctypes.c_double),
    ]


@dataclass(frozen=True)
class AlgorithmSettings:
    threshold: float
    coarse_threshold: float
    pyramid_factor: int
    refine_radius: int
    nms_iou: float
    max_results: int
    max_candidates: int
    block_threads: int


@dataclass(frozen=True)
class CropSettings:
    enabled: bool
    backend: str
    margin_x: int
    margin_y: int
    fill_value: int
    output_format: str
    jpeg_quality: int


@dataclass(frozen=True)
class JobParameters:
    dll_path: str
    template_path: str
    mode: str
    input_path: str
    output_dir: str
    recursive: bool
    save_csv: bool
    preview_each: bool
    algorithm: AlgorithmSettings
    crop: CropSettings


class CudaPatternMatcher:
    def __init__(self, dll_path: Path) -> None:
        if not dll_path.exists():
            raise FileNotFoundError(f"找不到 DLL：{dll_path}")

        try:
            self.dll = ctypes.CDLL(str(dll_path))
        except OSError as exc:
            raise RuntimeError(f"DLL 載入失敗：{exc}") from exc

        try:
            self.match_function = self.dll.cuda_pattern_match_pyramid_zncc_u8
        except AttributeError as exc:
            raise RuntimeError(
                "這是舊版 DLL，缺少 cuda_pattern_match_pyramid_zncc_u8。"
                "請重新下載或重新編譯目前版本 DLL。"
            ) from exc

        self.match_function.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(MatchResult),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(MatchTiming),
        ]
        self.match_function.restype = ctypes.c_int

        self.crop_function = getattr(self.dll, "cuda_crop_rois_u8", None)
        if self.crop_function is not None:
            self.crop_function.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(CropRoi),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_ubyte,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
                ctypes.POINTER(CropTiming),
            ]
            self.crop_function.restype = ctypes.c_int

        self.dll.cuda_pattern_match_error_string.argtypes = [ctypes.c_int]
        self.dll.cuda_pattern_match_error_string.restype = ctypes.c_char_p

    @property
    def supports_gpu_crop(self) -> bool:
        return self.crop_function is not None

    def error_string(self, code: int) -> str:
        raw = self.dll.cuda_pattern_match_error_string(code)
        return raw.decode("utf-8", errors="replace") if raw else "Unknown error"

    def match(
        self,
        image: np.ndarray,
        template: np.ndarray,
        settings: AlgorithmSettings,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if image.ndim != 2 or template.ndim != 2:
            raise ValueError("DLL 僅接受 8-bit 單通道影像。")

        image = np.ascontiguousarray(image, dtype=np.uint8)
        template = np.ascontiguousarray(template, dtype=np.uint8)

        result_buffer = (MatchResult * settings.max_results)()
        result_count = ctypes.c_int(0)
        timing = MatchTiming()

        code = self.match_function(
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            int(image.shape[1]),
            int(image.shape[0]),
            int(image.strides[0]),
            template.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            int(template.shape[1]),
            int(template.shape[0]),
            int(template.strides[0]),
            ctypes.c_float(settings.threshold),
            ctypes.c_float(settings.coarse_threshold),
            settings.pyramid_factor,
            settings.refine_radius,
            ctypes.c_float(settings.nms_iou),
            settings.max_results,
            settings.max_candidates,
            settings.block_threads,
            result_buffer,
            ctypes.byref(result_count),
            ctypes.byref(timing),
        )

        if code != 0:
            raise RuntimeError(f"CUDA DLL 錯誤 {code}：{self.error_string(code)}")

        results = [
            {
                "rank": index + 1,
                "x": int(result_buffer[index].x),
                "y": int(result_buffer[index].y),
                "width": int(result_buffer[index].width),
                "height": int(result_buffer[index].height),
                "score": float(result_buffer[index].score),
                "crop_path": "",
            }
            for index in range(result_count.value)
        ]

        timing_data: dict[str, Any] = {
            "DLL：Template 金字塔預處理": timing.host_prepare_ms,
            "GPU：Host → Device": timing.host_to_device_ms,
            "GPU：建立影像金字塔": timing.pyramid_build_ms,
            "GPU：粗層 ZNCC score map": timing.coarse_score_ms,
            "GPU：粗層局部極大值": timing.coarse_peak_ms,
            "GPU：逐層精搜": timing.refine_ms,
            "GPU：Device → Host": timing.device_to_host_ms,
            "CPU：排序與 NMS": timing.sort_nms_ms,
            "DLL：Pattern Match 總耗時": timing.total_ms,
            "粗搜位置數": int(timing.coarse_evaluated_positions),
            "精搜位置數": int(timing.refine_evaluated_positions),
            "粗搜候選總數": int(timing.raw_candidate_count),
            "實際精搜候選數": int(timing.stored_candidate_count),
            "最終結果數": int(timing.result_count),
            "候選容量溢位": bool(timing.candidate_overflow),
            "Pattern Match kernel launch 次數": int(timing.kernel_launch_count),
            "要求金字塔倍率": (
                "Auto"
                if timing.requested_pyramid_factor == 0
                else int(timing.requested_pyramid_factor)
            ),
            "實際金字塔倍率": int(timing.actual_pyramid_factor),
            "金字塔層數": int(timing.pyramid_level_count),
            "粗層大圖尺寸": (
                f"{timing.coarse_image_width} × {timing.coarse_image_height}"
            ),
            "粗層 Template 尺寸": (
                f"{timing.coarse_template_width} × {timing.coarse_template_height}"
            ),
            "粗搜 threads/block": int(timing.coarse_threads_per_block),
            "極大值 threads/block": int(timing.peak_threads_per_block),
            "精搜 threads/block": int(timing.refine_threads_per_block),
            "GPU 總 VRAM": f"{timing.device_total_vram_mib:,.1f} MiB",
            "執行前可用 VRAM": f"{timing.device_free_vram_before_mib:,.1f} MiB",
            "配置後最低可用 VRAM": (
                f"{timing.device_free_vram_after_alloc_mib:,.1f} MiB"
            ),
            "Pattern Match 估計峰值 VRAM": (
                f"{timing.estimated_vram_used_mib:,.1f} MiB"
            ),
        }
        return results, timing_data

    def crop_gpu(
        self,
        image: np.ndarray,
        origins: list[tuple[int, int]],
        crop_width: int,
        crop_height: int,
        fill_value: int,
        block_threads: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.crop_function is None:
            raise RuntimeError(
                "目前 DLL 沒有 cuda_crop_rois_u8；請重新編譯批次切圖版本。"
            )
        if not origins:
            return np.empty((0, crop_height, crop_width), dtype=np.uint8), {}

        image = np.ascontiguousarray(image, dtype=np.uint8)
        roi_buffer = (CropRoi * len(origins))(
            *(CropRoi(int(x), int(y)) for x, y in origins)
        )
        crops = np.empty(
            (len(origins), crop_height, crop_width),
            dtype=np.uint8,
        )
        timing = CropTiming()

        code = self.crop_function(
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            int(image.shape[1]),
            int(image.shape[0]),
            int(image.strides[0]),
            roi_buffer,
            len(origins),
            crop_width,
            crop_height,
            ctypes.c_ubyte(fill_value),
            block_threads,
            crops.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.c_size_t(crops.nbytes),
            ctypes.byref(timing),
        )
        if code != 0:
            raise RuntimeError(f"GPU 切圖錯誤 {code}：{self.error_string(code)}")

        timing_data: dict[str, Any] = {
            "GPU 切圖：Host 準備": timing.host_prepare_ms,
            "GPU 切圖：Host → Device": timing.host_to_device_ms,
            "GPU 切圖：ROI kernel": timing.kernel_ms,
            "GPU 切圖：Device → Host": timing.device_to_host_ms,
            "GPU 切圖：DLL 總耗時": timing.total_ms,
            "GPU 切圖：輸出像素數": int(timing.output_pixel_count),
            "GPU 切圖：ROI 數": int(timing.roi_count),
            "GPU 切圖：尺寸": f"{timing.crop_width} × {timing.crop_height}",
            "GPU 切圖：threads/block": int(timing.threads_per_block),
            "GPU 切圖：kernel launch 次數": int(timing.kernel_launch_count),
            "GPU 切圖：估計峰值 VRAM": (
                f"{timing.estimated_vram_used_mib:,.1f} MiB"
            ),
        }
        return crops, timing_data


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        return np.ascontiguousarray(np.asarray(grayscale), dtype=np.uint8)


def crop_cpu(
    image: np.ndarray,
    origins: Iterable[tuple[int, int]],
    crop_width: int,
    crop_height: int,
    fill_value: int,
) -> np.ndarray:
    origin_list = list(origins)
    crops = np.full(
        (len(origin_list), crop_height, crop_width),
        fill_value,
        dtype=np.uint8,
    )
    image_height, image_width = image.shape

    for index, (origin_x, origin_y) in enumerate(origin_list):
        source_x1 = max(0, origin_x)
        source_y1 = max(0, origin_y)
        source_x2 = min(image_width, origin_x + crop_width)
        source_y2 = min(image_height, origin_y + crop_height)
        if source_x1 >= source_x2 or source_y1 >= source_y2:
            continue

        destination_x1 = source_x1 - origin_x
        destination_y1 = source_y1 - origin_y
        destination_x2 = destination_x1 + (source_x2 - source_x1)
        destination_y2 = destination_y1 + (source_y2 - source_y1)
        crops[
            index,
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = image[source_y1:source_y2, source_x1:source_x2]

    return crops


def choose_crop_backend(
    requested: str,
    matcher: CudaPatternMatcher,
    image: np.ndarray,
    roi_count: int,
    crop_width: int,
    crop_height: int,
) -> str:
    if requested == "cpu":
        return "CPU"
    if requested == "gpu":
        if not matcher.supports_gpu_crop:
            raise RuntimeError("已指定 GPU 切圖，但目前 DLL 不支援 GPU 切圖。")
        return "GPU"

    # 獨立切圖函式需要重新上傳一次完整大圖。少量小 ROI 時，CPU 只複製
    # ROI 區域通常更快；輸出量夠大時才自動切到 GPU。
    output_bytes = roi_count * crop_width * crop_height
    image_bytes = int(image.nbytes)
    large_batch = roi_count >= 64 and output_bytes >= 16 * 1024 * 1024
    output_is_significant = output_bytes >= image_bytes // 8
    if matcher.supports_gpu_crop and (large_batch or output_is_significant):
        return "GPU"
    return "CPU"


def safe_component(text: str) -> str:
    forbidden = '<>:"/\\|?*'
    output = "".join("_" if char in forbidden else char for char in text).strip()
    return output or "image"


def save_crop(
    array: np.ndarray,
    path: Path,
    output_format: str,
    jpeg_quality: int,
) -> None:
    image = Image.fromarray(array, mode="L")
    kwargs: dict[str, Any] = {}
    if output_format == "JPEG":
        kwargs.update(quality=jpeg_quality, optimize=False)
    elif output_format == "PNG":
        kwargs.update(compress_level=3)
    elif output_format == "TIFF":
        kwargs.update(compression="tiff_lzw")
    image.save(path, format=output_format, **kwargs)


def output_extension(output_format: str) -> str:
    return {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif"}[output_format]


def write_matches_csv(
    csv_path: Path,
    source_path: Path,
    results: list[dict[str, Any]],
    crop_backend: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_image",
                "rank",
                "x",
                "y",
                "width",
                "height",
                "score",
                "crop_backend",
                "crop_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source_image": str(source_path),
                    "rank": result["rank"],
                    "x": result["x"],
                    "y": result["y"],
                    "width": result["width"],
                    "height": result["height"],
                    "score": f"{result['score']:.8f}",
                    "crop_backend": crop_backend,
                    "crop_path": result.get("crop_path", ""),
                }
            )


def write_batch_summary(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "source_image",
        "image_width",
        "image_height",
        "match_count",
        "crop_count",
        "crop_backend",
        "image_load_ms",
        "pattern_match_ms",
        "crop_extract_ms",
        "crop_save_ms",
        "total_ms",
        "message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ProcessingWorker(QObject):
    progress = Signal(int, int, str)
    item_finished = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parameters: JobParameters) -> None:
        super().__init__()
        self.parameters = parameters
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _collect_images(self) -> list[Path]:
        input_path = Path(self.parameters.input_path)
        template_path = Path(self.parameters.template_path).resolve()
        output_path = (
            Path(self.parameters.output_dir).resolve()
            if self.parameters.output_dir
            else None
        )

        if self.parameters.mode == "single":
            return [input_path]

        iterator = input_path.rglob("*") if self.parameters.recursive else input_path.glob("*")
        images: list[Path] = []
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved == template_path:
                continue
            if output_path is not None:
                try:
                    resolved.relative_to(output_path)
                    continue
                except ValueError:
                    pass
            images.append(path)
        images.sort(key=lambda item: str(item).casefold())
        return images

    def _output_base(self, source_path: Path) -> Path | None:
        if not self.parameters.output_dir:
            return None
        output_root = Path(self.parameters.output_dir)
        if self.parameters.mode == "folder":
            input_root = Path(self.parameters.input_path)
            try:
                relative_parent = source_path.parent.relative_to(input_root)
            except ValueError:
                relative_parent = Path()
            return output_root / relative_parent / safe_component(source_path.stem)
        return output_root / safe_component(source_path.stem)

    def _process_one(
        self,
        matcher: CudaPatternMatcher,
        template: np.ndarray,
        source_path: Path,
        common_timing: dict[str, Any],
    ) -> dict[str, Any]:
        total_start = time.perf_counter()

        load_start = time.perf_counter()
        image = load_grayscale(source_path)
        image_load_ms = (time.perf_counter() - load_start) * 1000.0

        match_start = time.perf_counter()
        results, timing = matcher.match(
            image,
            template,
            self.parameters.algorithm,
        )
        python_match_ms = (time.perf_counter() - match_start) * 1000.0

        crop_backend = "未啟用"
        crop_extract_ms = 0.0
        crop_save_ms = 0.0
        crop_timing: dict[str, Any] = {}
        output_base = self._output_base(source_path)

        if self.parameters.crop.enabled and results:
            crop_width = int(template.shape[1]) + self.parameters.crop.margin_x * 2
            crop_height = int(template.shape[0]) + self.parameters.crop.margin_y * 2
            origins = [
                (
                    int(result["x"]) - self.parameters.crop.margin_x,
                    int(result["y"]) - self.parameters.crop.margin_y,
                )
                for result in results
            ]
            crop_backend = choose_crop_backend(
                self.parameters.crop.backend,
                matcher,
                image,
                len(results),
                crop_width,
                crop_height,
            )

            crop_start = time.perf_counter()
            if crop_backend == "GPU":
                crops, crop_timing = matcher.crop_gpu(
                    image,
                    origins,
                    crop_width,
                    crop_height,
                    self.parameters.crop.fill_value,
                    self.parameters.algorithm.block_threads,
                )
            else:
                crops = crop_cpu(
                    image,
                    origins,
                    crop_width,
                    crop_height,
                    self.parameters.crop.fill_value,
                )
            crop_extract_ms = (time.perf_counter() - crop_start) * 1000.0

            if output_base is None:
                raise RuntimeError("已啟用切圖，但沒有設定輸出資料夾。")
            crops_dir = output_base / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            extension = output_extension(self.parameters.crop.output_format)

            save_start = time.perf_counter()
            for index, (crop, result) in enumerate(zip(crops, results), start=1):
                filename = (
                    f"{safe_component(source_path.stem)}__match_{index:05d}"
                    f"__x{result['x']}_y{result['y']}"
                    f"__s{result['score']:.5f}{extension}"
                )
                crop_path = crops_dir / filename
                save_crop(
                    crop,
                    crop_path,
                    self.parameters.crop.output_format,
                    self.parameters.crop.jpeg_quality,
                )
                result["crop_path"] = str(crop_path)
            crop_save_ms = (time.perf_counter() - save_start) * 1000.0

        if self.parameters.save_csv and output_base is not None:
            write_matches_csv(
                output_base / "matches.csv",
                source_path,
                results,
                crop_backend,
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        timing = {
            **common_timing,
            "Python：讀取大圖並轉灰階": image_load_ms,
            **timing,
            "Python：Pattern Match ctypes 區段": python_match_ms,
            **crop_timing,
            "切圖：擷取總耗時": crop_extract_ms,
            "切圖：編碼與寫入硬碟": crop_save_ms,
            "單張完整流程總耗時": total_ms,
        }

        payload = {
            "status": "成功",
            "source_path": str(source_path),
            "parameters": asdict(self.parameters),
            "image_size": (int(image.shape[1]), int(image.shape[0])),
            "template_size": (int(template.shape[1]), int(template.shape[0])),
            "results": results,
            "timing": timing,
            "crop_backend": crop_backend,
            "output_base": str(output_base) if output_base else "",
        }

        del image
        return payload

    @Slot()
    def run(self) -> None:
        whole_start = time.perf_counter()
        try:
            load_dll_start = time.perf_counter()
            matcher = CudaPatternMatcher(Path(self.parameters.dll_path))
            load_dll_ms = (time.perf_counter() - load_dll_start) * 1000.0

            template_start = time.perf_counter()
            template = load_grayscale(Path(self.parameters.template_path))
            template_load_ms = (time.perf_counter() - template_start) * 1000.0
            common_timing = {
                "批次共用：載入 DLL": load_dll_ms,
                "批次共用：讀取 Template": template_load_ms,
            }

            image_paths = self._collect_images()
            if not image_paths:
                raise RuntimeError("輸入位置中找不到可處理的影像。")

            summary_rows: list[dict[str, Any]] = []
            last_payload: dict[str, Any] | None = None
            total_count = len(image_paths)

            for index, source_path in enumerate(image_paths, start=1):
                if self._cancel_requested:
                    break
                self.progress.emit(index - 1, total_count, source_path.name)
                try:
                    payload = self._process_one(
                        matcher,
                        template,
                        source_path,
                        common_timing,
                    )
                    last_payload = payload
                    timing = payload["timing"]
                    row = {
                        "status": "成功",
                        "source_image": str(source_path),
                        "image_width": payload["image_size"][0],
                        "image_height": payload["image_size"][1],
                        "match_count": len(payload["results"]),
                        "crop_count": sum(
                            bool(result.get("crop_path"))
                            for result in payload["results"]
                        ),
                        "crop_backend": payload["crop_backend"],
                        "image_load_ms": round(
                            float(timing["Python：讀取大圖並轉灰階"]), 3
                        ),
                        "pattern_match_ms": round(
                            float(timing["DLL：Pattern Match 總耗時"]), 3
                        ),
                        "crop_extract_ms": round(
                            float(timing["切圖：擷取總耗時"]), 3
                        ),
                        "crop_save_ms": round(
                            float(timing["切圖：編碼與寫入硬碟"]), 3
                        ),
                        "total_ms": round(
                            float(timing["單張完整流程總耗時"]), 3
                        ),
                        "message": "",
                    }
                    summary_rows.append(row)
                    payload["summary_row"] = row
                    self.item_finished.emit(payload)
                except Exception as exc:  # noqa: BLE001
                    if self.parameters.mode == "single":
                        raise
                    row = {
                        "status": "失敗",
                        "source_image": str(source_path),
                        "image_width": "",
                        "image_height": "",
                        "match_count": 0,
                        "crop_count": 0,
                        "crop_backend": "",
                        "image_load_ms": "",
                        "pattern_match_ms": "",
                        "crop_extract_ms": "",
                        "crop_save_ms": "",
                        "total_ms": "",
                        "message": str(exc),
                    }
                    summary_rows.append(row)
                    self.item_finished.emit(
                        {
                            "status": "失敗",
                            "source_path": str(source_path),
                            "summary_row": row,
                            "error": "".join(traceback.format_exception(exc)),
                        }
                    )
                self.progress.emit(index, total_count, source_path.name)

            if (
                self.parameters.output_dir
                and self.parameters.mode == "folder"
                and self.parameters.save_csv
            ):
                write_batch_summary(
                    Path(self.parameters.output_dir) / "batch_summary.csv",
                    summary_rows,
                )

            completed = len(summary_rows)
            success_count = sum(row["status"] == "成功" for row in summary_rows)
            failed_count = completed - success_count
            self.finished.emit(
                {
                    "rows": summary_rows,
                    "last_payload": last_payload,
                    "cancelled": self._cancel_requested,
                    "total_count": total_count,
                    "completed_count": completed,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "total_ms": (time.perf_counter() - whole_start) * 1000.0,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit("".join(traceback.format_exception(exc)))


class ImageViewer(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor(35, 35, 35))
        self._has_image = False

    def set_image_and_results(
        self,
        image_path: str,
        results: list[dict[str, Any]],
    ) -> None:
        scene = self.scene()
        scene.clear()

        preview_limit = (4096, 4096)
        with Image.open(image_path) as source:
            original_width, original_height = source.size
            try:
                source.draft(source.mode, preview_limit)
            except (AttributeError, ValueError):
                pass
            source.thumbnail(preview_limit, Image.Resampling.LANCZOS)
            preview = source.convert("RGB")

        preview_width, preview_height = preview.size
        preview_bytes = preview.tobytes("raw", "RGB")
        qimage = QImage(
            preview_bytes,
            preview_width,
            preview_height,
            preview_width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            raise RuntimeError(f"Qt 無法顯示影像：{image_path}")

        scale_x = preview_width / original_width
        scale_y = preview_height / original_height
        scene.addPixmap(pixmap)
        scene.setSceneRect(QRectF(pixmap.rect()))

        pen = QPen(QColor(255, 40, 40))
        pen.setWidth(3)
        pen.setCosmetic(True)
        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)

        for index, result in enumerate(results):
            rect = scene.addRect(
                float(result["x"]) * scale_x,
                float(result["y"]) * scale_y,
                float(result["width"]) * scale_x,
                float(result["height"]) * scale_y,
                pen,
            )
            rect.setZValue(10)
            if index < 100:
                text = QGraphicsSimpleTextItem(
                    f"#{result['rank']}  {result['score']:.4f}"
                )
                text.setFont(label_font)
                text.setBrush(QColor(255, 255, 0))
                text.setPos(
                    float(result["x"]) * scale_x,
                    float(result["y"]) * scale_y - 18.0,
                )
                text.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                    True,
                )
                text.setZValue(20)
                scene.addItem(text)

        self._has_image = True
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._has_image and not self.scene().sceneRect().isEmpty():
            self.fitInView(
                self.scene().sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._has_image:
            return super().wheelEvent(event)
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.scale(factor, factor)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CUDA Pattern Match 批次比對與切圖工具")
        self.resize(1600, 950)

        self.thread: QThread | None = None
        self.worker: ProcessingWorker | None = None
        self.last_payload: dict[str, Any] | None = None

        project_dir = Path(__file__).resolve().parent
        self.dll_path_edit = QLineEdit(str(project_dir / "cuda_pattern_match.dll"))
        self.template_path_edit = QLineEdit()
        self.image_path_edit = QLineEdit()
        self.folder_path_edit = QLineEdit()
        self.output_path_edit = QLineEdit(str(project_dir / "output"))

        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItem("單張圖片", "single")
        self.input_mode_combo.addItem("資料夾批次", "folder")
        self.input_mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(self._path_row(self.image_path_edit, self.pick_image))
        self.input_stack.addWidget(self._path_row(self.folder_path_edit, self.pick_folder))

        self.recursive_checkbox = QCheckBox("包含子資料夾")
        self.recursive_checkbox.setChecked(False)
        self.preview_each_checkbox = QCheckBox("批次時逐張更新大圖預覽（較慢）")
        self.preview_each_checkbox.setChecked(False)
        self.save_csv_checkbox = QCheckBox("輸出每張 matches.csv 與批次總表")
        self.save_csv_checkbox.setChecked(True)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.85)

        self.coarse_threshold_spin = QDoubleSpinBox()
        self.coarse_threshold_spin.setRange(-1.0, 1.0)
        self.coarse_threshold_spin.setDecimals(4)
        self.coarse_threshold_spin.setSingleStep(0.01)
        self.coarse_threshold_spin.setValue(0.70)

        self.pyramid_factor_combo = QComboBox()
        self.pyramid_factor_combo.addItem("Auto（建議）", 0)
        for factor in (2, 4, 8, 16, 32):
            self.pyramid_factor_combo.addItem(f"{factor}×", factor)

        self.refine_radius_spin = QSpinBox()
        self.refine_radius_spin.setRange(1, 8)
        self.refine_radius_spin.setValue(2)

        self.block_threads_combo = QComboBox()
        self.block_threads_combo.addItem("Auto（建議）", 0)
        for threads in (128, 256, 512, 1024):
            self.block_threads_combo.addItem(str(threads), threads)

        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setDecimals(3)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.25)

        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 10000)
        self.max_results_spin.setValue(100)

        self.max_candidates_spin = QSpinBox()
        self.max_candidates_spin.setRange(1000, 5_000_000)
        self.max_candidates_spin.setSingleStep(10000)
        self.max_candidates_spin.setValue(100000)

        self.crop_enable_checkbox = QCheckBox("比對完成後自動切出每個匹配 ROI")
        self.crop_enable_checkbox.setChecked(True)
        self.crop_enable_checkbox.toggled.connect(self._update_crop_controls)

        self.crop_backend_combo = QComboBox()
        self.crop_backend_combo.addItem("Auto（少量用 CPU，大批量用 GPU）", "auto")
        self.crop_backend_combo.addItem("強制 GPU", "gpu")
        self.crop_backend_combo.addItem("強制 CPU", "cpu")

        self.margin_x_spin = QSpinBox()
        self.margin_x_spin.setRange(0, 10000)
        self.margin_x_spin.setValue(0)
        self.margin_y_spin = QSpinBox()
        self.margin_y_spin.setRange(0, 10000)
        self.margin_y_spin.setValue(0)
        self.fill_value_spin = QSpinBox()
        self.fill_value_spin.setRange(0, 255)
        self.fill_value_spin.setValue(0)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("PNG", "PNG")
        self.output_format_combo.addItem("TIFF（LZW）", "TIFF")
        self.output_format_combo.addItem("JPEG", "JPEG")
        self.output_format_combo.currentIndexChanged.connect(self._update_crop_controls)

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(1, 100)
        self.jpeg_quality_spin.setValue(95)

        self.run_button = QPushButton("開始單張 GPU Pattern Match")
        self.run_button.setMinimumHeight(44)
        self.run_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("批次完成目前影像後停止")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.save_button = QPushButton("另存目前標註影像")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_annotated_image)
        self.fit_button = QPushButton("影像符合視窗")
        self.fit_button.clicked.connect(self._fit_image)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("尚未開始")
        self.progress_label.setWordWrap(True)

        self.template_preview = QLabel("尚未選擇 Template")
        self.template_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.template_preview.setMinimumHeight(140)
        self.template_preview.setStyleSheet(
            "QLabel { background: #222; color: #bbb; border: 1px solid #555; }"
        )

        self.viewer = ImageViewer()
        self.results_table = self._create_results_table()
        self.timing_table = self._create_timing_table()
        self.batch_table = self._create_batch_table()

        self._build_ui()
        self._build_menu()
        self.setStatusBar(QStatusBar())
        self._on_mode_changed()
        self._update_crop_controls()
        self.statusBar().showMessage("請選擇輸入影像與 Template。")

    def _build_ui(self) -> None:
        controls_content = QWidget()
        controls_layout = QVBoxLayout(controls_content)

        file_group = QGroupBox("檔案與批次")
        file_layout = QFormLayout(file_group)
        file_layout.addRow("CUDA DLL", self._path_row(self.dll_path_edit, self.pick_dll))
        file_layout.addRow("Template", self._path_row(self.template_path_edit, self.pick_template))
        file_layout.addRow("輸入模式", self.input_mode_combo)
        file_layout.addRow("輸入位置", self.input_stack)
        file_layout.addRow("輸出資料夾", self._path_row(self.output_path_edit, self.pick_output_folder))
        file_layout.addRow("", self.recursive_checkbox)
        file_layout.addRow("", self.preview_each_checkbox)
        file_layout.addRow("", self.save_csv_checkbox)
        controls_layout.addWidget(file_group)

        parameter_group = QGroupBox("Pattern Match 參數")
        parameter_layout = QFormLayout(parameter_group)
        parameter_layout.addRow("最終分數門檻", self.threshold_spin)
        parameter_layout.addRow("粗搜分數門檻", self.coarse_threshold_spin)
        parameter_layout.addRow("金字塔粗搜倍率", self.pyramid_factor_combo)
        parameter_layout.addRow("每層精搜半徑 px", self.refine_radius_spin)
        parameter_layout.addRow("CUDA threads/block", self.block_threads_combo)
        parameter_layout.addRow("NMS IoU", self.nms_spin)
        parameter_layout.addRow("最多結果數", self.max_results_spin)
        parameter_layout.addRow("GPU 候選容量", self.max_candidates_spin)
        controls_layout.addWidget(parameter_group)

        crop_group = QGroupBox("批量切小圖")
        crop_layout = QFormLayout(crop_group)
        crop_layout.addRow("", self.crop_enable_checkbox)
        crop_layout.addRow("擷取後端", self.crop_backend_combo)
        crop_layout.addRow("左右額外邊界 px", self.margin_x_spin)
        crop_layout.addRow("上下額外邊界 px", self.margin_y_spin)
        crop_layout.addRow("超出原圖補值", self.fill_value_spin)
        crop_layout.addRow("輸出格式", self.output_format_combo)
        crop_layout.addRow("JPEG 品質", self.jpeg_quality_spin)
        controls_layout.addWidget(crop_group)

        hint = QLabel(
            "GPU 可以平行切 ROI，但 PNG/JPEG/TIFF 編碼與硬碟寫入仍在 CPU。\n"
            "Auto 在少量小 ROI 時使用 CPU，避免為切圖再次上傳整張大圖；"
            "ROI 數量或輸出量夠大時才使用 GPU。\n"
            "切圖尺寸 = Template 尺寸 + 左右/上下邊界，靠近邊緣時固定尺寸補值。"
        )
        hint.setWordWrap(True)
        controls_layout.addWidget(hint)
        controls_layout.addWidget(self.template_preview)
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.cancel_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.progress_label)

        button_row = QHBoxLayout()
        button_row.addWidget(self.fit_button)
        button_row.addWidget(self.save_button)
        controls_layout.addLayout(button_row)
        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_content)
        scroll.setMinimumWidth(430)
        scroll.setMaximumWidth(500)

        tabs = QTabWidget()
        tabs.addTab(self.results_table, "目前影像結果")
        tabs.addTab(self.timing_table, "目前影像耗時")
        tabs.addTab(self.batch_table, "批次總表")

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.viewer)
        right_splitter.addWidget(tabs)
        right_splitter.setSizes([650, 300])

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(scroll)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([470, 1130])
        self.setCentralWidget(main_splitter)

    def _build_menu(self) -> None:
        fit_action = QAction("符合視窗", self)
        fit_action.setShortcut(QKeySequence("F"))
        fit_action.triggered.connect(self._fit_image)
        self.addAction(fit_action)

    @staticmethod
    def _path_row(line_edit: QLineEdit, callback) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("選擇")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return widget

    @staticmethod
    def _create_results_table() -> QTableWidget:
        table = QTableWidget(0, 7)
        table.setHorizontalHeaderLabels(
            ["排名", "X", "Y", "寬", "高", "分數", "切圖路徑"]
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        return table

    @staticmethod
    def _create_timing_table() -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["項目", "數值"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        return table

    @staticmethod
    def _create_batch_table() -> QTableWidget:
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(
            ["狀態", "影像", "尺寸", "匹配", "切圖", "後端", "總耗時 ms", "訊息"]
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        return table

    def pick_dll(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 cuda_pattern_match.dll",
            self.dll_path_edit.text(),
            "DLL (*.dll);;All files (*.*)",
        )
        if path:
            self.dll_path_edit.setText(path)

    def pick_image(self) -> None:
        path = self._pick_image_file("選擇待搜尋大圖")
        if path:
            self.image_path_edit.setText(path)

    def pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "選擇批次影像資料夾",
            self.folder_path_edit.text(),
        )
        if path:
            self.folder_path_edit.setText(path)

    def pick_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "選擇輸出資料夾",
            self.output_path_edit.text(),
        )
        if path:
            self.output_path_edit.setText(path)

    def pick_template(self) -> None:
        path = self._pick_image_file("選擇 Template")
        if not path:
            return
        self.template_path_edit.setText(path)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.template_preview.setPixmap(
                pixmap.scaled(
                    self.template_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _pick_image_file(self, title: str) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        return path

    @Slot()
    def _on_mode_changed(self) -> None:
        mode = self.input_mode_combo.currentData()
        self.input_stack.setCurrentIndex(0 if mode == "single" else 1)
        is_folder = mode == "folder"
        self.recursive_checkbox.setEnabled(is_folder)
        self.preview_each_checkbox.setEnabled(is_folder)
        self.run_button.setText(
            "開始資料夾批次 Pattern Match"
            if is_folder
            else "開始單張 GPU Pattern Match"
        )

    @Slot()
    def _update_crop_controls(self) -> None:
        enabled = self.crop_enable_checkbox.isChecked()
        for widget in (
            self.crop_backend_combo,
            self.margin_x_spin,
            self.margin_y_spin,
            self.fill_value_spin,
            self.output_format_combo,
        ):
            widget.setEnabled(enabled)
        self.jpeg_quality_spin.setEnabled(
            enabled and self.output_format_combo.currentData() == "JPEG"
        )

    def _validate_parameters(self) -> JobParameters:
        dll_path = Path(self.dll_path_edit.text().strip())
        template_path = Path(self.template_path_edit.text().strip())
        mode = str(self.input_mode_combo.currentData())
        input_path = Path(
            self.image_path_edit.text().strip()
            if mode == "single"
            else self.folder_path_edit.text().strip()
        )
        output_text = self.output_path_edit.text().strip()

        if not dll_path.is_file():
            raise ValueError(f"DLL 不存在：{dll_path}")
        if not template_path.is_file():
            raise ValueError(f"Template 不存在：{template_path}")
        if mode == "single" and not input_path.is_file():
            raise ValueError(f"大圖不存在：{input_path}")
        if mode == "folder" and not input_path.is_dir():
            raise ValueError(f"批次資料夾不存在：{input_path}")
        if self.crop_enable_checkbox.isChecked() and not output_text:
            raise ValueError("啟用切圖時必須設定輸出資料夾。")
        if mode == "folder" and not output_text:
            raise ValueError("資料夾批次模式必須設定輸出資料夾。")

        threshold = float(self.threshold_spin.value())
        coarse_threshold = float(self.coarse_threshold_spin.value())
        if coarse_threshold > threshold:
            raise ValueError("粗搜分數門檻不可高於最終分數門檻。")

        algorithm = AlgorithmSettings(
            threshold=threshold,
            coarse_threshold=coarse_threshold,
            pyramid_factor=int(self.pyramid_factor_combo.currentData()),
            refine_radius=int(self.refine_radius_spin.value()),
            nms_iou=float(self.nms_spin.value()),
            max_results=int(self.max_results_spin.value()),
            max_candidates=int(self.max_candidates_spin.value()),
            block_threads=int(self.block_threads_combo.currentData()),
        )
        crop = CropSettings(
            enabled=self.crop_enable_checkbox.isChecked(),
            backend=str(self.crop_backend_combo.currentData()),
            margin_x=int(self.margin_x_spin.value()),
            margin_y=int(self.margin_y_spin.value()),
            fill_value=int(self.fill_value_spin.value()),
            output_format=str(self.output_format_combo.currentData()),
            jpeg_quality=int(self.jpeg_quality_spin.value()),
        )
        return JobParameters(
            dll_path=str(dll_path),
            template_path=str(template_path),
            mode=mode,
            input_path=str(input_path),
            output_dir=output_text,
            recursive=self.recursive_checkbox.isChecked(),
            save_csv=self.save_csv_checkbox.isChecked(),
            preview_each=self.preview_each_checkbox.isChecked(),
            algorithm=algorithm,
            crop=crop,
        )

    @Slot()
    def start_processing(self) -> None:
        if self.thread is not None:
            return
        try:
            parameters = self._validate_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "參數錯誤", str(exc))
            return

        self.last_payload = None
        self.results_table.setRowCount(0)
        self.timing_table.setRowCount(0)
        self.batch_table.setRowCount(0)
        self.save_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在整理影像清單…")

        self.thread = QThread(self)
        self.worker = ProcessingWorker(parameters)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.item_finished.connect(self.on_item_finished)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.failed.connect(self.on_processing_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._clear_worker)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.statusBar().showMessage("開始處理…")

    @Slot()
    def cancel_processing(self) -> None:
        if self.worker is not None:
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("已要求停止；目前這張影像完成後停止。")

    @Slot(int, int, str)
    def on_progress(self, completed: int, total: int, name: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(completed)
        self.progress_label.setText(f"{completed}/{total}　{name}")

    @Slot(object)
    def on_item_finished(self, payload: dict[str, Any]) -> None:
        self._append_batch_row(payload["summary_row"])
        if payload.get("status") != "成功":
            self.statusBar().showMessage(f"失敗：{payload.get('source_path', '')}")
            return

        parameters = payload["parameters"]
        should_preview = (
            parameters["mode"] == "single" or parameters["preview_each"]
        )
        self.last_payload = payload
        self._fill_results_table(payload["results"])
        self._fill_timing_table(payload["timing"])
        if should_preview:
            self._render_payload(payload)
        self.save_button.setEnabled(bool(payload["results"]))
        self.statusBar().showMessage(
            f"完成 {Path(payload['source_path']).name}："
            f"{len(payload['results'])} 個匹配，切圖後端 {payload['crop_backend']}。"
        )

    @Slot(object)
    def on_processing_finished(self, summary: dict[str, Any]) -> None:
        last_payload = summary.get("last_payload")
        if last_payload is not None:
            self.last_payload = last_payload
            self._fill_results_table(last_payload["results"])
            self._fill_timing_table(last_payload["timing"])
            if not last_payload["parameters"]["preview_each"]:
                self._render_payload(last_payload)
            self.save_button.setEnabled(bool(last_payload["results"]))

        self.progress_bar.setRange(0, max(1, int(summary["total_count"])))
        self.progress_bar.setValue(int(summary["completed_count"]))
        total_seconds = float(summary["total_ms"]) / 1000.0
        state = "已取消" if summary["cancelled"] else "全部完成"
        message = (
            f"{state}：成功 {summary['success_count']}，失敗 {summary['failed_count']}，"
            f"耗時 {total_seconds:.2f} 秒。"
        )
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot(str)
    def on_processing_failed(self, details: str) -> None:
        QMessageBox.critical(self, "執行失敗", details)
        self.statusBar().showMessage("執行失敗。")
        self.progress_label.setText("執行失敗")

    @Slot()
    def _clear_worker(self) -> None:
        self.worker = None
        self.thread = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _render_payload(self, payload: dict[str, Any]) -> None:
        render_start = time.perf_counter()
        try:
            self.viewer.set_image_and_results(
                payload["source_path"],
                payload["results"],
            )
            render_ms = (time.perf_counter() - render_start) * 1000.0
            payload["timing"]["GUI：載入預覽與繪製標註"] = render_ms
            self._fill_timing_table(payload["timing"])
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"比對已完成，但預覽失敗：{exc}")

    def _fill_results_table(self, results: list[dict[str, Any]]) -> None:
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            values = [
                result["rank"],
                result["x"],
                result["y"],
                result["width"],
                result["height"],
                f"{result['score']:.6f}",
                result.get("crop_path", ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, column, item)

    def _fill_timing_table(self, timing: dict[str, Any]) -> None:
        self.timing_table.setRowCount(len(timing))
        for row, (name, value) in enumerate(timing.items()):
            if isinstance(value, bool):
                display_value = "是" if value else "否"
            elif isinstance(value, float):
                display_value = f"{value:,.3f} ms"
            elif isinstance(value, int):
                display_value = f"{value:,}"
            else:
                display_value = str(value)
            self.timing_table.setItem(row, 0, QTableWidgetItem(name))
            value_item = QTableWidgetItem(display_value)
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.timing_table.setItem(row, 1, value_item)

    def _append_batch_row(self, row_data: dict[str, Any]) -> None:
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        width = row_data.get("image_width", "")
        height = row_data.get("image_height", "")
        size = f"{width} × {height}" if width != "" else ""
        values = [
            row_data.get("status", ""),
            row_data.get("source_image", ""),
            size,
            row_data.get("match_count", ""),
            row_data.get("crop_count", ""),
            row_data.get("crop_backend", ""),
            row_data.get("total_ms", ""),
            row_data.get("message", ""),
        ]
        for column, value in enumerate(values):
            self.batch_table.setItem(row, column, QTableWidgetItem(str(value)))

    @Slot()
    def _fit_image(self) -> None:
        self.viewer.fit_to_window()

    @Slot()
    def save_annotated_image(self) -> None:
        if not self.last_payload:
            return
        source_path = Path(self.last_payload["source_path"])
        default_path = source_path.with_name(f"{source_path.stem}_matched.png")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "另存標註影像",
            str(default_path),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)",
        )
        if not output_path:
            return

        try:
            with Image.open(source_path) as source:
                annotated = source.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            for result in self.last_payload["results"]:
                x1 = int(result["x"])
                y1 = int(result["y"])
                x2 = x1 + int(result["width"])
                y2 = y1 + int(result["height"])
                line_width = max(2, min(result["width"], result["height"]) // 30)
                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline=(255, 0, 0),
                    width=line_width,
                )
                draw.text(
                    (x1, max(0, y1 - 14)),
                    f"#{result['rank']} {result['score']:.4f}",
                    fill=(255, 255, 0),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
            annotated.save(output_path)
            self.statusBar().showMessage(f"標註影像已儲存：{output_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "儲存失敗", str(exc))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(
                self,
                "仍在執行",
                "請先停止批次，並等待目前影像處理完成。",
            )
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CUDA Pattern Match Batch GUI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
