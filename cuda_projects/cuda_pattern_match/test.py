from __future__ import annotations

import ctypes
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# 此工具只開啟使用者主動選取的本機影像。工業大圖常超過 Pillow
# 預設的 decompression-bomb 像素上限，因此停用該上限。
Image.MAX_IMAGE_PIXELS = None
from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDoubleSpinBox,
    QComboBox,
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
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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


@dataclass(frozen=True)
class MatchParameters:
    dll_path: str
    image_path: str
    template_path: str
    threshold: float
    coarse_threshold: float
    pyramid_factor: int
    refine_radius: int
    nms_iou: float
    max_results: int
    max_candidates: int
    block_threads: int


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
                "請重新下載或重新編譯正式版 DLL。"
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

        self.dll.cuda_pattern_match_error_string.argtypes = [ctypes.c_int]
        self.dll.cuda_pattern_match_error_string.restype = ctypes.c_char_p

    def error_string(self, code: int) -> str:
        raw = self.dll.cuda_pattern_match_error_string(code)
        return raw.decode("utf-8", errors="replace") if raw else "Unknown error"

    def match(
        self,
        image: np.ndarray,
        template: np.ndarray,
        parameters: MatchParameters,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if image.ndim != 2 or template.ndim != 2:
            raise ValueError("DLL 僅接受 8-bit 單通道影像。")

        image = np.ascontiguousarray(image, dtype=np.uint8)
        template = np.ascontiguousarray(template, dtype=np.uint8)

        result_buffer = (MatchResult * parameters.max_results)()
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
            ctypes.c_float(parameters.threshold),
            ctypes.c_float(parameters.coarse_threshold),
            parameters.pyramid_factor,
            parameters.refine_radius,
            ctypes.c_float(parameters.nms_iou),
            parameters.max_results,
            parameters.max_candidates,
            parameters.block_threads,
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
            }
            for index in range(result_count.value)
        ]

        timing_data = {
            "DLL：Template 金字塔預處理": timing.host_prepare_ms,
            "GPU：Host → Device": timing.host_to_device_ms,
            "GPU：建立影像金字塔": timing.pyramid_build_ms,
            "GPU：粗層 ZNCC score map": timing.coarse_score_ms,
            "GPU：粗層局部極大值": timing.coarse_peak_ms,
            "GPU：逐層精搜": timing.refine_ms,
            "GPU：Device → Host": timing.device_to_host_ms,
            "CPU：排序與 NMS": timing.sort_nms_ms,
            "DLL：總耗時": timing.total_ms,
            "粗搜位置數": int(timing.coarse_evaluated_positions),
            "精搜位置數": int(timing.refine_evaluated_positions),
            "粗搜候選總數": int(timing.raw_candidate_count),
            "實際精搜候選數": int(timing.stored_candidate_count),
            "最終結果數": int(timing.result_count),
            "候選容量溢位": bool(timing.candidate_overflow),
            "CUDA kernel launch 次數": int(timing.kernel_launch_count),
            "要求金字塔倍率": (
                "Auto" if timing.requested_pyramid_factor == 0
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
            "DLL 估計峰值 VRAM": f"{timing.estimated_vram_used_mib:,.1f} MiB",
        }
        return results, timing_data


class MatchWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parameters: MatchParameters) -> None:
        super().__init__()
        self.parameters = parameters

    @staticmethod
    def _load_grayscale(path: str) -> np.ndarray:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            return np.ascontiguousarray(np.asarray(grayscale), dtype=np.uint8)

    @Slot()
    def run(self) -> None:
        end_to_end_start = time.perf_counter()

        try:
            load_dll_start = time.perf_counter()
            matcher = CudaPatternMatcher(Path(self.parameters.dll_path))
            load_dll_ms = (time.perf_counter() - load_dll_start) * 1000.0

            image_start = time.perf_counter()
            image = self._load_grayscale(self.parameters.image_path)
            image_load_ms = (time.perf_counter() - image_start) * 1000.0

            template_start = time.perf_counter()
            template = self._load_grayscale(self.parameters.template_path)
            template_load_ms = (time.perf_counter() - template_start) * 1000.0

            call_start = time.perf_counter()
            results, timing = matcher.match(image, template, self.parameters)
            python_call_ms = (time.perf_counter() - call_start) * 1000.0

            timing = {
                "Python：載入 DLL": load_dll_ms,
                "Python：讀取大圖並轉灰階": image_load_ms,
                "Python：讀取 Template 並轉灰階": template_load_ms,
                **timing,
                "Python：ctypes 呼叫區段": python_call_ms,
                "Python：執行緒端到端":
                    (time.perf_counter() - end_to_end_start) * 1000.0,
            }

            payload = {
                "parameters": asdict(self.parameters),
                "image_size": (int(image.shape[1]), int(image.shape[0])),
                "template_size": (
                    int(template.shape[1]),
                    int(template.shape[0]),
                ),
                "results": results,
                "timing": timing,
            }
            self.finished.emit(payload)
        except Exception as exc:  # noqa: BLE001
            details = "".join(traceback.format_exception(exc))
            self.failed.emit(details)


class ImageViewer(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
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

            # 先縮小原始模式，再轉 RGB。避免 2 億像素大圖先產生一份
            # 約 600 MB 的完整 RGB 複本。draft 對 JPEG 等格式可提前降採樣。
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
        self.setWindowTitle("CUDA Pattern Match DLL 測試工具（PySide6）")
        self.resize(1500, 900)

        self.thread: QThread | None = None
        self.worker: MatchWorker | None = None
        self.last_payload: dict[str, Any] | None = None

        self.dll_path_edit = QLineEdit(
            str(Path(__file__).resolve().parent / "cuda_pattern_match.dll")
        )
        self.image_path_edit = QLineEdit()
        self.template_path_edit = QLineEdit()

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
        self.pyramid_factor_combo.setCurrentIndex(0)

        self.refine_radius_spin = QSpinBox()
        self.refine_radius_spin.setRange(1, 8)
        self.refine_radius_spin.setValue(2)

        self.block_threads_combo = QComboBox()
        self.block_threads_combo.addItem("Auto（建議）", 0)
        for threads in (128, 256, 512, 1024):
            self.block_threads_combo.addItem(str(threads), threads)
        self.block_threads_combo.setCurrentIndex(0)

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

        self.run_button = QPushButton("開始 GPU Pattern Match")
        self.run_button.setMinimumHeight(42)
        self.run_button.clicked.connect(self.start_match)

        self.save_button = QPushButton("另存標註影像")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_annotated_image)

        self.fit_button = QPushButton("影像符合視窗")
        self.fit_button.clicked.connect(self._fit_image)

        self.template_preview = QLabel("尚未選擇 Template")
        self.template_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.template_preview.setMinimumHeight(160)
        self.template_preview.setStyleSheet(
            "QLabel { background: #222; color: #bbb; border: 1px solid #555; }"
        )

        self.viewer = ImageViewer()
        self.results_table = self._create_results_table()
        self.timing_table = self._create_timing_table()

        self._build_ui()
        self._build_menu()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("請選擇大圖與 Template。")

    def _build_ui(self) -> None:
        controls = QWidget()
        controls.setMinimumWidth(360)
        controls.setMaximumWidth(430)
        controls_layout = QVBoxLayout(controls)

        file_group = QGroupBox("檔案")
        file_layout = QFormLayout(file_group)
        file_layout.addRow("CUDA DLL", self._path_row(self.dll_path_edit, self.pick_dll))
        file_layout.addRow("大圖", self._path_row(self.image_path_edit, self.pick_image))
        file_layout.addRow(
            "Template",
            self._path_row(self.template_path_edit, self.pick_template),
        )
        controls_layout.addWidget(file_group)

        parameter_group = QGroupBox("比對參數")
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

        hint = QLabel(
            "正式版先在縮小影像做完整粗搜，再逐層回到原圖精搜。\n"
            "204,800,000 px 大圖建議使用 Auto；DLL 會選 Template 仍可辨識的最大倍率。\n"
            "threads/block 與 VRAM 容量無直接關係；Auto 通常比強制 1024 更快。"
        )
        hint.setWordWrap(True)
        controls_layout.addWidget(hint)
        controls_layout.addWidget(self.template_preview)
        controls_layout.addWidget(self.run_button)

        button_row = QHBoxLayout()
        button_row.addWidget(self.fit_button)
        button_row.addWidget(self.save_button)
        controls_layout.addLayout(button_row)
        controls_layout.addStretch(1)

        tabs = QTabWidget()
        tabs.addTab(self.results_table, "比對結果")
        tabs.addTab(self.timing_table, "耗時細節")

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.viewer)
        right_splitter.addWidget(tabs)
        right_splitter.setSizes([650, 250])

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(controls)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([390, 1110])

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
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["排名", "X", "Y", "寬", "高", "分數"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    @staticmethod
    def _create_timing_table() -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["項目", "數值"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
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
            self.statusBar().showMessage(f"已選擇大圖：{path}")

    def pick_template(self) -> None:
        path = self._pick_image_file("選擇 Template")
        if path:
            self.template_path_edit.setText(path)
            pixmap = QPixmap(path)
            self.template_preview.setPixmap(
                pixmap.scaled(
                    self.template_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.statusBar().showMessage(f"已選擇 Template：{path}")

    def _pick_image_file(self, title: str) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        return path

    def _validate_parameters(self) -> MatchParameters:
        dll_path = Path(self.dll_path_edit.text().strip())
        image_path = Path(self.image_path_edit.text().strip())
        template_path = Path(self.template_path_edit.text().strip())

        if not dll_path.is_file():
            raise ValueError(f"DLL 不存在：{dll_path}")
        if not image_path.is_file():
            raise ValueError(f"大圖不存在：{image_path}")
        if not template_path.is_file():
            raise ValueError(f"Template 不存在：{template_path}")

        threshold = float(self.threshold_spin.value())
        coarse_threshold = float(self.coarse_threshold_spin.value())
        if coarse_threshold > threshold:
            raise ValueError("粗搜分數門檻不可高於最終分數門檻。")

        return MatchParameters(
            dll_path=str(dll_path),
            image_path=str(image_path),
            template_path=str(template_path),
            threshold=threshold,
            coarse_threshold=coarse_threshold,
            pyramid_factor=int(self.pyramid_factor_combo.currentData()),
            refine_radius=int(self.refine_radius_spin.value()),
            nms_iou=float(self.nms_spin.value()),
            max_results=int(self.max_results_spin.value()),
            max_candidates=int(self.max_candidates_spin.value()),
            block_threads=int(self.block_threads_combo.currentData()),
        )

    @Slot()
    def start_match(self) -> None:
        if self.thread is not None:
            return

        try:
            parameters = self._validate_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "參數錯誤", str(exc))
            return

        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.results_table.setRowCount(0)
        self.timing_table.setRowCount(0)
        self.statusBar().showMessage("GPU 比對中…")

        self.thread = QThread(self)
        self.worker = MatchWorker(parameters)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_match_finished)
        self.worker.failed.connect(self.on_match_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._clear_worker)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(object)
    def on_match_finished(self, payload: dict[str, Any]) -> None:
        render_start = time.perf_counter()
        self.last_payload = payload

        self.viewer.set_image_and_results(
            payload["parameters"]["image_path"],
            payload["results"],
        )
        self._fill_results_table(payload["results"])

        render_ms = (time.perf_counter() - render_start) * 1000.0
        payload["timing"]["GUI：載入大圖與繪製標註"] = render_ms
        payload["timing"]["GUI：完整流程總耗時"] = (
            payload["timing"]["Python：執行緒端到端"] + render_ms
        )
        self._fill_timing_table(payload["timing"])

        result_count = len(payload["results"])
        overflow = bool(payload["timing"]["候選容量溢位"])
        message = f"完成：找到 {result_count} 個結果。"
        if overflow:
            message += " GPU 候選容量已滿，結果可能不完整。"
        self.statusBar().showMessage(message)
        self.save_button.setEnabled(result_count > 0)

    @Slot(str)
    def on_match_failed(self, details: str) -> None:
        self.last_payload = None
        QMessageBox.critical(self, "執行失敗", details)
        self.statusBar().showMessage("執行失敗。")

    @Slot()
    def _clear_worker(self) -> None:
        self.worker = None
        self.thread = None
        self.run_button.setEnabled(True)

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
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.timing_table.setItem(row, 1, value_item)

    @Slot()
    def _fit_image(self) -> None:
        self.viewer.fit_to_window()

    @Slot()
    def save_annotated_image(self) -> None:
        if not self.last_payload:
            return

        source_path = Path(self.last_payload["parameters"]["image_path"])
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
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=line_width)
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
            QMessageBox.information(self, "仍在執行", "請等待目前的 GPU 比對結束。")
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CUDA Pattern Match GUI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
