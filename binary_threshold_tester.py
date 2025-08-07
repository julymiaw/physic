import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple


# Set video path directly here
# VIDEO_PATH = "test.mp4"  # Change to your video file path
VIDEO_PATH = "真实场景.mp4"


class ThresholdMethod(ABC):
    """阈值计算方法的抽象基类"""

    def __init__(
        self,
        name: str,
        color: str = "black",
        category: str = "Other",
        is_adaptive: bool = False,
        show_in_buttons: bool = True,
    ):
        self.name = name
        self.color = color
        self.category = category
        self.is_adaptive = is_adaptive  # 是否为自适应方法（影响UI显示）
        self.show_in_buttons = show_in_buttons  # 是否在快速按钮中显示

    @abstractmethod
    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        """计算阈值的抽象方法"""
        pass

    def apply_threshold(self, red_channel: np.ndarray, threshold: int) -> np.ndarray:
        """应用阈值进行二值化（可以被子类重写）"""
        _, binary_image = cv.threshold(
            red_channel, threshold, 255, cv.THRESH_BINARY_INV
        )
        return binary_image


class AdaptiveThresholdMethod(ThresholdMethod):
    """自适应阈值方法的基类"""

    def __init__(self, name: str, color: str = "black", category: str = "Adaptive"):
        super().__init__(name, color, category, is_adaptive=True, show_in_buttons=False)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        """自适应方法返回None，因为它们不使用固定阈值"""
        return None

    @abstractmethod
    def apply_threshold(
        self, red_channel: np.ndarray, threshold: int = 0
    ) -> np.ndarray:
        """自适应方法的二值化实现"""
        pass


# =================== 具体的阈值方法实现 ===================


class PeakBasedThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Peak-based", "red", "Basic", show_in_buttons=True)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
            hist[0] = 0

            peaks, _ = find_peaks(hist)
            sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
            if len(sorted_peaks) >= 2:
                second_peak_index = min(sorted_peaks[:2])
                min_val_after_peak = (
                    np.argmin(hist[second_peak_index:]) + second_peak_index
                )
                return min_val_after_peak
        except:
            pass
        return None


class MeanThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Mean", "purple", "Basic", show_in_buttons=True)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            valid_pixels = red_channel[red_channel > 0]
            if len(valid_pixels) > 0:
                return int(np.mean(valid_pixels))
        except:
            pass
        return None


class MedianThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Median", "orange", "Basic", show_in_buttons=True)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            valid_pixels = red_channel[red_channel > 0]
            if len(valid_pixels) > 0:
                return int(np.median(valid_pixels))
        except:
            pass
        return None


class OtsuThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Otsu", "green", "Basic", show_in_buttons=True)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            otsu_thresh, _ = cv.threshold(
                red_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )
            return int(otsu_thresh)
        except:
            pass
        return None


class ModeThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Mode", "brown", "Basic", show_in_buttons=True)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
            hist[0] = 0
            if len(hist) > 0:
                return np.argmax(hist)
        except:
            pass
        return None


class TriangleThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Triangle", "pink", "Advanced", show_in_buttons=False)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
            hist[0] = 0

            max_idx = np.argmax(hist)
            right_idx = len(hist) - 1
            while right_idx > max_idx and hist[right_idx] == 0:
                right_idx -= 1

            if right_idx > max_idx:
                max_distance = 0
                triangle_thresh = max_idx
                for i in range(max_idx, right_idx + 1):
                    distance = abs(
                        (hist[right_idx] - hist[max_idx]) * i
                        - (right_idx - max_idx) * hist[i]
                        + right_idx * hist[max_idx]
                        - max_idx * hist[right_idx]
                    ) / np.sqrt(
                        (hist[right_idx] - hist[max_idx]) ** 2
                        + (right_idx - max_idx) ** 2
                    )
                    if distance > max_distance:
                        max_distance = distance
                        triangle_thresh = i
                return triangle_thresh
        except:
            pass
        return None


class GMMThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("GMM", "cyan", "Advanced", show_in_buttons=False)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            valid_pixels = red_channel[red_channel > 0]
            if len(valid_pixels) > 100:
                gmm = GaussianMixture(n_components=2, random_state=42)
                data = valid_pixels.reshape(-1, 1)
                gmm.fit(data)
                means = gmm.means_.flatten()
                return int(np.mean(means))
        except:
            pass
        return None


class MinErrorThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Min Error", "gray", "Advanced", show_in_buttons=False)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
            hist[0] = 0

            min_error = float("inf")
            min_error_thresh = 128
            for t in range(1, 255):
                if np.sum(hist[:t]) > 0 and np.sum(hist[t:]) > 0:
                    w0 = np.sum(hist[:t])
                    w1 = np.sum(hist[t:])
                    if w0 > 0 and w1 > 0:
                        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
                        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1

                        var0 = (
                            np.sum(((np.arange(t) - mu0) ** 2) * hist[:t]) / w0
                            if w0 > 0
                            else 1
                        )
                        var1 = (
                            np.sum(((np.arange(t, 256) - mu1) ** 2) * hist[t:]) / w1
                            if w1 > 0
                            else 1
                        )

                        if var0 > 0 and var1 > 0:
                            error = (
                                w0 * np.log(var0)
                                + w1 * np.log(var1)
                                - w0 * np.log(w0)
                                - w1 * np.log(w1)
                            )
                            if error < min_error:
                                min_error = error
                                min_error_thresh = t
            return min_error_thresh
        except:
            pass
        return None


class WeightedMeanThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__("Weighted Mean", "magenta", "Advanced", show_in_buttons=False)

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
            hist[0] = 0

            weighted_sum = np.sum(np.arange(256) * hist)
            total_weight = np.sum(hist)
            if total_weight > 0:
                return int(weighted_sum / total_weight)
        except:
            pass
        return None


class PercentileThreshold(ThresholdMethod):
    def __init__(self, percentile: int):
        super().__init__(
            f"{percentile}th Percentile",
            self._get_color(percentile),
            "Percentile",
            show_in_buttons=False,
        )
        self.percentile = percentile

    def _get_color(self, percentile):
        colors = {25: "lightblue", 75: "lightgreen", 90: "lightyellow"}
        return colors.get(percentile, "lightgray")

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            valid_pixels = red_channel[red_channel > 0]
            if len(valid_pixels) > 0:
                return int(np.percentile(valid_pixels, self.percentile))
        except:
            pass
        return None


class AdaptiveMeanGlobalThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__(
            "Adaptive Mean (Global)", "darkred", "Adaptive", show_in_buttons=False
        )

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            C = 2
            mean_val = np.mean(red_channel[red_channel > 0])
            return max(0, min(255, int(mean_val - C)))
        except:
            pass
        return None


class AdaptiveGaussianGlobalThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__(
            "Adaptive Gaussian (Global)", "darkgreen", "Adaptive", show_in_buttons=False
        )

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            if mask is not None and mask.shape == red_channel.shape:
                red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

            gaussian_mean = np.mean(red_channel[red_channel > 0])
            gaussian_std = np.std(red_channel[red_channel > 0])
            C = 2
            return max(0, min(255, int(gaussian_mean - gaussian_std * 0.5 - C)))
        except:
            pass
        return None


class LocalAdaptiveCenterThreshold(ThresholdMethod):
    def __init__(self):
        super().__init__(
            "Local Adaptive (Center)", "darkorange", "Adaptive", show_in_buttons=False
        )

    def calculate_threshold(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[int]:
        try:
            kernel_size = 15
            C = 2
            h, w = red_channel.shape
            center_y, center_x = h // 2, w // 2
            half_kernel = kernel_size // 2

            center_region = red_channel[
                max(0, center_y - half_kernel) : min(h, center_y + half_kernel + 1),
                max(0, center_x - half_kernel) : min(w, center_x + half_kernel + 1),
            ]
            valid_center_pixels = center_region[center_region > 0]
            if len(valid_center_pixels) > 0:
                local_mean = np.mean(valid_center_pixels)
                return max(0, min(255, int(local_mean - C)))
        except:
            pass
        return None


# 自适应阈值方法实现
class AdaptiveMeanThreshold(AdaptiveThresholdMethod):
    def __init__(self):
        super().__init__("Adaptive Mean", "darkred")

    def apply_threshold(
        self, red_channel: np.ndarray, threshold: int = 0
    ) -> np.ndarray:
        return cv.adaptiveThreshold(
            red_channel, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 2
        )


class AdaptiveGaussianThreshold(AdaptiveThresholdMethod):
    def __init__(self):
        super().__init__("Adaptive Gaussian", "darkgreen")

    def apply_threshold(
        self, red_channel: np.ndarray, threshold: int = 0
    ) -> np.ndarray:
        return cv.adaptiveThreshold(
            red_channel, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 2
        )


class CLAHEThreshold(AdaptiveThresholdMethod):
    def __init__(self):
        super().__init__("CLAHE + Otsu", "darkblue")

    def apply_threshold(
        self, red_channel: np.ndarray, threshold: int = 0
    ) -> np.ndarray:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(red_channel)
        _, binary_image = cv.threshold(
            enhanced, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )
        return binary_image


# =================== 阈值管理器 ===================


class ThresholdMethodManager:
    """阈值方法管理器"""

    def __init__(self):
        self.methods: Dict[str, ThresholdMethod] = {}
        self.adaptive_methods: Dict[str, str] = {}  # name -> method_key mapping
        self._register_default_methods()

    def _register_default_methods(self):
        """注册默认的阈值方法"""
        # 基础方法
        self.register_method(PeakBasedThreshold())
        self.register_method(MeanThreshold())
        self.register_method(MedianThreshold())
        self.register_method(OtsuThreshold())
        self.register_method(ModeThreshold())

        # 高级方法
        self.register_method(TriangleThreshold())
        self.register_method(GMMThreshold())
        self.register_method(MinErrorThreshold())
        self.register_method(WeightedMeanThreshold())

        # 百分位数方法
        self.register_method(PercentileThreshold(25))
        self.register_method(PercentileThreshold(75))
        self.register_method(PercentileThreshold(90))

        # 自适应全局近似方法
        self.register_method(AdaptiveMeanGlobalThreshold())
        self.register_method(AdaptiveGaussianGlobalThreshold())
        self.register_method(LocalAdaptiveCenterThreshold())

        # 真正的自适应方法
        self.register_adaptive_method("adaptive_mean", AdaptiveMeanThreshold())
        self.register_adaptive_method("adaptive_gaussian", AdaptiveGaussianThreshold())
        self.register_adaptive_method("clahe", CLAHEThreshold())

    def register_method(self, method: ThresholdMethod):
        """注册阈值方法"""
        self.methods[method.name] = method

    def register_adaptive_method(self, key: str, method: AdaptiveThresholdMethod):
        """注册自适应阈值方法"""
        self.methods[method.name] = method
        self.adaptive_methods[method.name] = key

    def calculate_all_thresholds(
        self, red_channel: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[int]]:
        """计算所有阈值方法的结果"""
        results = {}
        for name, method in self.methods.items():
            if not method.is_adaptive:  # 只计算非自适应方法的阈值
                results[name] = method.calculate_threshold(red_channel, mask)
        return results

    def get_method(self, name: str) -> Optional[ThresholdMethod]:
        """获取指定名称的阈值方法"""
        return self.methods.get(name)

    def get_methods_by_category(self) -> Dict[str, list]:
        """按类别获取方法"""
        categories = {}
        for method in self.methods.values():
            if method.category not in categories:
                categories[method.category] = []
            categories[method.category].append(method)
        return categories

    def get_button_methods(self) -> list:
        """获取应该显示在快速按钮中的方法"""
        return [method for method in self.methods.values() if method.show_in_buttons]

    def get_adaptive_method_keys(self) -> list:
        """获取自适应方法的键列表"""
        return list(self.adaptive_methods.values())

    def get_adaptive_method_by_key(self, key: str) -> Optional[AdaptiveThresholdMethod]:
        """通过键获取自适应方法"""
        for name, method_key in self.adaptive_methods.items():
            if method_key == key:
                return self.methods.get(name)
        return None


# =================== 辅助函数 ===================


def segment_image(src, kernel_size):
    """
    对图像进行分割，并返回分割后的边界框和掩码。
    （从原始轮廓检测代码复制）
    """
    _, src_bin = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
    coords = cv.findNonZero(src_bin)
    contours, _ = cv.findContours(src_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv.boundingRect(coords), None
    contour = max(contours, key=cv.contourArea)
    mask = np.zeros(src.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 1, thickness=cv.FILLED)
    return cv.boundingRect(coords), mask


def crop_to_bbox(frame, bbox):
    """
    根据给定的边界框裁剪图像。
    （从原始轮廓检测代码复制）
    """
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w].copy()


def process_frame_for_threshold_test(
    frame, manual_threshold=128, threshold_method="manual", threshold_manager=None
):
    """
    按照轮廓检测流程处理帧，支持多种阈值方法
    """
    try:
        # 1. 提取红色通道
        red_channel = frame[:, :, 2]

        # 2. 进行有效区域分割（使用OTSU，与原始process_frame一致）
        bbox, mask = segment_image(red_channel, (5, 5))
        current_bbox_area = bbox[2] * bbox[3]

        if current_bbox_area == 0:
            return False, None, None, None, None, None

        # 3. 裁剪到有效区域
        red_channel_cropped = crop_to_bbox(red_channel, bbox)
        mask_cropped = crop_to_bbox(mask, bbox) if mask is not None else None
        cropped_frame = crop_to_bbox(frame, bbox)

        # 4. 根据选择的方法进行二值化
        if threshold_method == "manual":
            _, binary_image = cv.threshold(
                red_channel_cropped, manual_threshold, 255, cv.THRESH_BINARY_INV
            )
        else:
            # 使用阈值管理器中的自适应方法
            if threshold_manager:
                adaptive_method = threshold_manager.get_adaptive_method_by_key(
                    threshold_method
                )
                if adaptive_method:
                    binary_image = adaptive_method.apply_threshold(red_channel_cropped)
                else:
                    # 如果找不到方法，使用手动阈值作为备用
                    _, binary_image = cv.threshold(
                        red_channel_cropped, manual_threshold, 255, cv.THRESH_BINARY_INV
                    )
            else:
                # 备用方案
                _, binary_image = cv.threshold(
                    red_channel_cropped, manual_threshold, 255, cv.THRESH_BINARY_INV
                )

        return (
            True,
            cropped_frame,
            red_channel_cropped,
            mask_cropped,
            binary_image,
            current_bbox_area,
        )

    except Exception as e:
        print(f"Error processing frame: {e}")
        return False, None, None, None, None, None


# =================== 主应用程序 ===================


class ThresholdAfterCropTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Threshold Test Tool for Cropped Region (After Segmentation)")
        self.root.geometry("1600x900")

        # 阈值方法管理器
        self.threshold_manager = ThresholdMethodManager()

        # Frame management
        self.frames_buffer = {}
        self.buffer_size = 20
        self.current_index = 0
        self.total_frames = 0
        self.cap = None

        # UI state
        self.current_threshold = 128
        self.threshold_method = "manual"
        self.all_thresholds = {}
        self.histogram_data = None
        self.cropped_red_channel = None

        self.setup_ui()
        self.open_video()

    def setup_ui(self):
        # Top control bar
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        # Frame navigation
        ttk.Button(control_frame, text="Previous", command=self.prev_frame).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Next", command=self.next_frame).pack(
            side=tk.LEFT, padx=5
        )

        # Jump to frame input
        ttk.Label(control_frame, text="Jump to frame:").pack(side=tk.LEFT, padx=(20, 5))
        self.jump_entry = ttk.Entry(control_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Jump", command=self.jump_to_frame).pack(
            side=tk.LEFT, padx=5
        )

        self.frame_label = ttk.Label(control_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=20)

        # Threshold method selection
        method_frame = ttk.Frame(self.root)
        method_frame.pack(pady=5)

        ttk.Label(method_frame, text="Threshold Method:").pack(side=tk.LEFT, padx=5)
        self.method_var = tk.StringVar(value="manual")

        # 动态获取自适应方法列表
        adaptive_keys = self.threshold_manager.get_adaptive_method_keys()
        method_values = ["manual"] + adaptive_keys

        method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=method_values,
            state="readonly",
            width=15,
        )
        method_combo.pack(side=tk.LEFT, padx=5)
        method_combo.bind("<<ComboboxSelected>>", self.on_method_change)

        # Threshold control (only for manual method)
        threshold_frame = ttk.Frame(self.root)
        threshold_frame.pack(pady=5)

        ttk.Label(threshold_frame, text="Manual Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.IntVar(value=128)
        self.threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0,
            to=255,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=300,
            command=self.on_threshold_change,
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=5)

        self.threshold_label = ttk.Label(threshold_frame, text="128")
        self.threshold_label.pack(side=tk.LEFT, padx=5)

        # Quick threshold buttons - 动态生成
        button_frame = ttk.Frame(threshold_frame)
        button_frame.pack(side=tk.LEFT, padx=10)

        self.threshold_buttons = {}
        button_methods = self.threshold_manager.get_button_methods()

        for i, method in enumerate(button_methods):
            row = i // 5  # 每行5个按钮
            col = i % 5
            btn = ttk.Button(
                button_frame,
                text=method.name,
                width=10,
                command=lambda m=method.name: self.use_method_threshold(m),
            )
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.threshold_buttons[method.name] = btn

        # Status
        self.status_label = ttk.Label(self.root, text="Status: Loading video...")
        self.status_label.pack(pady=5)

        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side - Images (2 images side by side)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Images frame
        images_frame = ttk.Frame(left_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)

        # Cropped region image (left)
        left_image_frame = ttk.Frame(images_frame)
        left_image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(
            left_image_frame, text="Cropped Valid Region", font=("Arial", 12)
        ).pack()
        self.cropped_label = ttk.Label(left_image_frame)
        self.cropped_label.pack()

        # Binary image (right)
        right_image_frame = ttk.Frame(images_frame)
        right_image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.binary_title_label = ttk.Label(
            right_image_frame,
            text="Binary Image (Manual Threshold)",
            font=("Arial", 12),
        )
        self.binary_title_label.pack()
        self.binary_label = ttk.Label(right_image_frame)
        self.binary_label.pack()

        # Right side - Histogram and threshold info
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Histogram
        ttk.Label(
            right_frame,
            text="Cropped Red Channel Histogram with Multiple Thresholds",
            font=("Arial", 12),
        ).pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Threshold values display
        threshold_info_frame = ttk.Frame(right_frame)
        threshold_info_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            threshold_info_frame,
            text="Calculated Thresholds:",
            font=("Arial", 11, "bold"),
        ).pack()

        self.threshold_info_text = tk.Text(
            threshold_info_frame, height=8, width=30, font=("Arial", 9)
        )
        scrollbar = ttk.Scrollbar(
            threshold_info_frame,
            orient="vertical",
            command=self.threshold_info_text.yview,
        )
        self.threshold_info_text.configure(yscrollcommand=scrollbar.set)

        self.threshold_info_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind keyboard events
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.focus_set()

    def on_method_change(self, event=None):
        """当阈值方法改变时调用"""
        self.threshold_method = self.method_var.get()

        # 更新UI标题
        if self.threshold_method == "manual":
            title = "Binary Image (Manual Threshold)"
        else:
            adaptive_method = self.threshold_manager.get_adaptive_method_by_key(
                self.threshold_method
            )
            if adaptive_method:
                title = f"Binary Image ({adaptive_method.name})"
            else:
                title = "Binary Image"

        self.binary_title_label.config(text=title)

        # 根据方法启用/禁用手动阈值控件
        if self.threshold_method == "manual":
            self.threshold_scale.config(state="normal")
            for btn in self.threshold_buttons.values():
                btn.config(state="normal")
        else:
            self.threshold_scale.config(state="disabled")
            for btn in self.threshold_buttons.values():
                btn.config(state="disabled")

        # 更新二值化图像
        self.update_binary_image_only()

    def use_method_threshold(self, method_name):
        """Use threshold from specific method"""
        if (
            method_name in self.all_thresholds
            and self.all_thresholds[method_name] is not None
        ):
            self.threshold_var.set(self.all_thresholds[method_name])
            self.current_threshold = self.all_thresholds[method_name]
            self.threshold_label.config(text=str(self.current_threshold))
            self.update_histogram()
            self.update_binary_image_only()

    def open_video(self):
        """Open video and get basic info"""
        try:
            self.cap = cv.VideoCapture(VIDEO_PATH)

            if not self.cap.isOpened():
                self.status_label.config(
                    text=f"Error: Cannot open video file {VIDEO_PATH}"
                )
                return

            self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            print(f"Video opened: {VIDEO_PATH}, Total frames: {self.total_frames}")

            self.status_label.config(
                text=f"Video: {VIDEO_PATH} - Total {self.total_frames} frames"
            )

            # Load initial frames
            self.load_frames_around(0)
            self.update_display()

        except Exception as e:
            self.status_label.config(text=f"Failed to open video: {str(e)}")
            print(f"Error opening video: {e}")

    def load_frames_around(self, center_frame):
        """Load frames around the center frame"""
        if not self.cap:
            return

        half_buffer = self.buffer_size // 2
        start_frame = max(0, center_frame - half_buffer)
        end_frame = min(self.total_frames, center_frame + half_buffer)

        print(
            f"Loading frames {start_frame} to {end_frame-1} around frame {center_frame}"
        )

        keys_to_remove = []
        for frame_idx in self.frames_buffer.keys():
            if frame_idx < start_frame or frame_idx >= end_frame:
                keys_to_remove.append(frame_idx)

        for key in keys_to_remove:
            del self.frames_buffer[key]

        for frame_idx in range(start_frame, end_frame):
            if frame_idx not in self.frames_buffer:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if ret:
                    self.frames_buffer[frame_idx] = frame
                else:
                    print(f"Failed to read frame {frame_idx}")

    def get_frame(self, frame_index):
        """Get a specific frame, loading if necessary"""
        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        if frame_index not in self.frames_buffer:
            self.load_frames_around(frame_index)

        return self.frames_buffer.get(frame_index)

    def on_threshold_change(self, value):
        self.current_threshold = int(float(value))
        self.threshold_label.config(text=str(self.current_threshold))
        self.update_histogram()
        self.update_binary_image_only()

    def update_binary_image_only(self):
        """只更新二值化图像，不重新处理整个帧"""
        if self.cropped_red_channel is None:
            return

        # 根据选择的方法进行二值化
        if self.threshold_method == "manual":
            _, binary_image = cv.threshold(
                self.cropped_red_channel,
                self.current_threshold,
                255,
                cv.THRESH_BINARY_INV,
            )
        else:
            # 使用阈值管理器中的自适应方法
            adaptive_method = self.threshold_manager.get_adaptive_method_by_key(
                self.threshold_method
            )
            if adaptive_method:
                binary_image = adaptive_method.apply_threshold(self.cropped_red_channel)
            else:
                # 备用方案
                _, binary_image = cv.threshold(
                    self.cropped_red_channel,
                    self.current_threshold,
                    255,
                    cv.THRESH_BINARY_INV,
                )

        # 显示二值化图像
        self.display_image(
            cv.cvtColor(binary_image, cv.COLOR_GRAY2RGB), self.binary_label, 400
        )

    def jump_to_frame(self):
        try:
            target_frame = int(self.jump_entry.get()) - 1
            if 0 <= target_frame < self.total_frames:
                self.current_index = target_frame
                self.load_frames_around(target_frame)
                self.update_display()
            else:
                print(f"Frame number out of range: 1-{self.total_frames}")
        except ValueError:
            print("Please enter valid frame number")

    def update_histogram(self):
        if self.histogram_data is None:
            return

        self.ax.clear()

        # Plot histogram
        self.ax.plot(
            range(256),
            self.histogram_data,
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Histogram",
        )

        # Plot threshold lines using method colors
        for method_name, threshold in self.all_thresholds.items():
            if threshold is not None and 0 <= threshold <= 255:
                method = self.threshold_manager.get_method(method_name)
                color = method.color if method else "black"
                self.ax.axvline(
                    x=threshold,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{method_name}: {threshold}",
                )

        # Highlight current manual threshold
        self.ax.axvline(
            x=self.current_threshold,
            color="black",
            linestyle="-",
            linewidth=3,
            alpha=0.9,
            label=f"Current Manual: {self.current_threshold}",
        )

        self.ax.set_xlabel("Pixel Value")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("Cropped Red Channel Histogram with Multiple Thresholds")

        # Create legend with smaller font
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def update_threshold_info_display(self):
        """Update the text display with threshold information"""
        self.threshold_info_text.delete(1.0, tk.END)

        # 按类别分组显示
        categories = self.threshold_manager.get_methods_by_category()
        info_lines = []

        for category, methods in categories.items():
            info_lines.append(f"=== {category} ===")
            for method in methods:
                if not method.is_adaptive:  # 只显示非自适应方法的阈值
                    threshold = self.all_thresholds.get(method.name)
                    if threshold is not None:
                        info_lines.append(f"{method.name}: {threshold}")
                    else:
                        info_lines.append(f"{method.name}: Failed")
            info_lines.append("")  # 空行分隔

        info_lines.append("=== Current Settings ===")
        info_lines.append(f"Manual Threshold: {self.current_threshold}")
        info_lines.append(f"Active Method: {self.threshold_method}")

        self.threshold_info_text.insert(1.0, "\n".join(info_lines))

    def update_display(self):
        """完整更新显示，包括加载帧和计算所有阈值"""
        frame = self.get_frame(self.current_index)
        if frame is None:
            print(f"Cannot get frame {self.current_index}")
            return

        # 按照轮廓检测流程处理帧：先分割，再裁剪，最后计算阈值
        (
            success,
            cropped_frame,
            cropped_red_channel,
            mask_cropped,
            binary_image,
            bbox_area,
        ) = process_frame_for_threshold_test(
            frame, self.current_threshold, self.threshold_method, self.threshold_manager
        )

        if not success:
            self.status_label.config(
                text="Status: Failed to process frame - no valid region found"
            )
            return

        # 保存裁剪后的红色通道用于后续的阈值更新
        self.cropped_red_channel = cropped_red_channel

        # 在裁剪后的红色通道上计算各种阈值
        self.all_thresholds = self.threshold_manager.calculate_all_thresholds(
            cropped_red_channel, mask_cropped
        )

        # 获取直方图数据
        if mask_cropped is not None and mask_cropped.shape == cropped_red_channel.shape:
            masked_channel = cv.bitwise_and(
                cropped_red_channel, cropped_red_channel, mask=mask_cropped
            )
        else:
            masked_channel = cropped_red_channel
        self.histogram_data = cv.calcHist(
            [masked_channel], [0], None, [256], [0, 256]
        ).flatten()
        self.histogram_data[0] = 0

        # Update histogram
        self.update_histogram()

        # Update threshold info display
        self.update_threshold_info_display()

        # Update status
        self.frame_label.config(
            text=f"Frame: {self.current_index + 1}/{self.total_frames}"
        )

        if cropped_frame is not None:
            h, w = cropped_frame.shape[:2]
            status_text = f"Success - Cropped region: {w}x{h}, Area: {bbox_area}, Method: {self.threshold_method}"
        else:
            status_text = f"Processing failed - Method: {self.threshold_method}"

        self.status_label.config(text=f"Status: {status_text}")

        # Display images
        if cropped_frame is not None:
            self.display_image(cropped_frame, self.cropped_label, 400)
        if binary_image is not None:
            self.display_image(
                cv.cvtColor(binary_image, cv.COLOR_GRAY2RGB), self.binary_label, 400
            )

        print(f"Buffer contains frames: {sorted(self.frames_buffer.keys())}")

    def display_image(self, cv_image, label, max_size=300):
        height, width = cv_image.shape[:2]

        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv.resize(cv_image, (new_width, new_height))

        if len(cv_image.shape) == 3:
            rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)

        label.config(image=photo)
        label.image = photo

    def next_frame(self):
        if self.current_index < self.total_frames - 1:
            self.current_index += 1
            half_buffer = self.buffer_size // 2
            if (
                self.current_index + half_buffer not in self.frames_buffer
                and self.current_index + half_buffer < self.total_frames
            ):
                self.load_frames_around(self.current_index)
            self.update_display()

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            half_buffer = self.buffer_size // 2
            if (
                self.current_index - half_buffer not in self.frames_buffer
                and self.current_index - half_buffer >= 0
            ):
                self.load_frames_around(self.current_index)
            self.update_display()

    def __del__(self):
        if self.cap:
            self.cap.release()


def main():
    root = tk.Tk()
    app = ThresholdAfterCropTester(root)
    root.mainloop()


if __name__ == "__main__":
    main()
