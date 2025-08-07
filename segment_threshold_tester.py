import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


# Set video path directly here
# VIDEO_PATH = "test.mp4"  # Change to your video file path
VIDEO_PATH = "真实场景.mp4"


def calculate_multiple_thresholds(red_channel, mask=None):
    """
    Calculate various thresholds using different methods.
    """
    if mask is not None and mask.shape == red_channel.shape:
        red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)

    hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
    hist[0] = 0  # Remove zero values

    thresholds = {}

    # 1. Original peak-based method
    try:
        peaks, _ = find_peaks(hist)
        sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
        if len(sorted_peaks) >= 2:
            second_peak_index = min(sorted_peaks[:2])
            min_val_after_peak = np.argmin(hist[second_peak_index:]) + second_peak_index
            thresholds["Peak-based"] = min_val_after_peak
    except:
        thresholds["Peak-based"] = None

    # 2. Mean threshold
    valid_pixels = red_channel[red_channel > 0]  # Exclude zero values
    if len(valid_pixels) > 0:
        thresholds["Mean"] = int(np.mean(valid_pixels))
    else:
        thresholds["Mean"] = None

    # 3. Median threshold
    if len(valid_pixels) > 0:
        thresholds["Median"] = int(np.median(valid_pixels))
    else:
        thresholds["Median"] = None

    # 4. Otsu's method
    try:
        otsu_thresh, _ = cv.threshold(
            red_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        thresholds["Otsu"] = int(otsu_thresh)
    except:
        thresholds["Otsu"] = None

    # 5. Triangle method (approximation)
    try:
        # Find the peak of the histogram
        max_idx = np.argmax(hist)
        # Find the rightmost non-zero bin
        right_idx = len(hist) - 1
        while right_idx > max_idx and hist[right_idx] == 0:
            right_idx -= 1

        if right_idx > max_idx:
            # Calculate triangle threshold
            max_distance = 0
            triangle_thresh = max_idx
            for i in range(max_idx, right_idx + 1):
                # Distance from point to line
                distance = abs(
                    (hist[right_idx] - hist[max_idx]) * i
                    - (right_idx - max_idx) * hist[i]
                    + right_idx * hist[max_idx]
                    - max_idx * hist[right_idx]
                ) / np.sqrt(
                    (hist[right_idx] - hist[max_idx]) ** 2 + (right_idx - max_idx) ** 2
                )
                if distance > max_distance:
                    max_distance = distance
                    triangle_thresh = i
            thresholds["Triangle"] = triangle_thresh
        else:
            thresholds["Triangle"] = None
    except:
        thresholds["Triangle"] = None

    # 6. Percentile-based thresholds
    if len(valid_pixels) > 0:
        thresholds["25th Percentile"] = int(np.percentile(valid_pixels, 25))
        thresholds["75th Percentile"] = int(np.percentile(valid_pixels, 75))
        thresholds["90th Percentile"] = int(np.percentile(valid_pixels, 90))
    else:
        thresholds["25th Percentile"] = None
        thresholds["75th Percentile"] = None
        thresholds["90th Percentile"] = None

    # 7. Mode (most frequent value)
    if len(hist) > 0:
        mode_idx = np.argmax(hist)
        thresholds["Mode"] = mode_idx
    else:
        thresholds["Mode"] = None

    # 8. Gaussian Mixture Model (2 components)
    try:
        if len(valid_pixels) > 100:  # Need enough samples
            gmm = GaussianMixture(n_components=2, random_state=42)
            data = valid_pixels.reshape(-1, 1)
            gmm.fit(data)
            means = gmm.means_.flatten()
            # Threshold is between the two means
            gmm_thresh = int(np.mean(means))
            thresholds["GMM"] = gmm_thresh
        else:
            thresholds["GMM"] = None
    except:
        thresholds["GMM"] = None

    # 9. Weighted mean (weight by frequency)
    try:
        weighted_sum = np.sum(np.arange(256) * hist)
        total_weight = np.sum(hist)
        if total_weight > 0:
            thresholds["Weighted Mean"] = int(weighted_sum / total_weight)
        else:
            thresholds["Weighted Mean"] = None
    except:
        thresholds["Weighted Mean"] = None

    # 10. Minimum error threshold (Kittler-Illingworth)
    try:
        min_error = float("inf")
        min_error_thresh = 128
        for t in range(1, 255):
            if np.sum(hist[:t]) > 0 and np.sum(hist[t:]) > 0:
                w0 = np.sum(hist[:t])
                w1 = np.sum(hist[t:])
                if w0 > 0 and w1 > 0:
                    mu0 = np.sum(np.arange(t) * hist[:t]) / w0
                    mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1

                    # Calculate variances
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
        thresholds["Min Error"] = min_error_thresh
    except:
        thresholds["Min Error"] = None

    return thresholds, hist


def calculate_threshold(red_channel, mask=None):
    """
    Original threshold calculation for backward compatibility.
    """
    thresholds, hist = calculate_multiple_thresholds(red_channel, mask)
    peaks = None

    # Try to get peaks for visualization
    try:
        peaks_indices, _ = find_peaks(hist)
        sorted_peaks = sorted(peaks_indices, key=lambda x: hist[x], reverse=True)
        peaks = sorted_peaks[:2] if len(sorted_peaks) >= 2 else None
    except:
        peaks = None

    peak_thresh = thresholds.get("Peak-based")
    if peak_thresh is None:
        raise ValueError("Cannot find two peaks")

    return peak_thresh, hist, peaks


def segment_image_with_threshold(src, threshold, kernel_size):
    """
    Use manual threshold to segment image and return bounding box and mask.
    """
    _, src_bin = cv.threshold(src, threshold, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
    coords = cv.findNonZero(src_bin)
    contours, _ = cv.findContours(src_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours or coords is None:
        return None, None, src_bin
    contour = max(contours, key=cv.contourArea)
    bbox = cv.boundingRect(coords)
    mask = np.zeros(src.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 1, thickness=cv.FILLED)
    return bbox, mask, src_bin


def process_frame_with_manual_threshold(frame, threshold, kernel_size=(5, 5)):
    """
    Process frame with manual threshold for segmentation testing.
    """
    try:
        red_channel = frame[:, :, 2]

        bbox, mask, binary_image = segment_image_with_threshold(
            red_channel, threshold, kernel_size
        )

        if bbox is None:
            return False, None, 0, binary_image

        current_bbox_area = bbox[2] * bbox[3]

        if current_bbox_area == 0 or mask is None:
            return False, None, current_bbox_area, binary_image

        return True, bbox, current_bbox_area, binary_image

    except Exception as e:
        print(f"Error processing frame: {e}")
        return False, None, 0, None


class ThresholdTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Threshold Test Tool with Multiple Methods")
        self.root.geometry("1600x900")

        # Frame management
        self.frames_buffer = {}
        self.buffer_size = 20
        self.current_index = 0
        self.total_frames = 0
        self.cap = None

        # UI state
        self.current_threshold = 128
        self.kernel_size = (5, 5)
        self.all_thresholds = {}
        self.histogram_data = None
        self.peaks = None

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

        # Threshold control
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

        # Quick threshold buttons
        button_frame = ttk.Frame(threshold_frame)
        button_frame.pack(side=tk.LEFT, padx=10)

        self.threshold_buttons = {}
        methods = ["Peak-based", "Otsu", "Mean", "Median", "Mode"]
        for i, method in enumerate(methods):
            btn = ttk.Button(
                button_frame,
                text=method,
                width=10,
                command=lambda m=method: self.use_method_threshold(m),
            )
            btn.grid(row=0, column=i, padx=2)
            self.threshold_buttons[method] = btn

        # Kernel size control
        ttk.Label(threshold_frame, text="Kernel Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.kernel_var = tk.IntVar(value=5)
        kernel_spinbox = ttk.Spinbox(
            threshold_frame,
            from_=1,
            to=15,
            textvariable=self.kernel_var,
            width=5,
            command=self.on_kernel_change,
        )
        kernel_spinbox.pack(side=tk.LEFT, padx=5)

        # Status
        self.status_label = ttk.Label(self.root, text="Status: Loading video...")
        self.status_label.pack(pady=5)

        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side - Images (simplified to 2 images side by side)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Images frame
        images_frame = ttk.Frame(left_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)

        # Original image with contour (left)
        left_image_frame = ttk.Frame(images_frame)
        left_image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(
            left_image_frame, text="Original with Segmentation", font=("Arial", 12)
        ).pack()
        self.result_label = ttk.Label(left_image_frame)
        self.result_label.pack()

        # Binary image (right)
        right_image_frame = ttk.Frame(images_frame)
        right_image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(right_image_frame, text="Binary Image", font=("Arial", 12)).pack()
        self.binary_label = ttk.Label(right_image_frame)
        self.binary_label.pack()

        # Right side - Histogram and threshold info
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Histogram
        ttk.Label(
            right_frame,
            text="Red Channel Histogram with Multiple Thresholds",
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

    def use_method_threshold(self, method):
        """Use threshold from specific method"""
        if method in self.all_thresholds and self.all_thresholds[method] is not None:
            self.threshold_var.set(self.all_thresholds[method])
            self.current_threshold = self.all_thresholds[method]
            self.threshold_label.config(text=str(self.current_threshold))
            self.update_histogram()  # 更新直方图中的当前阈值线
            self.update_segmentation_only()  # 只更新分割结果

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
        self.update_segmentation_only()

    def update_segmentation_only(self):
        """只更新分割结果，不重新加载帧或计算直方图"""
        frame = self.get_frame(self.current_index)
        if frame is None:
            return

        # 只处理当前帧的分割，不重新计算阈值
        success, bbox, current_bbox_area, binary_image = (
            process_frame_with_manual_threshold(
                frame, self.current_threshold, self.kernel_size
            )
        )

        # Create result image
        result_frame = frame.copy()

        if success and bbox:
            x, y, w, h = bbox
            cv.rectangle(result_frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            status_text = f"Success - Region: {w}x{h}, Area: {current_bbox_area}, Manual: {self.current_threshold}"
        else:
            status_text = f"Segmentation Failed - Manual: {self.current_threshold}"

        # Update status
        self.status_label.config(text=f"Status: {status_text}")

        self.display_image(result_frame, self.result_label, 400)
        if binary_image is not None:
            self.display_image(
                cv.cvtColor(binary_image, cv.COLOR_GRAY2RGB), self.binary_label, 400
            )

    def on_kernel_change(self):
        kernel_size = self.kernel_var.get()
        self.kernel_size = (kernel_size, kernel_size)
        # 内核大小变化也只需要更新分割结果
        self.update_segmentation_only()

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

        # Color scheme for different threshold methods
        colors = {
            "Peak-based": "red",
            "Otsu": "green",
            "Mean": "purple",
            "Median": "orange",
            "Mode": "brown",
            "Triangle": "pink",
            "GMM": "cyan",
            "Min Error": "gray",
            "25th Percentile": "lightblue",
            "75th Percentile": "lightgreen",
            "90th Percentile": "lightyellow",
            "Weighted Mean": "magenta",
        }

        # Plot threshold lines
        for method, threshold in self.all_thresholds.items():
            if threshold is not None and 0 <= threshold <= 255:
                color = colors.get(method, "black")
                self.ax.axvline(
                    x=threshold,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{method}: {threshold}",
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
        self.ax.set_title("Red Channel Histogram with Multiple Thresholds")

        # Create legend with smaller font
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def update_threshold_info_display(self):
        """Update the text display with threshold information"""
        self.threshold_info_text.delete(1.0, tk.END)

        info_lines = []
        for method, threshold in sorted(self.all_thresholds.items()):
            if threshold is not None:
                info_lines.append(f"{method}: {threshold}")
            else:
                info_lines.append(f"{method}: Failed")

        info_lines.append(f"\nCurrent Manual: {self.current_threshold}")

        self.threshold_info_text.insert(1.0, "\n".join(info_lines))

    def update_display(self):
        """完整更新显示，包括加载帧和计算所有阈值"""
        frame = self.get_frame(self.current_index)
        if frame is None:
            print(f"Cannot get frame {self.current_index}")
            return

        red_channel = frame[:, :, 2]

        # Calculate all thresholds - 只在切换帧时计算
        self.all_thresholds, self.histogram_data = calculate_multiple_thresholds(
            red_channel
        )

        # Update histogram
        self.update_histogram()

        # Update threshold info display
        self.update_threshold_info_display()

        # Process current frame with manual threshold
        success, bbox, current_bbox_area, binary_image = (
            process_frame_with_manual_threshold(
                frame, self.current_threshold, self.kernel_size
            )
        )

        # Create result image
        result_frame = frame.copy()

        if success and bbox:
            x, y, w, h = bbox
            cv.rectangle(result_frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            status_text = f"Success - Region: {w}x{h}, Area: {current_bbox_area}, Manual: {self.current_threshold}"
        else:
            status_text = f"Segmentation Failed - Manual: {self.current_threshold}"

        # Update status
        self.frame_label.config(
            text=f"Frame: {self.current_index + 1}/{self.total_frames}"
        )
        self.status_label.config(text=f"Status: {status_text}")

        # Display only the two main images
        self.display_image(result_frame, self.result_label, 400)
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
    app = ThresholdTester(root)
    root.mainloop()


if __name__ == "__main__":
    main()
