import cv2 as cv
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks


def calculate_threshold(red_channel, mask=None):
    """
    计算阈值，阈值为出现频率最高的两个峰值之间的最低值。

    参数:
    red_channel (numpy.ndarray): 输入的红色通道图像。
    mask (numpy.ndarray, optional): 可选的掩码图像，用于指定计算阈值的区域，形状必须与 red_channel 相同。

    返回:
    int: 计算得到的阈值。

    异常:
    ValueError: 如果无法找到两个峰值，则抛出此异常。
    """
    if mask is not None and mask.shape == red_channel.shape:
        red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)
    hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
    hist[0] = 0
    peaks, _ = find_peaks(hist)
    sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)

    # 找到出现频率最高的两个峰值
    if len(sorted_peaks) < 2:
        raise ValueError("无法找到两个峰值")

    second_peak_index = min(sorted_peaks[:2])
    min_val_after_peak = np.argmin(hist[second_peak_index:]) + second_peak_index
    return min_val_after_peak


def segment_image(src, threshold, kernel_size):
    """
    对图像进行分割，并返回分割后的边界框和掩码。

    参数:
    src (numpy.ndarray): 输入的源图像。
    threshold (int): 用于二值化的阈值。
    kernel_size (tuple): 用于形态学操作的内核大小。

    返回:
    tuple: 包含两个元素的元组：
        - (x, y, w, h) (tuple): 分割后的边界框，包含左上角坐标 (x, y) 和宽度、高度 (w, h)。
        - mask (numpy.ndarray or None): 分割后的掩码图像，如果没有找到轮廓则为 None。

    异常:
    无
    """
    _, src_bin = cv.threshold(src, threshold, 255, cv.THRESH_BINARY)
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

    参数:
    frame (numpy.ndarray): 输入的图像。
    bbox (tuple): 边界框，包含左上角坐标 (x, y) 和宽度、高度 (w, h)。

    返回:
    numpy.ndarray: 裁剪后的图像（深拷贝）。
    """
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w].copy()


def classify_contour(contour, binary_image, thickness=20):
    """
    根据轮廓内像素的平均值对轮廓进行分类。

    参数:
    contour (numpy.ndarray): 输入的轮廓。
    binary_image (numpy.ndarray): 输入的二值化图像。
    thickness (int, optional): 用于绘制轮廓的厚度。默认值为 20。

    返回:
    tuple: 轮廓的颜色，根据平均值分类为绿色 (0, 255, 0) 或蓝色 (255, 0, 0)。
    """
    # 创建与 binary_image 大小相同的掩码
    mask = np.zeros_like(binary_image)
    cv.drawContours(mask, [contour], -1, 255, thickness)

    # 创建轮廓内涂满的图像
    filled_contour_image = np.zeros_like(binary_image)
    cv.drawContours(filled_contour_image, [contour], -1, 255, thickness=cv.FILLED)

    # 获取在掩码图像和轮廓内涂满的图像中都为255的点
    combined_mask = cv.bitwise_and(mask, filled_contour_image)

    # 获取这些点在原来的二值化图像中的颜色
    masked_points = binary_image[combined_mask == 255]

    # 计算这些点的像素平均值
    avg_color = np.mean(masked_points)

    # 根据平均值对边缘分类
    if avg_color < 128:
        color = (0, 255, 0)  # 红色圆环外侧边缘为绿色
    else:
        color = (255, 0, 0)  # 红色圆环内侧边缘为蓝色
    return color


def process_frame(
    frame,
    previous_bbox_area=0,
    kernel_size=(9, 9),
    center_color_threshold=0,
    min_length=500,
    circularity_threshold=0.7,
    area_threshold=1000,
    region_size=10,
    center_circle_diameter=50,
    center_circle_thickness=2,
):
    """
    处理单帧图像，检测轮廓并计算公共中心。

    必填参数:
    frame (numpy.ndarray): 输入的图像帧。

    可选参数:
    计算相关:
    previous_bbox_area (float): 前一帧的边界框面积。默认值为 0。
    kernel_size (tuple): 用于形态学操作的内核大小。默认值为 (9, 9)。
    center_color_threshold (int): 判断中心颜色的阈值。默认值为 0。
    min_length (int): 轮廓的最小长度阈值。默认值为 500。
    circularity_threshold (float): 轮廓的圆形度阈值。默认值为 0.7。
    area_threshold (int): 视为同一轮廓的最大面积差距阈值。默认值为 1000。
    region_size (int): 判断中心颜色的判定区域大小。默认值为 10。

    绘图相关:
    center_circle_diameter (int): 画出公共中心时的直径。默认值为 50。
    center_circle_thickness (int): 画出公共中心时的线宽。默认值为 2。

    返回:
    tuple: 包含四个元素的元组：
        - processed_frame (numpy.ndarray): 处理后的图像帧。
        - current_bbox_area (float): 当前帧的边界框面积。
        - contours_info (list): 包含轮廓面积和颜色信息的列表。
        - center_color (int or None): 公共中心的颜色，红色为 255，黑色为 0，无效为 None。
    """
    red_channel = frame[:, :, 2]
    threshold = calculate_threshold(red_channel)

    bbox, mask = segment_image(red_channel, threshold, (5, 5))
    current_bbox_area = bbox[2] * bbox[3]

    if previous_bbox_area > 0 and current_bbox_area > previous_bbox_area * 1.5:
        bbox, mask = segment_image(red_channel, threshold, (7, 7))
        current_bbox_area = bbox[2] * bbox[3]

    if current_bbox_area == 0:
        return frame, previous_bbox_area, [], None

    red_channel = crop_to_bbox(red_channel, bbox)
    mask = crop_to_bbox(mask, bbox)
    cropped_frame = crop_to_bbox(frame, bbox)

    threshold = calculate_threshold(red_channel, mask)
    _, binary_image = cv.threshold(red_channel, threshold, 255, cv.THRESH_BINARY_INV)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    processed_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    processed_image = cv.morphologyEx(processed_image, cv.MORPH_CLOSE, kernel)

    # 轮廓发现
    edges = cv.Canny(processed_image, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # 计算每个轮廓的面积和圆形度
    max_radius = min(bbox[2:]) // 2
    max_area = np.pi * (max_radius**2)

    # 按照面积顺序处理轮廓
    contours = sorted(contours, key=cv.contourArea)

    contours_info = []
    filtered_contours = []
    pre_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter < min_length or area > max_area:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity >= circularity_threshold:
            if pre_area != 0 and abs(area - pre_area) < area_threshold:
                continue
            pre_area = area
            color = classify_contour(contour, binary_image)
            contours_info.append((area, color))
            filtered_contours.append(contour)
            cv.drawContours(cropped_frame, [contour], -1, color, 2)

    # 计算公共中心
    if filtered_contours:
        centers = [cv.moments(contour) for contour in filtered_contours]
        cx = int(sum(m["m10"] / m["m00"] for m in centers) / len(centers))
        cy = int(sum(m["m01"] / m["m00"] for m in centers) / len(centers))

        # 判断公共中心邻域范围内的像素点颜色
        region = binary_image[
            max(0, cy - region_size) : min(binary_image.shape[0], cy + region_size),
            max(0, cx - region_size) : min(binary_image.shape[1], cx + region_size),
        ]
        avg_color = np.mean(region)

        if avg_color < 128 - center_color_threshold:
            center_color = 255  # 红色
            border_color = (0, 255, 0)  # 绿色
        elif avg_color > 128 + center_color_threshold:
            center_color = 0  # 黑色
            border_color = (255, 0, 0)  # 蓝色
        else:
            center_color = None  # 无效
            border_color = (255, 255, 255)  # 白色

        # 画出公共中心
        cv.circle(
            cropped_frame,
            (cx, cy),
            center_circle_diameter // 2,
            border_color,
            center_circle_thickness,
        )

        return cropped_frame, current_bbox_area, contours_info, center_color
    return cropped_frame, current_bbox_area, contours_info, None


class ContourDetection:
    DEFAULT_SETTINGS = {
        "kernel_size": (9, 9),
        "center_color_threshold": 0,
        "min_length": 500,
        "circularity_threshold": 0.7,
        "area_threshold": 1000,
        "region_size": 10,
        "center_circle_diameter": 50,
        "center_circle_thickness": 2,
        "fps": 10,
        "mode": "zoom_in",
    }

    def __init__(self, settings=None):
        """
        初始化 ContourDetection 类。

        参数:
        settings (dict, optional): 包含设置参数的字典。默认值为 None。
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        self.reset_counters()
        self.reset_state()

    def reset_counters(self):
        """
        重置计数器。
        """
        self.green_count = 0
        self.blue_count = 0
        self.white_count = 0
        self.count_value = 0
        self.total_frames = 0

    def reset_state(self):
        """
        重置状态属性。
        """
        self.previous_bbox_area = 0
        self.previous_contours_info = []
        self.previous_center_color = None
        self.status = None

    def update(self, frame):
        """
        更新并处理当前帧。

        参数:
        frame (numpy.ndarray): 当前最新的帧。
        """
        self.total_frames += 1
        (
            processed_frame,
            self.previous_bbox_area,
            current_contours_info,
            current_center_color,
        ) = process_frame(
            frame,
            self.previous_bbox_area,
            self.settings["kernel_size"],
            self.settings["center_color_threshold"],
            self.settings["min_length"],
            self.settings["circularity_threshold"],
            self.settings["area_threshold"],
            self.settings["region_size"],
            self.settings["center_circle_diameter"],
            self.settings["center_circle_thickness"],
        )

        # 匹配轮廓并判断放大或缩小
        scale_count = 0
        area_threshold = 50000  # 面积变化阈值
        if self.previous_contours_info and current_contours_info:
            for prev_area, prev_color in self.previous_contours_info:
                for curr_area, curr_color in current_contours_info:
                    if (
                        prev_color == curr_color
                        and abs(curr_area - prev_area) < area_threshold
                    ):
                        if curr_area > prev_area:
                            scale_count += 1
                        else:
                            scale_count -= 1
        if scale_count > 0:
            border_color = (0, 255, 0)  # 绿色
            self.green_count += 1
            self.status = "zoom_out"
        elif scale_count < 0:
            border_color = (255, 0, 0)  # 蓝色
            self.blue_count += 1
            self.status = "zoom_in"
        else:
            border_color = (255, 255, 255)  # 默认白色
            self.white_count += 1

        border_thickness = min(10 + abs(scale_count), 50)

        # 判断计数模式和更新计数值
        if (
            self.previous_center_color is not None
            and current_center_color is not None
            and self.status is not None
        ):
            if (
                self.previous_center_color == 255
                and current_center_color == 0
                and self.settings["mode"] == self.status
            ):
                self.count_value += 1
            if (
                self.previous_center_color == 0
                and current_center_color == 255
                and self.settings["mode"] != self.status
            ):
                self.count_value -= 1

        # 给画面增加边框
        processed_frame = cv.copyMakeBorder(
            processed_frame,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv.BORDER_CONSTANT,
            value=border_color,
        )

        frame[: processed_frame.shape[0], : processed_frame.shape[1]] = processed_frame

        # 在整体的右上角打印计数值
        cv.putText(
            frame,
            str(self.count_value),
            (frame.shape[1] - 400, 300),  # 右上角位置
            cv.FONT_HERSHEY_SIMPLEX,
            5,
            (255, 255, 255),
            10,
        )

        self.previous_contours_info = current_contours_info
        self.previous_center_color = current_center_color

        return frame

    def process_video(self, input_path, output_path):
        """
        处理视频文件，用于测试效果。

        参数:
        input_path (str): 输入视频文件路径。
        output_path (str): 输出视频文件路径。
        """
        cap = cv.VideoCapture(input_path)
        # ret, frame = cap.read()
        # original_height, original_width = frame.shape[:2]
        # fourcc = cv.VideoWriter_fourcc(*"mp4v")
        # out = cv.VideoWriter(
        # output_path, fourcc, self.settings["fps"], (original_width, original_height)
        # )
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        self.reset_counters()
        self.reset_state()

        for i in tqdm(range(frame_count), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.update(frame)
            # out.write(frame)

        cap.release()
        # out.release()

        # 打印统计信息
        total_frames = self.green_count + self.blue_count + self.white_count
        print(
            f"绿色边框帧数: {self.green_count} ({self.green_count / total_frames * 100:.2f}%)"
        )
        print(
            f"蓝色边框帧数: {self.blue_count} ({self.blue_count / total_frames * 100:.2f}%)"
        )
        print(
            f"白色边框帧数: {self.white_count} ({self.white_count / total_frames * 100:.2f}%)"
        )
        print(f"总圈数: {self.count_value}")


def main():
    video_path = "/home/july/physic/test/真实场景.mp4"
    output_path = "轮廓检测+计数（python版）.mp4"

    # 创建 ContourDetection 实例
    contour_detection = ContourDetection()

    # 处理视频文件
    contour_detection.process_video(video_path, output_path)


if __name__ == "__main__":
    main()
