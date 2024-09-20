import cv2 as cv
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks


def calculate_threshold(red_channel, mask=None):
    if mask is not None:
        red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)
    hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
    hist[0] = 0
    peaks, _ = find_peaks(hist)
    sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
    second_peak_index = sorted_peaks[1]
    min_val_after_peak = np.argmin(hist[second_peak_index:]) + second_peak_index
    return min_val_after_peak


def segment_image(src, threshold, kernel_size):
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
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w]


def classify_contour(contour, binary_image, thickness=20):
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

    # 根据平均值分类
    if avg_color < 128:
        color = (0, 255, 0)  # 绿色
    else:
        color = (255, 0, 0)  # 蓝色

    return color


def process_frame(frame, previous_bbox_area, kernel_size=(9, 9)):
    red_channel = frame[:, :, 2]
    threshold = calculate_threshold(red_channel)

    bbox, mask = segment_image(red_channel, threshold, (5, 5))
    current_bbox_area = bbox[2] * bbox[3]

    if previous_bbox_area > 0 and current_bbox_area > previous_bbox_area * 1.5:
        bbox, mask = segment_image(red_channel, threshold, (7, 7))
        current_bbox_area = bbox[2] * bbox[3]

    if current_bbox_area == 0:
        return frame, previous_bbox_area, []

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
    min_length = 500  # 设置最小长度阈值
    max_radius = min(bbox[2:]) // 2
    max_area = np.pi * (max_radius**2)
    circularity_threshold = 0.7  # 设置圆形度阈值
    area_threshold = 1000  # 面积差距阈值

    # 按照面积顺序处理轮廓
    contours = sorted(contours, key=cv.contourArea)

    contours_info = []
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
            cv.drawContours(cropped_frame, [contour], -1, color, 2)

    return cropped_frame, current_bbox_area, contours_info


def process_video(input_path, output_path, fps=24):
    cap = cv.VideoCapture(input_path)
    ret, frame = cap.read()
    target_height, target_width = 2000, 1400
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    bbox_area = 0
    previous_contours_info = []
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # 初始化计数器
    green_count = 0
    blue_count = 0
    white_count = 0

    for i in tqdm(range(frame_count), desc="Processing video"):
        _, frame = cap.read()
        processed_frame, bbox_area, current_contours_info = process_frame(
            frame, bbox_area
        )

        # 匹配轮廓并判断放大或缩小
        scale_count = 0
        area_threshold = 50000  # 面积变化阈值
        if previous_contours_info and current_contours_info:
            for prev_area, prev_color in previous_contours_info:
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
            green_count += 1
        elif scale_count < 0:
            border_color = (255, 0, 0)  # 蓝色
            blue_count += 1
        else:
            border_color = (255, 255, 255)  # 默认白色
            white_count += 1

        border_thickness = min(10 + abs(scale_count), 50)

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

        # 调整帧大小到目标大小
        resized_frame = cv.resize(processed_frame, (target_width, target_height))
        out.write(resized_frame)

        previous_contours_info = current_contours_info

    cap.release()
    out.release()

    # 打印统计信息
    total_frames = green_count + blue_count + white_count
    print(f"绿色边框帧数: {green_count} ({green_count / total_frames * 100:.2f}%)")
    print(f"蓝色边框帧数: {blue_count} ({blue_count / total_frames * 100:.2f}%)")
    print(f"白色边框帧数: {white_count} ({white_count / total_frames * 100:.2f}%)")


def main():
    video_path = "/home/july/physic/test/真实场景.mp4"
    output_path = "轮廓检测.mp4"

    # 处理视频文件
    process_video(video_path, output_path, 10)


if __name__ == "__main__":
    main()
