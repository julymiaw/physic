import cv2 as cv
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
import imageio


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


def process_frame(frame, previous_bbox_area, kernel_size=(9, 9)):
    red_channel = frame[:, :, 2]
    threshold = calculate_threshold(red_channel)

    bbox, mask = segment_image(red_channel, threshold, (5, 5))
    current_bbox_area = bbox[2] * bbox[3]

    if previous_bbox_area > 0 and current_bbox_area > previous_bbox_area * 1.5:
        bbox, mask = segment_image(red_channel, threshold, (7, 7))
        current_bbox_area = bbox[2] * bbox[3]

    if current_bbox_area == 0:
        return frame, previous_bbox_area

    red_channel = crop_to_bbox(red_channel, bbox)
    mask = crop_to_bbox(mask, bbox)
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

    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter < min_length or area > max_area:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity >= circularity_threshold:
            filtered_contours.append(contour)

    # 在 cropped_frame 上绘制过滤后的轮廓
    cropped_frame = crop_to_bbox(frame, bbox)
    cv.drawContours(cropped_frame, filtered_contours, -1, (0, 255, 0), 2)  # 绿色轮廓

    return cropped_frame, current_bbox_area


def process_gif(input_path, output_path, fps=24):
    # 读取GIF文件的所有帧
    gif_frames = imageio.mimread(input_path)
    frames = []
    previous_bbox_area = 0

    for frame in tqdm(gif_frames, desc="Processing GIF"):
        # 将帧从RGBA转换为BGR
        frame = cv.cvtColor(frame, cv.COLOR_RGBA2BGR)
        # 处理帧
        processed_frame, previous_bbox_area = process_frame(frame, previous_bbox_area)
        frames.append(processed_frame)

    # 获取帧的宽度和高度
    height, width, _ = frames[0].shape
    # 定义视频编写器
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将处理后的帧写入视频文件
    for frame in frames:
        out.write(frame)

    out.release()


def process_video(input_path, output_path, fps=24):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    target_height, target_width = 2000, 1400
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    previous_bbox_area = 0
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}.")
            break
        processed_frame, previous_bbox_area = process_frame(frame, previous_bbox_area)

        # 调整帧大小到目标大小
        resized_frame = cv.resize(processed_frame, (target_width, target_height))
        out.write(resized_frame)

    cap.release()
    out.release()


def main():
    video_path = "/home/july/physic/test/真实场景.mp4"
    gif_path = "/home/july/physic/test/红色扩张版.gif"
    output_path = "圆识别.mp4"

    # 处理视频文件
    process_video(video_path, output_path, 10)

    # 处理GIF文件
    # process_gif(gif_path, "圆识别_gif.mp4", 10)


if __name__ == "__main__":
    main()
