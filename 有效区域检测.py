import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm


def process_frame(frame, previous_bbox_area):
    red_channel = frame[:, :, 2]
    threshold = calculate_threshold(red_channel)
    kernel_size = (5, 5)
    bbox = segment_image(red_channel, threshold, kernel_size)
    current_bbox_area = bbox[2] * bbox[3]

    if previous_bbox_area > 0 and current_bbox_area > previous_bbox_area * 1.5:
        kernel_size = (7, 7)
        bbox = segment_image(red_channel, threshold, kernel_size)

    x, y, w, h = bbox
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return frame, w * h


def segment_image(src, threshold, kernel_size):
    _, src_bin = cv.threshold(src, threshold, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
    coords = cv.findNonZero(src_bin)
    return cv.boundingRect(coords)


def calculate_threshold(red_channel):
    hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
    peaks, _ = find_peaks(hist)
    sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
    second_peak_index = sorted_peaks[1]
    min_val_after_peak = np.argmin(hist[second_peak_index:]) + second_peak_index
    return min_val_after_peak


def process_video(input_path, output_path, fps=24):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    previous_bbox_area = 0
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}.")
            break
        processed_frame, previous_bbox_area = process_frame(frame, previous_bbox_area)
        out.write(processed_frame)

    cap.release()
    out.release()


def main():
    video_path = "/home/july/physic/test/真实场景.mp4"
    output_path = "区域检测.mp4"
    process_video(video_path, output_path)


if __name__ == "__main__":
    main()
