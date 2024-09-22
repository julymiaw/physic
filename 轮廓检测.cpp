#include <opencv2/opencv.hpp>
#include <emscripten/bind.h>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>

// 计算阈值函数
int calculate_threshold(const cv::Mat &red_channel, const cv::Mat &mask = cv::Mat()) {
    cv::Mat masked_red_channel;
    if (!mask.empty() && mask.size() == red_channel.size()) {
        cv::bitwise_and(red_channel, red_channel, masked_red_channel, mask);
    } else {
        masked_red_channel = red_channel;
    }

    // 计算直方图
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::Mat hist;
    cv::calcHist(&masked_red_channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    hist.at<float>(0) = 0; // 忽略第一个bin

    // 找到峰值
    std::vector<int> peaks;
    for (int i = 1; i < histSize - 1; ++i) {
        if (hist.at<float>(i) > hist.at<float>(i - 1) && hist.at<float>(i) > hist.at<float>(i + 1)) {
            peaks.push_back(i);
        }
    }

    if (peaks.size() < 2) {
        throw std::runtime_error("无法找到两个峰值");
    }

    std::sort(peaks.begin(), peaks.end(), [&hist](int a, int b) {
        return hist.at<float>(a) > hist.at<float>(b);
    });

    int second_peak_index = std::min(peaks[0], peaks[1]);
    int min_val_after_peak = std::distance(hist.begin<float>(), std::min_element(hist.begin<float>() + second_peak_index, hist.end<float>()));
    return min_val_after_peak;
}

// 图像分割函数
std::pair<cv::Rect, cv::Mat> segment_image(const cv::Mat &src, int threshold, const cv::Size &kernel_size) {
    cv::Mat src_bin;
    cv::threshold(src, src_bin, threshold, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    cv::morphologyEx(src_bin, src_bin, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(src_bin, src_bin, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(src_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        cv::Mat coords;
        cv::findNonZero(src_bin, coords);
        return {cv::boundingRect(coords), cv::Mat()};
    }

    auto contour = *std::max_element(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, 1, cv::FILLED);
    return {cv::boundingRect(contour), mask};
}

// 裁剪图像函数
cv::Mat crop_to_bbox(const cv::Mat &frame, const cv::Rect &bbox) {
    return frame(bbox).clone();
}

// 轮廓分类函数
cv::Scalar classify_contour(const std::vector<cv::Point> &contour, const cv::Mat &binary_image, int thickness = 20) {
    cv::Mat mask = cv::Mat::zeros(binary_image.size(), CV_8U);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, thickness);

    cv::Mat filled_contour_image = cv::Mat::zeros(binary_image.size(), CV_8U);
    cv::drawContours(filled_contour_image, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);

    cv::Mat combined_mask;
    cv::bitwise_and(mask, filled_contour_image, combined_mask);

    cv::Scalar avg_color = cv::mean(binary_image, combined_mask);

    if (avg_color[0] < 128) {
        return cv::Scalar(0, 255, 0); // 绿色
    } else {
        return cv::Scalar(255, 0, 0); // 蓝色
    }
}

std::tuple<cv::Mat, float, std::vector<std::pair<float, cv::Scalar>>, int> process_frame(
    const cv::Mat &frame,
    float previous_bbox_area,
    const cv::Size &kernel_size,
    int center_color_threshold,
    int min_length,
    float circularity_threshold,
    int area_threshold,
    int region_size,
    int center_circle_diameter,
    int center_circle_thickness) {

    cv::Mat red_channel = frame.clone();
    cv::extractChannel(frame, red_channel, 2);
    int threshold = calculate_threshold(red_channel);

    auto [bbox, mask] = segment_image(red_channel, threshold, cv::Size(5, 5));
    float current_bbox_area = bbox.area();

    if (previous_bbox_area > 0 && current_bbox_area > previous_bbox_area * 1.5) {
        std::tie(bbox, mask) = segment_image(red_channel, threshold, cv::Size(7, 7));
        current_bbox_area = bbox.area();
    }

    if (current_bbox_area == 0) {
        return std::make_tuple(frame, previous_bbox_area, std::vector<std::pair<float, cv::Scalar>>(), -1);
    }

    red_channel = crop_to_bbox(red_channel, bbox);
    mask = crop_to_bbox(mask, bbox);
    cv::Mat cropped_frame = crop_to_bbox(frame, bbox);

    threshold = calculate_threshold(red_channel, mask);
    cv::Mat binary_image;
    cv::threshold(red_channel, binary_image, threshold, 255, cv::THRESH_BINARY_INV);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    cv::morphologyEx(binary_image, binary_image, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary_image, binary_image, cv::MORPH_CLOSE, kernel);

    // 轮廓发现
    cv::Mat edges;
    cv::Canny(binary_image, edges, 50, 150);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 计算每个轮廓的面积和圆形度
    int max_radius = std::min(bbox.width, bbox.height) / 2;
    float max_area = CV_PI * (max_radius * max_radius);

    // 按照面积顺序处理轮廓
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });

    std::vector<std::pair<float, cv::Scalar>> contours_info;
    std::vector<std::vector<cv::Point>> filtered_contours;
    float pre_area = 0;
    for (const auto &contour : contours) {
        float area = cv::contourArea(contour);
        float perimeter = cv::arcLength(contour, true);
        if (perimeter < min_length || area > max_area) {
            continue;
        }
        float circularity = 4 * CV_PI * (area / (perimeter * perimeter));
        if (circularity >= circularity_threshold) {
            if (pre_area != 0 && std::abs(area - pre_area) < area_threshold) {
                continue;
            }
            pre_area = area;
            cv::Scalar color = classify_contour(contour, binary_image);
            contours_info.emplace_back(area, color);
            filtered_contours.push_back(contour);
            cv::drawContours(cropped_frame, std::vector<std::vector<cv::Point>>{contour}, -1, color, 2);
        }
    }

    // 计算公共中心
    if (!filtered_contours.empty()) {
        std::vector<cv::Moments> centers;
        for (const auto &contour : filtered_contours) {
            centers.push_back(cv::moments(contour));
        }
        int cx = static_cast<int>(std::accumulate(centers.begin(), centers.end(), 0.0, [](double sum, const cv::Moments &m) {
                                      return sum + m.m10 / m.m00;
                                  }) /
                                  centers.size());
        int cy = static_cast<int>(std::accumulate(centers.begin(), centers.end(), 0.0, [](double sum, const cv::Moments &m) {
                                      return sum + m.m01 / m.m00;
                                  }) /
                                  centers.size());

        // 判断公共中心邻域范围内的像素点颜色
        cv::Rect region_rect(std::max(0, cx - region_size), std::max(0, cy - region_size), std::min(binary_image.cols - cx + region_size, region_size * 2), std::min(binary_image.rows - cy + region_size, region_size * 2));
        cv::Mat region = binary_image(region_rect);
        double avg_color = cv::mean(region)[0];

        int center_color;
        cv::Scalar border_color;
        if (avg_color < 128 - center_color_threshold) {
            center_color = 255;                   // 红色
            border_color = cv::Scalar(0, 255, 0); // 绿色
        } else if (avg_color > 128 + center_color_threshold) {
            center_color = 0;                     // 黑色
            border_color = cv::Scalar(255, 0, 0); // 蓝色
        } else {
            center_color = -1;                        // 无效
            border_color = cv::Scalar(255, 255, 255); // 白色
        }

        // 画出公共中心
        cv::circle(cropped_frame, cv::Point(cx, cy), center_circle_diameter / 2, border_color, center_circle_thickness);

        return std::make_tuple(cropped_frame, current_bbox_area, contours_info, center_color);
    }

    return std::make_tuple(cropped_frame, current_bbox_area, contours_info, -1);
}

class ContourDetection {
public:
    ContourDetection(const std::map<std::string, int> &settings = DEFAULT_SETTINGS);
    void reset_counters();
    void reset_state();
    uint8_t *update(const uint8_t *frame_data, int width, int height);

private:
    std::map<std::string, int> settings;
    int green_count;
    int blue_count;
    int white_count;
    int count_value;
    int total_frames;
    float previous_bbox_area;
    std::vector<std::pair<float, cv::Scalar>> previous_contours_info;
    int previous_center_color;
    int status;

    static const std::map<std::string, int> DEFAULT_SETTINGS;
};

// 默认设置
const std::map<std::string, int> ContourDetection::DEFAULT_SETTINGS = {
    {"kernel_size", 9},
    {"center_color_threshold", 0},
    {"min_length", 500},
    {"circularity_threshold", 70}, // 0.7 * 100 for integer representation
    {"area_threshold", 1000},
    {"region_size", 10},
    {"center_circle_diameter", 50},
    {"center_circle_thickness", 2},
    {"mode", 0} // 0 for "zoom_in", 1 for "zoom_out"
};

ContourDetection::ContourDetection(const std::map<std::string, int> &settings) {
    this->settings = DEFAULT_SETTINGS;
    for (const auto &setting : settings) {
        this->settings[setting.first] = setting.second;
    }
    reset_counters();
    reset_state();
}

void ContourDetection::reset_counters() {
    green_count = 0;
    blue_count = 0;
    white_count = 0;
    count_value = 0;
    total_frames = 0;
}

void ContourDetection::reset_state() {
    previous_bbox_area = 0;
    previous_contours_info.clear();
    previous_center_color = -1; // Use -1 to represent None
    status = -1;
}

uint8_t *ContourDetection::update(const uint8_t *frame_data, int width, int height) {
    total_frames++;
    cv::Mat rgba_frame(height, width, CV_8UC4, const_cast<uint8_t *>(frame_data));
    cv::Mat frame;
    cv::cvtColor(rgba_frame, frame, cv::COLOR_RGBA2BGR);
    auto [processed_frame, current_bbox_area, current_contours_info, current_center_color] = process_frame(
        frame,
        previous_bbox_area,
        cv::Size(settings["kernel_size"], settings["kernel_size"]),
        settings["center_color_threshold"],
        settings["min_length"],
        settings["circularity_threshold"] / 100.0, // Convert back to float
        settings["area_threshold"],
        settings["region_size"],
        settings["center_circle_diameter"],
        settings["center_circle_thickness"]);

    // 匹配轮廓并判断放大或缩小
    int scale_count = 0;
    int area_threshold = 50000; // 面积变化阈值
    if (!previous_contours_info.empty() && !current_contours_info.empty()) {
        for (const auto &[prev_area, prev_color] : previous_contours_info) {
            for (const auto &[curr_area, curr_color] : current_contours_info) {
                if (prev_color == curr_color && std::abs(curr_area - prev_area) < area_threshold) {
                    if (curr_area > prev_area) {
                        scale_count++;
                    } else {
                        scale_count--;
                    }
                }
            }
        }
    }
    cv::Scalar border_color;
    if (scale_count > 0) {
        border_color = cv::Scalar(0, 255, 0); // 绿色
        green_count++;
        status = 1; // zoom_out
    } else if (scale_count < 0) {
        border_color = cv::Scalar(255, 0, 0); // 蓝色
        blue_count++;
        status = 0; // zoom_in
    } else {
        border_color = cv::Scalar(255, 255, 255); // 默认白色
        white_count++;
        status = -1; // 无效状态
    }

    int border_thickness = std::min(10 + std::abs(scale_count), 50);

    // 判断计数模式和更新计数值
    if (previous_center_color != -1 && current_center_color != -1 && status != -1) {
        if (previous_center_color == 255 && current_center_color == 0 && settings["mode"] == status) {
            count_value++;
        }
        if (previous_center_color == 0 && current_center_color == 255 && settings["mode"] != status) {
            count_value--;
        }
    }

    // 给画面增加边框
    cv::copyMakeBorder(processed_frame, processed_frame, border_thickness, border_thickness, border_thickness, border_thickness, cv::BORDER_CONSTANT, border_color);

    // 在整体的右上角打印计数值
    cv::putText(frame, std::to_string(count_value), cv::Point(frame.cols - 400, 300), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(255, 255, 255), 10);

    // 将处理后的帧复制回原始帧
    processed_frame.copyTo(frame(cv::Rect(0, 0, processed_frame.cols, processed_frame.rows)));

    // 转换为 RGBA 格式
    cv::cvtColor(frame, rgba_frame, cv::COLOR_BGR2RGBA);

    // 分配内存并将数据复制到 Emscripten 堆中
    uint8_t *output_data = new uint8_t[rgba_frame.total() * rgba_frame.elemSize()];
    std::memcpy(output_data, rgba_frame.data, rgba_frame.total() * rgba_frame.elemSize());

    previous_contours_info = current_contours_info;
    previous_center_color = current_center_color;

    return output_data;
}

int main() {
    return 0;
}

// 导出类和函数
EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::class_<ContourDetection>("ContourDetection")
        .constructor<const std::map<std::string, int> &>()
        .function("reset_counters", &ContourDetection::reset_counters)
        .function("reset_state", &ContourDetection::reset_state)
        .function("update", &ContourDetection::update, emscripten::allow_raw_pointers());

    emscripten::register_map<std::string, int>("MapStringInt");
}