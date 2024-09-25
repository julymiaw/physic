#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>

// 计数信息结构体
struct CountInfo {
    int green_count;
    int blue_count;
    int white_count;
    int count_value;
    int total_frames;
};

// 状态信息结构体
struct StateInfo {
    float bbox_area;                                   // 有效区域面积
    std::vector<std::pair<float, bool>> contours_info; // 轮廓面积以及是否为圆环外侧边缘
    int center_color;                                  // 圆心颜色, 1 为红色, 0 为黑色
    int status;                                        // 当前帧的状态, 0 为 "收缩", 1 为 "扩张"
    cv::Point center;                                  // 圆心在有效区域中的坐标
};

// 默认设置
const std::map<std::string, int> DEFAULT_SETTINGS = {
    {"kernel_size", 9},            // 形态学分析去除噪点的卷积核大小
    {"center_color_threshold", 0}, // 中心颜色视为未知的阈值
    {"min_length", 500},           // 有效轮廓的最小长度
    {"circularity_threshold", 70}, // 有效轮廓的圆度阈值（百分比）
    {"area_threshold", 1000},      // 有效轮廓的面积阈值
    {"region_size", 10},           // 中心颜色检测区域大小
    {"mode", 0}                    // 0 为 "收缩计数模式", 1 为 "扩张计数模式"
};

class ContourDetection {
public:
    ContourDetection(const std::map<std::string, int> &settings = DEFAULT_SETTINGS);
    void reset_counters();
    void reset_state();
    cv::Mat process_frame(const cv::Mat &frame);
    void print_result();
    int get_count_value() const { return count_info.count_value; }

private:
    std::map<std::string, int> settings;
    CountInfo count_info;
    StateInfo state_info;

    int calculate_threshold(const cv::Mat &red_channel, const cv::Mat &mask = cv::Mat());
    std::pair<cv::Rect, cv::Mat> segment_image(const cv::Mat &src, int threshold, const cv::Size &kernel_size);
    bool classify_contour(const std::vector<cv::Point> &contour, const cv::Mat &binary_image, int thickness = 20);
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
    count_info.green_count = 0;
    count_info.blue_count = 0;
    count_info.white_count = 0;
    count_info.count_value = 0;
    count_info.total_frames = 0;
}

void ContourDetection::reset_state() {
    state_info.bbox_area = 0;
    state_info.contours_info.clear();
    state_info.center_color = -1;
    state_info.status = -1;
    state_info.center = cv::Point(-1, -1);
}

// 计算阈值函数
int ContourDetection::calculate_threshold(const cv::Mat &red_channel, const cv::Mat &mask) {
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
std::pair<cv::Rect, cv::Mat> ContourDetection::segment_image(const cv::Mat &src, int threshold, const cv::Size &kernel_size) {
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

// 轮廓分类函数
bool ContourDetection::classify_contour(const std::vector<cv::Point> &contour, const cv::Mat &binary_image, int thickness) {
    cv::Mat mask = cv::Mat::zeros(binary_image.size(), CV_8U);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, thickness);

    cv::Mat filled_contour_image = cv::Mat::zeros(binary_image.size(), CV_8U);
    cv::drawContours(filled_contour_image, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);

    cv::Mat combined_mask;
    cv::bitwise_and(mask, filled_contour_image, combined_mask);

    cv::Scalar avg_color = cv::mean(binary_image, combined_mask);

    if (avg_color[0] < 128) {
        return true; // 红色圆环外侧边缘
    } else {
        return false; // 红色圆环内侧边缘
    }
}

cv::Mat ContourDetection::process_frame(const cv::Mat &frame) {
    // 提取红色通道
    cv::Mat red_channel = frame.clone();
    cv::extractChannel(frame, red_channel, 2);

    // 提取有效区域
    int threshold = calculate_threshold(red_channel);
    auto [bbox, mask] = segment_image(red_channel, threshold, cv::Size(5, 5));
    float bbox_area = bbox.area();

    // 动态控制形态学操作的核大小
    if (state_info.bbox_area > 0 && bbox_area > state_info.bbox_area * 1.5) {
        std::tie(bbox, mask) = segment_image(red_channel, threshold, cv::Size(7, 7));
        bbox_area = bbox.area();
    }

    // 如果没有有效区域，则当前帧无效(罕见情况！)
    if (bbox_area == 0) {
        return frame;
    }

    // 开始处理当前帧
    count_info.total_frames++;
    state_info.bbox_area = bbox_area;

    // 裁剪图像
    red_channel = red_channel(bbox);
    mask = mask(bbox);
    cv::Mat cropped_frame = frame(bbox);

    // 生成逆二值化图像，此时0代表红色，255代表黑色背景
    threshold = calculate_threshold(red_channel, mask);
    cv::Mat binary_image;
    cv::threshold(red_channel, binary_image, threshold, 255, cv::THRESH_BINARY_INV);

    // 通过形态学操作去除噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(settings["kernel_size"], settings["kernel_size"]));
    cv::morphologyEx(binary_image, binary_image, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary_image, binary_image, cv::MORPH_CLOSE, kernel);

    // 轮廓发现
    cv::Mat edges;
    cv::Canny(binary_image, edges, 50, 150);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 计算理论最大轮廓面积
    int max_radius = std::min(bbox.width, bbox.height) / 2;
    float max_area = CV_PI * (max_radius * max_radius);

    // 按照面积排序
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });

    // 提取有效轮廓信息
    std::vector<std::pair<float, bool>> contours_info;
    std::vector<cv::Moments> centers;
    float pre_area = 0;
    for (const auto &contour : contours) {
        float area = cv::contourArea(contour);
        float perimeter = cv::arcLength(contour, true);
        if (perimeter < settings["min_length"] || area > max_area) {
            continue;
        }
        float circularity = 4 * CV_PI * (area / (perimeter * perimeter));
        if (circularity >= settings["circularity_threshold"] / 100.0) {
            if (pre_area != 0 && std::abs(area - pre_area) < settings["area_threshold"]) {
                continue;
            }
            pre_area = area;
            bool isOuterEdge = classify_contour(contour, binary_image);
            contours_info.emplace_back(area, isOuterEdge);
            centers.push_back(cv::moments(contour));
            cv::Scalar color = isOuterEdge ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
            cv::drawContours(cropped_frame, std::vector<std::vector<cv::Point>>{contour}, -1, color, 2);
        }
    }

    // 未发现轮廓(常见情况)
    if (contours_info.empty()) {
        state_info.contours_info.clear();
        state_info.center_color = -1;
        state_info.status = -1;
        count_info.white_count++;
        return cropped_frame;
    }

    // 计算公共中心
    int cx = static_cast<int>(std::accumulate(centers.begin(), centers.end(), 0.0, [](double sum, const cv::Moments &m) {
                                  return sum + m.m10 / m.m00;
                              }) /
                              centers.size());
    int cy = static_cast<int>(std::accumulate(centers.begin(), centers.end(), 0.0, [](double sum, const cv::Moments &m) {
                                  return sum + m.m01 / m.m00;
                              }) /
                              centers.size());

    // 判断公共中心邻域范围内的像素点颜色
    cv::Rect region_rect(cx - settings["region_size"], cy - settings["region_size"], settings["region_size"] * 2, settings["region_size"] * 2);
    cv::Mat region = binary_image(region_rect);
    double avg_color = cv::mean(region)[0];

    state_info.center = cv::Point(cx, cy);

    int center_color;
    if (avg_color < 128 - settings["center_color_threshold"]) {
        center_color = 1; // 红色
    } else if (avg_color > 128 + settings["center_color_threshold"]) {
        center_color = 0; // 黑色
    } else {
        center_color = -1; // 无效
    }

    // 匹配轮廓并判断放大或缩小
    int scale_count = 0;
    int area_threshold = 50000; // 面积变化阈值
    if (!state_info.contours_info.empty() && !contours_info.empty()) {
        for (const auto &[prev_area, prev_type] : state_info.contours_info) {
            for (const auto &[curr_area, curr_type] : contours_info) {
                if (prev_type == curr_type && std::abs(curr_area - prev_area) < area_threshold) {
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
        count_info.green_count++;
        state_info.status = 1; // zoom_out
    } else if (scale_count < 0) {
        border_color = cv::Scalar(255, 0, 0); // 蓝色
        count_info.blue_count++;
        state_info.status = 0; // zoom_in
    } else {
        border_color = cv::Scalar(255, 255, 255); // 默认白色
        count_info.white_count++;
        // state_info.status = -1; // 无效状态
    }
    cv::copyMakeBorder(cropped_frame, cropped_frame, 10, 10, 10, 10, cv::BORDER_CONSTANT, border_color);

    // 判断计数模式和更新计数值
    if (state_info.center_color != -1 && center_color != -1 && state_info.status != -1) {
        if (state_info.center_color == 1 && center_color == 0 && settings["mode"] == state_info.status) {
            count_info.count_value++;
        }
        if (state_info.center_color == 0 && center_color == 1 && settings["mode"] != state_info.status) {
            count_info.count_value--;
        }
    }

    state_info.contours_info = contours_info;
    state_info.center_color = center_color;
    return cropped_frame;
}

void ContourDetection::print_result() {
    int total_frames = count_info.green_count + count_info.blue_count + count_info.white_count;
    if (total_frames == 0) {
        std::cout << "没有处理任何帧。" << std::endl;
        return;
    }

    std::cout << "绿色边框帧数: " << count_info.green_count << " (" << std::fixed << std::setprecision(2) << (count_info.green_count / static_cast<float>(total_frames) * 100) << "%)" << std::endl;
    std::cout << "蓝色边框帧数: " << count_info.blue_count << " (" << std::fixed << std::setprecision(2) << (count_info.blue_count / static_cast<float>(total_frames) * 100) << "%)" << std::endl;
    std::cout << "白色边框帧数: " << count_info.white_count << " (" << std::fixed << std::setprecision(2) << (count_info.white_count / static_cast<float>(total_frames) * 100) << "%)" << std::endl;
    std::cout << "总圈数: " << count_info.count_value << std::endl;
}

void print_progress(float progress, int current_frame, int total_frames, double elapsed_time, double estimated_time) {
    int bar_width = 50;
    std::cout << "\rProcessing video: ";
    std::cout << std::fixed << std::setprecision(2) << progress << "% [";
    int pos = bar_width * progress / 100.0;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "#";
        else
            std::cout << "-";
    }
    std::cout << "] " << current_frame << "/" << total_frames;
    std::cout << " [" << std::fixed << std::setprecision(2) << elapsed_time << "s<" << estimated_time << "s, " << (elapsed_time / current_frame) << "s/it]";
    std::cout.flush();
}

int main() {
    std::string video_path = "/home/july/physic/test/真实场景.mp4";

    cv::VideoCapture cap(video_path);

    cv::Mat frame;
    int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    ContourDetection contour_detection = ContourDetection();

    auto start_time = std::chrono::steady_clock::now();

    std::vector<std::pair<int, double>> count_timestamps; // 记录计数值和时间戳

    for (int i = 0; i < frame_count; ++i) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        contour_detection.process_frame(frame);

        // 更新进度条
        float progress = (i + 1) / static_cast<float>(frame_count) * 100;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        double estimated_time = (elapsed_time / (i + 1)) * (frame_count - (i + 1));
        print_progress(progress, i + 1, frame_count, elapsed_time, estimated_time);

        // 记录当前计数值和时间戳
        count_timestamps.emplace_back(contour_detection.get_count_value(), elapsed_time);
    }

    std::cout << std::endl;

    contour_detection.print_result();

    // 计算最大速率
    double max_rate = 0.0;
    for (size_t i = 1; i < count_timestamps.size(); ++i) {
        int count_diff = count_timestamps[i].first - count_timestamps[i - 1].first;
        double time_diff = count_timestamps[i].second - count_timestamps[i - 1].second;
        if (time_diff > 0) {
            double rate = count_diff / time_diff;
            if (rate > max_rate) {
                max_rate = rate;
            }
        }
    }

    std::cout << "最大吞吐计数速率: " << max_rate << " 次/秒" << std::endl;

    cap.release();
}