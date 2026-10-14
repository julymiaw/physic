# 迈克尔逊干涉条纹净计数

用 C++17 跟踪光屏，再读取少量局部亮度信号，估计条纹相位的推进和回退。不依赖逐帧识别完整圆环。最终目标是低计算量、能适应不同拍摄场景的净计数；当前交付为 Native 研究版本。

## 当前主版本

`adapt/bilibili`的改进已合并到 `main`：只在中心标定前增加时间平均背景扣除，B站标定后6个临时视觉参考事件全部匹配，同时自录88个确定事件验收保持通过。见 [最小修改与双视频结果](experiments/results/bilibili-adapt/README.md) 和 [鲁棒性开发约束](docs/ROBUSTNESS.md)。这两段已是开发数据，陌生场景仍未验证；标签 `self-recorded-v1`保持合并前的原基线，适配分支及其提交历史保留。

## 已冻结的自录基线

固定9.6–11.6秒标定，逐帧测量11.632622–59.831322秒，共1158帧。

- 88个确定事件与视觉复核开发参考一对一匹配，漏计、多计、方向错误均为0。
- 34秒局部小亮点存在物理解释歧义；用户接受散斑或短暂往返两种解释。算法实际另输出一对减一、加一，区间结束净值不变；不把它们宣称为确定真值。
- 片尾净84圈，最大确认延迟4帧，无跳过、重置或隐藏失效。四种真实截断回放检查通过。
- Apple M4、Release、单线程，跟踪＋采样＋计数的单次计时P95为0.608毫秒；不包含解码、预处理和初始化。

这证明当前自录开发参考下的结果，**不证明陌生视频、手机实时能力或大角度倾斜已经可用**。完整结果、参考修订依据和复现记录见 [自录验证说明](experiments/results/physical-count/README.md)。

[观看算法可视化](experiments/results/physical-count/physical-count-explained.mp4)

## 构建与复现

Native核心仅需C++17编译器和CMake，无Eigen、OpenCV或其他C++库依赖。

```sh
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native
```

重现真实视频结果还需Python 3.10+、FFmpeg及原始录像。大视频和派生缓存没有提交到Git；原片放置位置及SHA-256见 [数据说明](data/raw/README.md)。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python research/prepare_original.py
bash research/replay_physical.sh
```

准备工具核对原片并生成带真实时间戳的96像素图，验证派生输入与冻结基线的SHA-256相同；不同解码环境若有差异会明确停止，不把另一份输入的结果冒充冻结基线。每次回放输出到Git忽略的 `experiments/runs/self-recorded/`，不覆盖已提交的基线证据。回放入口检查完整覆盖、逐事件方向与时间、检查点、片尾候选及最多4帧确认，不默认重复计时。

## 当前结构与开发规则

- `research/real_pair.cpp`：唯一当前计数实现，标定和测量共享坐标映射。
- `research/screen_tracker.h`、`screen_track.cpp`：光屏轮廓跟踪及独立诊断入口。
- `research/check_real.py`：固定自录参考的逐事件检查；参考不输入算法。
- `research/prepare_gray.cpp`、`prepare_original.py`：真实录像准备与面积积分降采样。
- `research/render_physical_video.py`：可视化保存的Native结果，未实现另一份计数器。
- `experiments/results/physical-count/`：参考、输出和验证证据；历史快照不参与编译。

`main`保存当前通过双视频开发检查的实现，`self-recorded-v1`保留最初自录基线。后续新场景适配在单独分支进行，每次有意义的修改必须同时回放新视频和自录视频，保留失败结果。现有两段视频一旦用于修改算法，就都属于开发数据；陌生场景的泛化能力需新增未参与调参的视频检验，不能由双视频通过推断。

本科旧文件已经从当前版本移除，Git历史和本地归档仍保留。本轮不把早期失败的正式接口、WASM、合成成绩或小程序界面作为当前算法交付。
