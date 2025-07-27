import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 关闭所有图窗
plt.close("all")

# 加载图像
image_path = "images/frame_1634.png"  # 替换为你的图像路径
image = cv.imread(image_path)

# 提取红色通道
red_channel = image[:, :, 2]

# # 绘制红色通道的条形统计图
# plt.figure()
# plt.title("Histogram of Red Channel")
# plt.hist(red_channel.ravel(), bins=256, range=[0, 256])
# plt.show()

# 设置阈值和内核大小
threshold = 226  # 根据需要调整阈值
kernel_size = (5, 5)  # 根据需要调整内核大小

ret, src_bin = cv.threshold(red_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 阈值化
# _, src_bin = cv.threshold(red_channel, threshold, 255, cv.THRESH_BINARY)
plt.figure()
plt.title("Binary Image")
plt.imshow(src_bin, cmap="gray")
plt.show()

# 输出阈值
print("Calculated Otsu's threshold:", ret)

# # 形态学操作
# kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
# src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
# plt.figure()
# plt.title("After Morphological Opening")
# plt.imshow(src_bin, cmap="gray")
# plt.show()

# src_bin = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
# plt.figure()
# plt.title("After Morphological Closing")
# plt.imshow(src_bin, cmap="gray")
# plt.show()

# # 找到非零像素点
# coords = cv.findNonZero(src_bin)
# plt.figure()
# plt.title("Non-zero Coordinates")
# plt.imshow(src_bin, cmap="gray")
# plt.scatter(coords[:, 0, 0], coords[:, 0, 1], s=1, c="red")
# plt.show()

# 找到轮廓
contours, _ = cv.findContours(src_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# if not contours:
#     bbox = cv.boundingRect(coords)
#     mask = None
# else:
# 找到最大轮廓
contour = max(contours, key=cv.contourArea)
mask = np.zeros(red_channel.shape, dtype=np.uint8)
cv.drawContours(mask, [contour], -1, 1, thickness=cv.FILLED)
plt.figure()
plt.title("Mask with Contour")
plt.imshow(mask, cmap="gray")
plt.show()
bbox = cv.boundingRect(contour)

# 绘制边界框
x, y, w, h = bbox
cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Segmented Image with Bounding Box")
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")

plt.show()
