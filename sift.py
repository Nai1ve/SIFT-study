import cv2
import matplotlib.pyplot as plt
import pandas as pd

def unpack_octave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    octave = octave if octave < 128 else -(256 - octave)
    return octave, layer

# 读取图像
image = cv2.imread('Zeeta.jpg')
# 原始图像690*1035

# 对图像进行缩放
# image = cv2.resize(image, (300, 450))
# 缩放后的图像300*450

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测关键点并计算描述符
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

print('Number of keypoints:', len(keypoints))
print(keypoints[2])

pd.DataFrame(descriptors).to_csv('descriptors.csv',index=False)


keypoint = keypoints[2]
octave, layer = unpack_octave(keypoint)
keypoint_properties = {
    'pt': keypoint.pt,  # 关键点坐标 (x, y)
    'size': keypoint.size,  # 关键点大小
    'angle': keypoint.angle,  # 关键点方向
    'response': keypoint.response,  # 反应值
    'octave': octave,  # 金字塔层数
    'layer': layer,  #
    'class_id': keypoint.class_id  # 类别ID
}

# 打印格式化属性
for prop, value in keypoint_properties.items():
    print(f"{prop}: {value}")

print('Descriptor shape:', descriptors.shape)
print(descriptors)

# 在图像上绘制关键点
# output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示结果图像
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# plt.title('SIFT Keypoints')
# plt.axis('off')
# plt.show()

# 保存结果图像
#cv2.imwrite('zeeta_sift_output.jpg', output_image)
#cv2.imwrite('zeeta_sift_output_300_450.jpg', output_image)
