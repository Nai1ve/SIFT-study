import cv2
import numpy as np

def gaussian_blur(image, times=2):
    for _ in range(times):
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def main():
    # 读取原始图像
    input_image = cv2.imread('zeeta.JPEG')  # 替换为你的图像路径

    # 确保输入图像不为空
    if input_image is None:
        print("无法读取图像，请检查路径。")
        return

    # 创建一个列表来存储输出图像
    output_images = []

    # 对每一层处理
    for i in range(5):  # 总共生成五张图像
        if i == 0:
            # 将图像升采样到2的幂次，例如512x512
            enlarged_image = cv2.resize(input_image, (512, 512))
        else:
            # 对前一张未模糊图像行列缩小2倍
            enlarged_image = cv2.resize(enlarged_image, (enlarged_image.shape[1] // 2, enlarged_image.shape[0] // 2))

        # 保存未模糊的图像
        original_image = enlarged_image.copy()

        # 应用高斯模糊
        blurred_image_1 = gaussian_blur(enlarged_image, times=1)  # 第一次模糊
        blurred_image_2 = gaussian_blur(blurred_image_1, times=1)  # 第二次模糊

        # 纵向拼接未模糊的图像和模糊结果
        combined_image = np.vstack((original_image, blurred_image_1, blurred_image_2))
        output_images.append(combined_image)

    # 显示和保存结果图像
    for i, image in enumerate(output_images):
        cv2.imshow(f'Image {i + 1}', image)
        cv2.imwrite(f'output_image_{i + 1}.jpg', image)  # 保存图像为文件

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
