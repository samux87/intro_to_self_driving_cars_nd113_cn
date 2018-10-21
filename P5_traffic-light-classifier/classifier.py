
# 1 加载图像
# 加载库

import cv2 
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

# 定义图像路径
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# 加载数据集
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# 标准化图像的尺寸
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    return standard_im

# 创建标签转换函数
# 红灯：[1, 0, 0]
# 绿灯：[0, 1, 0]
# 黄灯：[0, 0, 1]
def one_hot_encode(label):
    one_hot_encode = [] 
    if label == "red":
        one_hot_encode = [1, 0, 0]
    elif label == "yellow":
        one_hot_encode = [0, 1, 0]
    else:
        one_hot_encode = [0, 0, 1]
    return one_hot_encode

# 标准化输出。把图像原来的字符串标签，
# 修改成数字标签。
def standardize(image_list):
    standard_list = []
    for item in image_list:
        image = item[0]
        label = item[1]
        standardized_im = standardize_input(image)
        one_hot_label = one_hot_encode(label)    
        standard_list.append((standardized_im, one_hot_label))
    return standard_list

# 标准化数据集
STANDARDIZED_LIST = standardize(IMAGE_LIST)
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# 特征1：亮度
def create_feature(rgb_image):
    # 裁剪图像，去掉背景的干扰
    image_crop = rgb_image[3:-3, 7:-7]   
    # 转换成hsv色彩空间
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
    # 提取hsv的3个通道的值
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    # 创建亮度遮罩，凸显亮着的灯
    lower = np.array(170)
    upper = np.array(255)
    mask = cv2.inRange(v, lower, upper)
    # 屏蔽亮度低的像素
    hsv_mask = np.copy(hsv)
    hsv_mask[mask != 255] = [0, 0, 0]
    # 提取屏蔽处理后的图像的hsv的值
    h_mask = hsv_mask[:,:,0]
    s_mask = hsv_mask[:,:,1]
    v_mask = hsv_mask[:,:,2]
    # 把图像分成up、mid、low3部分
    # 并获得各部分的亮度总和
    up = np.sum(v_mask[0:9])
    mid = np.sum(v_mask[9:18])
    low = np.sum(v_mask[18:])
    return (up, mid, low, h_mask)

# 特征2：平均色调hue
def create_feature_hsv(h_mask):
    # 提取色调特征
    x = h_mask.shape[0]
    y = h_mask.shape[1]
    area = x * y
    avg_h = (np.sum(h_mask)/area)
    if 0 < avg_h < 1:
        feature = [1, 0, 0]
    elif 1 <= avg_h < 150:
        feature = [0, 1, 0]
    else:
        feature = [0, 0, 1]
    return feature


# 分类器
def estimate_label(rgb_image):
    up, mid, low, h_mask = create_feature(rgb_image)
    i = 1.8
    j = 0.7
    k = 0.5
    if up > (mid + low) * i:
        predicted_label = [1, 0, 0]
    elif mid > (up + low) * j:
        predicted_label = [0, 1, 0]
    elif low > (up + mid) * k:
        predicted_label = [0, 0, 1]
    else:
        predicted_label = [1, 0, 0]
    # 经过测试，特征2的准确率较低，所以放弃。
    #else:
        #predicted_label = create_feature_hsv(h_mask)
    return predicted_label



# 测试函数，跟踪识别错误的图片，预测标签，正确标签。
def get_misclassified_images(test_images):
    misclassified_images_labels = []
    for image in test_images:
        im = image[0]
        true_label = image[1]
        # 检测是否为独热编码（one-hot encode)
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # 比较预测便签和正确标签，判断是否预测正确
        # 跟踪不正确的图像及其两个标签
        if(predicted_label != true_label):
            misclassified_images_labels.append((im, predicted_label, true_label))
    return misclassified_images_labels


# 获得所有预测错误的标签
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)
# MISCLASSIFIED = get_misclassified_images(STANDARDIZED_LIST)

# 数据集中数据的个数
total = len(STANDARDIZED_TEST_LIST)
# 正确率
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


# 将识别错误的图像可视化
def display_error_image(error_image):
    for each in MISCLASSIFIED:
        plt.imshow(each[0])
        plt.show()


# display_error_image(MISCLASSIFIED)  


# 显示把红灯显示成绿灯的图片
def display_error_image_red_to_green(error_image):
    red_to_green_num = []
    for each in MISCLASSIFIED:
        predicted_label = each[1]
        true_label = each[2]
        if predicted_label == [0, 0, 1] and true_label == [1, 0, 0]:
            #plt.imshow(each[0])
            #plt.show()
            red_to_green_num.append(each[0])
    return len(red_to_green_num)


# display_error_image_red_to_green(MISCLASSIFIED)    
print("Number of red_images to green = ", display_error_image_red_to_green(MISCLASSIFIED))



# 对于把红灯显示成绿灯的图片，进行裁剪、转换成hsv色彩空间、遮罩处理
def display_error_image_red_to_green_modified(error_image):
    for each in error_image:
        # 裁剪图像，去掉背景的干扰
        image_crop = each[0][3:-3, 7:-7]   
        # 转换成hsv色彩空间
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
        # 提取hsv的3个通道的值
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        # 创建亮度遮罩，凸显亮着的灯
        lower = np.array(200)
        upper = np.array(255)
        mask = cv2.inRange(v, lower, upper)
        # 屏蔽亮度低的像素
        hsv_mask = np.copy(hsv)
        hsv_mask[mask != 255] = [0, 0, 0]
        # 提取屏蔽处理后的图像的hsv的值
        h_mask = hsv_mask[:,:,0]
        s_mask = hsv_mask[:,:,1]
        v_mask = hsv_mask[:,:,2]
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(10, 5))
        ax1.imshow(h_mask, cmap="gray")
        ax2.imshow(s_mask, cmap="gray")
        ax3.imshow(v_mask, cmap="gray")
        ax4.imshow(hsv, cmap="gray")
        ax5.imshow(image_crop)
        plt.show()
    # return (up, mid, low, h_mask)

# display_error_image_red_to_green_modified(MISCLASSIFIED)   




