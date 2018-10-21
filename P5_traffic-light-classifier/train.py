
# 1 加载图像
# 加载库

import cv2 
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

# 定义图像路径
# IMAGE_DIR_TRAINING: the directory where our training image data is stored
# IMAGE_DIR_TEST: the directory where our test image data is stored
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# 加载数据集
# 加载训练数据集
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
IMAGE_TEST = helpers.load_dataset(IMAGE_DIR_TEST)


def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
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
    
    # Empty image data array
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        # Standardize the image
        standardized_im = standardize_input(image)
        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    
        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
    return standard_list

# 标准化训练集
STANDARDIZED_LIST = standardize(IMAGE_LIST)
STANDARDIZED_TEST = standardize(IMAGE_TEST)

# 把平均亮度作为特征值
def create_feature(rgb_image):
    # 获取图像和标签
    #image = rgb_image[0]
    #label = rgb_image[1]

    # 裁剪图像，去掉背景的干扰
    image_crop = rgb_image[3:-3, 7:-7]   

    # 转换成hsv色彩空间
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

    # 提取HSV的3个通道
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

    # 提取hsv的值
    h_mask = hsv_mask[:,:,0]
    s_mask = hsv_mask[:,:,1]
    v_mask = hsv_mask[:,:,2]
    up = np.sum(v_mask[0:9])
    mid = np.sum(v_mask[9:18])
    low = np.sum(v_mask[18:])
    return (up, mid, low, h_mask)

# 区分hsv的色调hue
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
def estimate_label(rgb_image, i, j, k):
    up, mid, low, h_mask = create_feature(rgb_image)
    if up > (mid + low) * i:
        label = [1, 0, 0]
    elif mid > (up + low) * j:
        label = [0, 1, 0]
    elif low > (up + mid) * k:
        label = [0, 0, 1]
    #else:
        #label = [1, 0, 0]
    else:
        label = create_feature_hsv(h_mask)
    return label


# 跟踪识别错误的图片的索引
def collect_error_image(STANDARDIZED_LIST):
    error_image_num = float("inf")
    best_i, best_j, best_k = None, None, None
    for each in train():
        i, j, k = each
        error = []
        # 跟踪图像索引
        n = 0
        for each_image in STANDARDIZED_LIST:
            image = each_image[0]
            label = each_image[1]
            predicted_label = estimate_label(image, i, j, k)
            # 如果把红色识别成绿灯，跳过本次循环
            if label == [1, 0, 0] and predicted_label == [0, 0, 1]:
                continue
            # 如果没有把红灯识别成绿灯，对于预测错了的图像，记录该图像的索引值n
            if predicted_label != label:
                error.append((n, label, predicted_label)) 
            n += 1
        if len(error) < error_image_num:
            error_image_num = len(error)
            best_i, best_j, best_k = i, j, k
            print("i:", best_i, "j:", best_j, "k:", best_k, "error image num:", error_image_num)
    print("number of error image:", error_image_num)
    print("{:.2f}%".format((1 - error_image_num/1187)*100))
    return best_i, best_j, best_k, error_image_num



    # 显示识别错误的图片
    #for each in error_red_to_green:
        #plt.imshow(STANDARDIZED_LIST[each][0])
        #plt.show()


# 训练参数i，j，k
def train():
    num = []
    for i in np.arange(0.5, 2, 0.1):
        for j in np.arange(0.5, 2, 0.1):
            for k in np.arange(0.5, 2, 0.1):
                num.append((i, j, k))
    return num
print(train())

print("start")
#print(collect_error_image(STANDARDIZED_LIST))
print(collect_error_image(STANDARDIZED_TEST))

print("end")
