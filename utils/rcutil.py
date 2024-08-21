import cv2
import time
import pandas as pd
from pynput import keyboard
import numpy as np
from matplotlib import pyplot as plt
import robomaster
from robomaster import robot
from robomaster import camera

# 平移对齐图片
def align_images(img1, img2):
    # 将图片转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用 SIFT 特征检测器检测关键点和描述符
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用 FLANN 匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 选择最佳匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 使用 RANSAC 算法估计单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 对图像进行透视变换
    h, w = img1.shape[:2]
    img1_aligned = cv2.warpPerspective(img1, M, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return img1_aligned, img2

def calculate_rotation_angle(img1, img2):
    # 读取图片
    # img1 = cv2.imread("train/images/1.jpg")
    # img2 = cv2.imread("train/images/2.jpg")

    # 将图片转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用 SIFT 特征检测器检测关键点和描述符
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用 FLANN 匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 选择最佳匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 使用 RANSAC 算法估计单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 获取旋转角度
    angle = np.arctan(M[0, 1] / M[0, 0]) * 180 / np.pi
    return angle
    
    
def find_index(lst, target):
    # 使用 lambda 函数计算每个元素与目标数的绝对差值，并返回最小差值的索引
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))
w_pressed = o_pressed = n_pressed = False
def on_press(key):
    try:
        if key.char == 'w':
            global w_pressed
            w_pressed = True
        elif key.char == 'o':
            global o_pressed
            o_pressed = True
            return False
        elif key.char == 'n':
            global n_pressed
            n_pressed = True
    except AttributeError:
        pass
def on_release(key):
    try:
        if key.char == 'w':
            global w_pressed
            w_pressed = False
        elif key.char == 'o':
            global o_pressed
            o_pressed = False
            return False
        elif key.char == 'n':
            global n_pressed
            n_pressed = False
    except AttributeError:
        pass
if __name__ == "__main__":
    datas = pd.read_csv('train/dataset.csv')
    # update the angle with images in "train" folder and save it in "train/dataset.csv"
    for i in range(1, len(datas)):
        img1 = cv2.imread(datas['image_path'][i-1])
        img2 = cv2.imread(datas['image_path'][i])
        try:
            #calculate it ten times and take the average
            angle = calculate_rotation_angle(img1, img2)
        except:
            angle = 1
        datas['angle'][i] = angle
    datas.to_csv('train/dataset.csv', index=False)