import os
import logging
import random

import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import morphsnakes as ms

if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('TkAgg')

def visual_callback_2d(background, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()   # 清除整个当前图形
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    # ax2 = fig.add_subplot(1, 2, 2)  # matplotlib默认的颜色映射为十色环
    # 修改映射
    # plt.set_cmap('binary')
    # ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1, cmap=plt.cm.gray)
    # ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')  # 在原图上绘制轮廓线
        # ax_u.set_data(levelset)  # 将Axes对象中显示的图形数据设置为levelset所表示的数据 实现更新显示的效果
        fig.canvas.draw() # 重新绘制该Figure对象中的所有Axes对象
        plt.pause(0.001)  # 暂停执行 动态变化

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def test_MorphACWE(imgcolor):
    logging.info('Running: example_camera (MorphACWE)...')

    # Load the image.
    imgcolor = imgcolor/255.0
    imggray = rgb2gray(imgcolor)

    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # Morphological Chan-Vese (or ACWE)
    u = ms.morphological_chan_vese(imggray, 40,
                               smoothing=5, lambda1=1, lambda2=1,
                               iter_callback=callback)
    
    # 判断
    u = u_invert(u, imggray)

    # 在原图上进行绘制
    img_contours = draw_contours(u, imgcolor)

    return u, img_contours

# 判断是否修改映射
def u_invert(u, imggray):
    # u为int8类型 需要转化成cv中的uint8
    u = np.uint8(u)
    
    # 轮廓线contours的位置
    contours, _ = cv2.findContours(u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 随机取像素点
    height, width= imggray.shape
    random_h = np.random.randint(0, height, size=100)
    random_w = np.random.randint(0, width, size=100)
    coordinates = np.column_stack((random_h, random_w))  # (100, 2)
    # coordinates = tuple(coordinates.tolist())

    # 获取轮廓线内部和外部的灰度值
    pixel_inside = []
    pixel_outside = []
    # 灰度值对应的坐标值
    point_inside = []
    point_outside = []

    # 某个情况 点A位于轮廓线M外部 但是位于轮廓线N的内部 这样point_outside 就会有重复
    # 内部 只要is_inside == 1就可以
    # 外部 要所有的is_inside都等于-1才是真正的外部轮廓点
    flag = 0
    for (y, x) in coordinates:
        point = (np.float32(x), np.float32(y))
        for contour in contours:
            is_inside = cv2.pointPolygonTest(contour, point, False)
            flag += is_inside
            if is_inside == 1:   # ==0是在轮廓线上
                pixel_inside.append(imggray[y, x])
                point_inside.append((y, x))
                
        if flag == -len(contours):   # is_inside == -1
            pixel_outside.append(imggray[y, x])
            point_outside.append((y, x))
        
        flag = 0

    mean_pixel_inside = np.mean(pixel_inside)
    mean_pixel_outside = np.mean(pixel_outside)
    # 判断
    y, x = random.choice(point_inside) if mean_pixel_inside > mean_pixel_outside else random.choice(point_outside)
    if u[y, x] != 1:
        u = np.invert(u)/255.0
    
    return u

def draw_contours(u, imgcolor):
    img_u = np.uint8(u)
    contours, _ = cv2.findContours(img_u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.drawContours(imgcolor, contours, -1, (255, 0, 0), 2)

    return img_contours

def coverage_rate(levelset):
    rows, cols = levelset.shape
    # levelset 取值为0和1
    sum_pixel = rows * cols   # 总像素数
    white_pixel = np.sum(levelset == 1)  # 白色像素数 即秸秆

    # 覆盖率
    rate = (white_pixel / sum_pixel) * 100

    return rate

def uniformity_rate(levelset, grid_size):
    rows, cols = levelset.shape
    # 每行和每列的网格数
    num_rows = rows // grid_size
    num_cols = cols // grid_size
    # 划分图像的网格区域
    grid_image = []
    for i in range(num_rows):
        for j in range(num_cols):
            left = j * grid_size
            top = i * grid_size
            right = left + grid_size
            bottom = top + grid_size

            grid_image.append(levelset[top:bottom, left:right])

    # 计算均值
    grid_rate = [coverage_rate(grid_image[i]) for i in range(len(grid_image))]
    grid_rate_sum = sum(grid_rate) 
    grid_rate_mean = grid_rate_sum / len(grid_rate)

    # 均匀度
    tmp = 0
    for i in range(len(grid_rate)):
        tmp += abs(grid_rate[i] - grid_rate_mean)
    
    res = 100 - tmp/grid_rate_sum*100

    return res
