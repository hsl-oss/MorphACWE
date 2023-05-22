import numpy as np
from imageio import imread

from matplotlib import pyplot as plt
import cv2
import morphsnakes as ms


def visual_callback_2d(background, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()   # 清除整个当前图形
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)  # matplotlib默认的颜色映射为十色环
    # 修改映射
    plt.set_cmap('binary_r')
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')  # 在原图上绘制轮廓线
        ax_u.set_data(levelset)  # 将Axes对象中显示的图形数据设置为levelset所表示的数据 实现更新显示的效果
        fig.canvas.draw() # 重新绘制该Figure对象中的所有Axes对象
        plt.pause(0.001)  # 暂停执行 动态变化

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def test_MorphGAC(img):

    # Load the image.
    imgcolor = img/255.0
    # imgcolor = cv2.medianBlur(imgcolor, 3)
    img = rgb2gray(imgcolor)
    # g(I)
    gimg = ms.inverse_gaussian_gradient(img, alpha=300, sigma=7)

    # Manual initialization of the level set
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[3:-3, 3:-3] = 1

    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # MorphGAC.
    ms.morphological_geodesic_active_contour(gimg, 200, init_ls,
                                             smoothing=1, threshold=0.68,
                                             balloon=-1, iter_callback=callback)

def test_MorphACWE(img):
    # Load the image.
    imgcolor = img/255.0
    img = rgb2gray(imgcolor)

    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # Morphological Chan-Vese (or ACWE)
    u = ms.morphological_chan_vese(img, 40,
                               smoothing=3, lambda1=1, lambda2=1,
                               iter_callback=callback)  
    return u

def coverage_rate(levelset):
    rows, cols = levelset.shape
    # levelset 取值为0和1
    sum_pixel = rows * cols   # 总像素数
    white_pixel = np.sum(levelset == 1)  # 白色像素数 即秸秆

    # 覆盖率
    rate = (white_pixel / sum_pixel) * 100

    return rate

def uniformity_rate(levelset, grid_size):
    rows, cols = levelset.shape   # (240, 320)  高  宽
    # 每行和每列的网格数
    num_rows = rows // grid_size  # 3
    num_cols = cols // grid_size  # 4
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


