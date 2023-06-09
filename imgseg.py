from matplotlib import pyplot as plt
from utils import test_MorphACWE, coverage_rate, uniformity_rate

class Imgseg():
    def __init__(self, img_path, grid_size):
        self.img_path = img_path
        self.grid_size = grid_size   # 网格大小 与原始图像有关 

    def seg_rate(self):
        # 分割结果 u为黑白图像 img_contours为原图绘制的轮廓线
        u, img_contours = test_MorphACWE(self.img_path)
        # 覆盖率
        c_rate = coverage_rate(u)
        # 均匀度
        u_rate = uniformity_rate(u, self.grid_size)

        return u, img_contours, c_rate, u_rate 


