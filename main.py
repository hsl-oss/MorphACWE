from utils import Imgseg
from PIL import Image
import numpy as np

img_path = 'images/img2.jpeg'
grid_size = 80

img = Image.open(img_path)
img = img.resize((320,240))
img = np.array(img)   # (240, 320, 3)   高 宽 通道

seg = Imgseg(img_path=img, grid_size=grid_size)
u, c_rate, u_rate = seg.seg_rate()

print(c_rate, u_rate)