import cv2
import numpy as np
import matplotlib.pyplot as plt

#直方图正规化
# def zftzgh(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     Imin, Imax = cv2.minMaxLoc(gray)[:2]
#     Omin, Omax = 0, 255
#     #将I中出现的最小灰度级记为I_{min}，最大灰度级记为I_{max}
#     #为使输出图像O的灰度级范围为 [O_{min},O_{max}]
#     a = float(Omax - Omin) / (Imax - Imin)
#     b = Omin - a * Imin
#     out = a * gray + b
#     out = out.astype(np.uint8)
#     # gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     # out = cv2.equalizeHist(gray2)
#     return out

#正规化函数
# def zhengguihua(img):
#     out = np.zeros(img.shape, np.uint8)
#     cv2.normalize(img, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
#     return out

#灰度直方图均衡化
def zftjhh_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #将图片转化为灰度图
    dst = cv2.equalizeHist(gray)
    # 使用全局直方图均衡化
    return dst

#对图像的RGB通道分别进行直方图均衡化
# def zftjhh_rgb(img):
#     (b, g, r) = cv2.split(img)
#     # 将RGB三通道分开
#     bH = cv2.equalizeHist(b)
#     gH = cv2.equalizeHist(g)
#     rH = cv2.equalizeHist(r)
#     # 合并每一个通道
#     result = cv2.merge((bH, gH, rH))
#     return result
#


#img = cv2.imread("crop.jpg")
# out1 = zftzgh(img)
#out2 = zhengguihua(img)
# cv2.imwrite("output1.jpg", out1)
#cv2.imwrite("output.jpg", out2)
