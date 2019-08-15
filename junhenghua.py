import cv2
import numpy as np
import matplotlib.pyplot as plt

# def zftzgh(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     Imin, Imax = cv2.minMaxLoc(gray)[:2]
#     Omin, Omax = 0, 255
#     a = float(Omax - Omin) / (Imax - Imin)
#     b = Omin - a * Imin
#     out = a * gray + b
#     out = out.astype(np.uint8)
#     # gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     # out = cv2.equalizeHist(gray2)
#     return out

def zftjhh_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    return dst

# def zftjhh_rgb(img):
#     (b, g, r) = cv2.split(img)
#     bH = cv2.equalizeHist(b)
#     gH = cv2.equalizeHist(g)
#     rH = cv2.equalizeHist(r)
#     # 合并每一个通道
#     result = cv2.merge((bH, gH, rH))
#     return result
#
# def zftjhh_lab(img):
#     img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     # print(img1[:,:, 0])
#     plt.figure()
#     img_l = img_lab[:, :, 0]
#     img_l = cv2.equalizeHist(img_l)
#     return img_l

#img = cv2.imread("crop.jpg")
# out1 = zftzgh(img)
#out2 = zftjhh_gray(img)
# cv2.imwrite("output1.jpg", out1)
#cv2.imwrite("output.jpg", out2)
