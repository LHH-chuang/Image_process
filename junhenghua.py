import cv2
import numpy as np


def junhenghua(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Imin, Imax = cv2.minMaxLoc(gray)[:2]
    Omin, Omax = 0, 255
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * gray + b
    out = out.astype(np.uint8)
    # gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # out = cv2.equalizeHist(gray2)
    return out


#img = cv2.imread("include/crop.jpg")
#out1 = junhenghua(img)
#cv2.imwrite("output.jpg", out1)