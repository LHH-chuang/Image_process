import cv2
import  numpy as np
import math


def secha(img1,img2):
    img1 =img1[200:1200, 200:1200]
    img_lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img_ls1 = np.average(img_lab1[..., 0])
    img_as1 = np.average(img_lab1[..., 1])
    img_bs1 = np.average(img_lab1[..., 2])
    img_cs1 = (img_as1 ** 2 + img_bs1 ** 2) ** 0.5

    img2 = img2[200:1200, 200:1200]
    img_lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    img_ls2 = np.average(img_lab2[..., 0])
    img_as2 = np.average(img_lab2[..., 1])
    img_bs2 = np.average(img_lab2[..., 2])
    img_cs2 = (img_as2 ** 2 + img_bs2 ** 2) ** 0.5

    cm = (img_cs1 + img_cs2) / 2
    G = 0.5 * (1 - ((cm ** 7) / (cm ** 7 + 25 ** 7)) ** 0.5 )
    a1 = (1 + G) * img_as1
    a2 = (1 + G) * img_as2
    C1 = (img_as1 ** 2 + img_bs1 ** 2) ** 0.5
    h1 = (180 / math.pi) * math.asin(int(img_bs1 / a1))
    C2 = (img_as2 ** 2 + img_bs2 ** 2) ** 0.5
    h2 = (180 / math.pi) * math.asin(int(img_bs2 / a2))
    Cm = (C1 + C2) / 2

    hm = (h1 + h2) / 2
    if math.fabs(h1 - h2) > 360:
        hm = hm -360
    elif hm < 0:
        hm = hm + 360

    Dh = h2 - h1
    if Dh > 180 :
        Dh = Dh - 2 * 180
    elif Dh < -180:
        Dh = Dh + 2 * 180

    rad = math.pi / 180
    DL = img_ls2 - img_ls1
    DC = C2 - C1
    DH = 2 * ((C1 * C2) ** 0.5) * math.sin(rad * Dh / 2)

    T = 1 - 0.17 * math.cos(rad * (hm - 30)) + 0.24 * math.cos(rad * 2 * hm) + 0.32 * math.cos(rad * (3 * hm + 6)) - 0.2 * math.cos(rad * (4 * hm - 63))
    SL = 1 + (0.015 * ((img_ls1 + img_ls2) / 2 - 50) ** 2) /((20 + ((img_ls1 + img_ls2) / 2 - 50) ** 2) ** 0.5)
    SC = 1 + 0.045 * Cm
    SH = 1 + 0.015 * Cm * T

    kL = 1
    kC = 1
    kH = 1
    Dt = 30 * math.exp( -(((hm - 275) / 25) ** 2))
    RC = 2 * ((Cm ** 7) / (Cm ** 7 + 25 ** 7)) ** 5
    RT = - math.sin(2 * rad * Dt) * RC
    Rot = RT * (DC / (SC * kC)) * (DH / (SH * kH))
    DE = ((DL / (SL * kL)) ** 2 + (DC / (SC * kC)) ** 2 + (DH / (SH * kH)) ** 2 + Rot) ** 0.5

    return DE

img1 = cv2.imread("1.bmp")
img2 = cv2.imread("3.bmp")
a = secha(img1,img2)
print(a)