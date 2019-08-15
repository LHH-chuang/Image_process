import cv2
import os
import numpy as np
import test3

#15厘米剪裁成九个五厘米
#输入为剪裁后的15厘米图片的路径
def fivehuoqu(ppp):
    wenjianming = os.path.basename(ppp)
    wenjianming2 = os.path.splitext(wenjianming)
    if not os.path.exists("cheng5/" + wenjianming2[0]):
        os.mkdir("cheng5/" + wenjianming2[0])
    img = cv2.imread(ppp)
    h, w = img.shape[:2]
    ww = int(w / 3)
    #三分之一宽
    hh = int(h / 3)
    # 三分之一高
    for x in range(3):
        w1 = x * ww
        #剪裁左边的位置
        w2 = (x + 1) * ww
        #剪裁右边的位置
        for y in range(3):
            print(x)
            h1 = y * hh
            #剪裁上面的位置
            h2 = (y + 1) * hh
            #剪裁下面的位置
            img1 = img[w1:w2, h1:h2]
            zz = wenjianming2[0] + "$" + str(3 * x + y + 1) + ".jpg"
            zz = os.path.join("cheng5", wenjianming2[0], zz)
            #print(zz)
            cv2.imwrite(zz, img1)

#15厘米剪裁四个十厘米
#输入为剪裁后的15厘米图片的路径
def tenhuoqu(ppp):
    wenjianming = os.path.basename(ppp)
    wenjianming2 = os.path.splitext(wenjianming)
    if not os.path.exists("10cheng10/" + wenjianming2[0]):
        os.mkdir("10cheng10/" + wenjianming2[0])
    img = cv2.imread(ppp)
    h, w = img.shape[:2]
    ww = int(w / 3)
    #三分之一宽
    hh = int(h / 3)
    # 三分之一高
    for x in range(2):
        w1 = x * ww
        #剪裁左边的位置
        w2 = (x + 2) * ww
        #剪裁右边的位置
        for y in range(2):
            h1 = y * hh
            #剪裁上面的位置
            h2 = (y + 2) * hh
            #剪裁下面的位置
            img1 = img[w1:w2, h1:h2]
            mingzi = wenjianming2[0] + "$" + str(2 * x + y + 1) + ".jpg"
            #对输出的图片进行命名
            lujing = os.path.join("10cheng10", wenjianming2[0], mingzi)
            #合成输出图片的路径
            cv2.imwrite(lujing, img1)

#遍历整个文件夹内的所有图片
#传入为一个文件夹路径
def bianli(wenjianjia):
    for root, dirs, files in os.walk(wenjianjia, topdown=False):
        for name in files:
            tenhuoqu(os.path.join(root, name))

#wenjianjia = "CHP"
#bianli(wenjianjia)