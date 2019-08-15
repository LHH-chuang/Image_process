import cv2
import numpy as np
import os

def above(img):#img 输入图像
    wenjianming = os.path.basename(img)
    wenjianming2 = os.path.splitext(wenjianming)
    img1 = cv2.imread(img)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2BGRA)
    #cvtColor（src，type）
    #src原图像，type转换类型
    #cv2.COLOR_BGR2BGRA 将alpha通道增加到BGR或者RGB图像中
    ret,binary = cv2.threshold(gray1,70,255,cv2.THRESH_BINARY)
    #threshold(src,thresh,maxval,type)
    #src原图像，thresh阈值，maxval输出图像的最大值，type阈值类型
    #THRESH_BINARY---二值阈值化
    kernel = np.ones((50, 50), np.uint8)
    #设置方框大小及类型
    dst = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #cv2.morphologyEx(src, type, kernel)
    # src 原图像 type 运算类型 kernel 结构元素
    #cv2.MORPH_OPEN 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作
    #开运算(open)：先腐蚀后膨胀的过程。
    kernel = np.ones((50, 50), np.uint8)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    # cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
    #闭运算(close)：先膨胀后腐蚀的过程。
    #cv2.imwrite("dst.jpg",dst)
    gray2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    # BGR和灰度图的转换使用 cv2.COLOR_BGR2GRAY
    h, w = gray2.shape
    #cv2.imwrite("0.jpg", gray2)
    for x in range(h):
        for y in range(w):
            pv = gray2[x,y]
            gray2[x,y] = 255 - pv
    #cv2.imwrite("2.jpg", gray2)
    contours, hierarchy = cv2.findContours(gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.findContours(image, mode, method)
    #image一个8位单通道二值图像（非0即1） mode轮廓的检索模式:cv2.RETR_EXTERNAL表示只检测外轮廓  method为轮廓的近似办法: cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
    #contours返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
    #hierarchy返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    cnt = contours[max_idx]
    #将contours列表中的元素面积进行大小排序
    #选择contours列表中面积最大的元素
    M = cv2.moments(cnt)
    #moments的到轮廓的一些特征 返回为一个字典 M["m00"]表示轮廓面积
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(cX,cY)
    #获得轮廓的重心
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    #计算轮廓的半径
    R = (bottommost[1] - topmost[1])/2
    a = int(cX + R * 0.95)
    b = int(cX - R * 0.95)
    c = int(cY + R * 0.95)
    d = int(cY - R * 0.95)
    print(R,a,b,c,d)
    crop = img1[d:c, b:a]
        # 剪裁图像的区域
    #crop = cv2.normalize(crop, None, 0,45, cv2.NORM_MINMAX, cv2.CV_8U)
    mingzi = wenjianming2[0] + ".jpg"
    # 对输出的图片进行命名
    lujing = os.path.join("ceshichp", mingzi)
    cv2.imwrite(lujing,crop)

def bianli(wenjianjia):
    for root, dirs, files in os.walk(wenjianjia, topdown=False):
        for name in files:
            above(os.path.join(root, name))

wenjianjia = "cheshi"
bianli(wenjianjia)