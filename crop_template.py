import cv2
import numpy as np
import math
#import time

def above(img):#img 输入图像
    #gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    #cvtColor（src，type）
    #src原图像，type转换类型
    #cv2.COLOR_BGR2BGRA 将alpha通道增加到BGR或者RGB图像中

    ret,binary = cv2.threshold(img,133,255,cv2.THRESH_BINARY)
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
    # h, w = gray2.shape
    gray2 = cv2.bitwise_not(gray2)
    cv2.imwrite("4.jpg", gray2)
    contours, hierarchy = cv2.findContours(gray2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #获得所有的轮廓
    #print("contours", contours)
    #print("hierarchy", hierarchy)
    #cv2.findContours(image, mode, method)
    #image一个8位单通道二值图像（非0即1） mode轮廓的检索模式:cv2.RETR_EXTERNAL表示只检测外轮廓  method为轮廓的近似办法: cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
    #contours返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
    #hierarchy返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。

    if len(contours) == 0:
        return img,0
    #如果没有发现轮廓，将返回原图，并输出0

    area = []#设置一个存放面积的空列表
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
        #将每个轮廓的面积依次写入列表
    #print(area)

    #max_idx = np.argmax(area)
    #del area[int(max_idx)]
    #获得面积最大的那个轮廓，然后删除（目的是删除白色的那个轮廓）

    for x in range(len(contours)):
        max_idx = np.argmax(area)
        cnt = contours[max_idx]
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        h = bottommost[1] - topmost[1]
        w = rightmost[0] - leftmost[0]
        bili1 = h / w
        bili2 = w / h
        if  bili1 > 0.5 and  bili2 > 0.5:
            continue
        else:
            del area[int(max_idx)]
    #遍历剩余的所有轮廓
    #通过计算轮廓的宽高比例，获得轮廓近似为正方形或圆形的图片
    #将轮廓为正方形和圆形的保留在轮廓列表中，其余删除
    if len(area) == 0:
        return img,0

    max_idx = np.argmax(area)
    cnt = contours[max_idx]
    M = cv2.moments(cnt)
    # moments的到轮廓的一些特征 返回为一个字典 M["m00"]表示轮廓面积
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #由于裁剪区域（正方形）必定比比对区域（圆形）大，所以在此获得正方形的轮廓
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print(approx)
    list =[]
    list.append(approx[0][0])
    list.append(approx[1][0])
    list.append(approx[2][0])
    list.append(approx[3][0])
    for x in range(len(list)):
        if  list[x][0] - cX <= 0 and list[x][1] - cY >= 0 :
            point0 = list[x]
        elif list[x][0] - cX <= 0 and list[x][1] - cY <= 0 :
            point1 = list[x]
        elif list[x][0] - cX >= 0 and list[x][1] - cY <= 0 :
            point2 = list[x]
        else :
            point3 = list[x]

    R = int(point3[1] - point1[1])
    #获得正方形的宽
    pts1 = np.float32([point0, point1, point3,point2])
    pts2 = np.float32([[0, 0], [R, 0], [0, R], [R, R]])
    MM = cv2.getPerspectiveTransform(pts1, pts2)
    # cv2.getPerspectiveTransform(src,dst) 计算转换矩阵
    # src输入图像的，dst输出图像
    crop = cv2.warpPerspective(img, MM, (R, R))
    h ,w =crop.shape[:2]
    a = int(w * 0.05)
    b = int(w * 0.95)
    c = int(h * 0.05)
    d = int(h * 0.95)
    crop = crop[a:b,c:d]
    # 剪裁图像的区域

    print(area)
    max_idx1 = np.argmax(area)
    del area[int(max_idx1)]

    if len(area) == 0:
        return crop,0

    max_idx2 = np.argmax(area)
    cont = contours[max_idx2]
    #由于裁剪区域必定比比对区域小，所以在此获得圆形的轮廓
    left= tuple(cont[cont[:, :, 0].argmin()][0])
    right= tuple(cont[cont[:, :, 0].argmax()][0])
    X = right[0] - left[0]
    # 获得圆形最上、最下的点
    # 获得圆形的直径

    #print(bottom,top)
    print(R,X)
    if   R / X < 3 :
        chicun = 5
        print("截出长为5厘米正方形")
    elif R / X >= 3 and R / X < 5 :
        chicun = 10
        print("截出长为10厘米正方形")
    elif R / X >= 5 and R / X < 7 :
        chicun = 15
        print("截出长为15厘米正方形")
    elif R / X >= 7 and R / X < 9:
        chicun = 20
        print("截出长为20厘米正方形")
    else:
        chicun = 0
    return  crop,chicun

#img = cv2.imread("hz/10-4.jpg")
#crop = above(img)
#cv2.imwrite("bt.jpg", crop[0])