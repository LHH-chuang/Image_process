import cv2
import numpy as np
import copy

def crop_coin_above(image):
    h, w = image.shape[:2]
    reSize = cv2.resize(image, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_CUBIC)#将轮廓缩小一倍
    blurred = cv2.medianBlur(reSize, 19)#中值滤波
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)#高斯滤波
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(gradx, grady, 50, 150)#获取硬币的轮廓特征

    #获取所有的外轮廓
    contours, hierarchy = cv2.findContours(edge_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #如果没有轮廓直接输出原图
    if len(contours) == 0:
        return image, 0

    area = []  # 设置一个存放面积的空列表
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
        # 将每个轮廓的面积依次写入列表

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
        if bili1 > 0.5 and bili2 > 0.5:
            continue
        else:
            del area[int(max_idx)]
    # 遍历剩余的所有轮廓
    # 通过计算轮廓的宽高比例，获得轮廓近似为正方形或圆形的图片
    # 将轮廓为正方形和圆形的保留在轮廓列表中，其余删除
    if len(area) == 0:
        return img, 0

    max_idx = np.argmax(area)
    cnt = contours[max_idx]

    M = cv2.moments(cnt)
    #moments的到轮廓的一些特征 返回为一个字典 M["m00"]表示轮廓面积
    cX = int(2 * M["m10"] / M["m00"])
    cY = int(2 * M["m01"] / M["m00"])
    #获得需要去截取轮廓的重心

    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    radio = int(bottommost[1] - topmost[1])
    #计算需要去截取轮廓的半径

    if cX < w/2 and cY < h/2:
        x = w - cX
        y = h -cY
    elif cX<w/2 and cY>h/2:
        x = w - cX
        y = cY
    elif cX>w/2 and cY<h/2:
        x = cX
        y = h - cY
    else:
        x = cX
        y = cY
    z = min(x, y)
    #取较小的边作为比较进行截取图片

    if  z / radio > 13 :
        time = 13
        jiequgeshi = 15
    elif z / radio > 9 and  z / radio <= 13:
        time = 9
        jiequgeshi = 10
    elif z / radio > 5 and  z / radio <= 9:
        time = 5
        jiequgeshi = 5
    else:
        jiequgeshi = 0
        return  image,jiequgeshi
        #如果均不符合上述条件，则输出原图

    if cX<w/2 and cY<h/2:
        a = int(cX + time * radio)
        b = int(cX + radio)
        c = int(cY + time * radio)
        d = int(cY + radio)
        crop = image[d:c, b:a]
    elif cX<w/2 and cY>h/2:
        a = int(cX + time * radio)
        b = int(cX + radio)
        c = int(cY - time * radio)
        d = int(cY - radio)
        crop = image[c:d, b:a]
    elif cX>w/2 and cY<h/2:
        a = int(cX - time * radio)
        b = int(cX - radio)
        c = int(cY + time * radio)
        d = int(cY + radio)
        crop = image[d:c, a:b]
    else:
        a = int(cX - time * radio)
        b = int(cX - radio)
        c = int(cY - time * radio)
        d = int(cY - radio)
        crop = image[c:d, a:b]
    return crop,jiequgeshi


def crop_coin_italic(image):
    h, w = image.shape[:2]
    reSize = cv2.resize(image, (int(w / 4), int(h / 4)), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.medianBlur(reSize, 11)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(gradx, grady, 50, 150)

    contours, hierarchy = cv2.findContours(edge_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    cnt = contours[max_idx]

    leftmost=tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost=tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost=tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
    #获得轮廓的最上面，最下面，最左边，最右边的点。
    x = 4 * (bottommost[1] - topmost[1])
    r = 4 * (rightmost[0] - leftmost[0])
    # 得到椭圆的长直径,短直径
    if x > r:
        t = r
        r = x
        x = t

    # 获得轮廓的重心
    M = cv2.moments(cnt)
    cX = int(4 * M["m10"] / M["m00"])
    cY = int(4 * M["m01"] / M["m00"])

    # 取较小的边作为比较进行截取图片
    if cX < w/2 and cY < h/2:
        y = h -cY
    elif cX<w/2 and cY>h/2:
        y = cY
    elif cX>w/2 and cY<h/2:
        y = h - cY
    else:
        y = cY
    z = min(w,y)

    if z / r > 6:
        time1 = 6
        time2 = 3
        jiequgeshi = 15
    elif z / r > 4 and z / r <= 6:
        time1 = 4
        time2 = 2
        jiequgeshi = 10
    elif z / r > 2 and z / r <= 4:
        time1 = 2
        time2 = 1
        jiequgeshi = 5
    else:
        jiequgeshi = 0
        return image, jiequgeshi

    if  cY < h / 2 :
        cX = w / 2
        cY = cY + x * 2 / 3
        pts1 = np.float32([[cX - time2 * x, cY], [cX + time2 * x, cY], [cX - time2 * r, cY + time1 * r], [cX + time2 * r, cY + time1 * r]])
        pts2 = np.float32([[0, 0], [time1 * r, 0], [0, time1 * r], [time1 * r, time1 * r]])
        MM = cv2.getPerspectiveTransform(pts1, pts2)
        # cv2.getPerspectiveTransform(src,dst) 计算转换矩阵
        # src输入图像的，dst输出图像
        dst = cv2.warpPerspective(image, MM, (time1 * r, time1 * r))
    else :
        cX = w / 2
        cY = cY -  x / 2
        pts1 = np.float32([[cX - time2 * x, cY - time1 * r], [cX + time2 * x, cY - time1 * r],[cX - time2 * r, cY], [cX + time2 *r, cY]])
        pts2 = np.float32([[0, 0], [time1 * r, 0], [0, time1 * r], [time1 * r, time1 * r]])
        MM = cv2.getPerspectiveTransform(pts1, pts2)
        # cv2.getPerspectiveTransform(src,dst) 计算转换矩阵
        # src输入图像的，dst输出图像
        dst = cv2.warpPerspective(image, MM, (time1 * r, time1 * r))

    #dst = cv2.bitwise_and(image, image, mask=edge_output)
    return dst,jiequgeshi

def crop_coin(image):
    crop = crop_coin_above(image)
    h ,w = image.shape[:2]
    h1 ,w1 = crop[0].shape[:2]
    if h == h1:
        crop = crop_coin_italic(image)
        print('1')
    return crop

#src = cv2.imread("include/coin21.jpg")
#img = crop_coin(src)
#cv2.imwrite("2.jpg",img[0])
