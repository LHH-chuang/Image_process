import cv2
import  numpy as np
import time

def secha(img1,img2):
    img1 =img1[200:1200, 200:1200]
    img_lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    #print(img1[:,:, 0])
    img_ls1 = np.average(img_lab1[:,:, 0])
    img_as1 = np.average(img_lab1[:,:, 1])
    img_bs1 = np.average(img_lab1[:,:, 2])
    #img_E1 = (img_ls1 ** 2 + img_as1 ** 2 + img_bs1 ** 2) ** (1 / 2)

    img2 = img2[200:1200, 200:1200]
    img_lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    img_ls2 = np.average(img_lab2[:,:, 0])
    img_as2 = np.average(img_lab2[:,:, 1])
    img_bs2 = np.average(img_lab2[:,:, 2])
    #img_E2 = (img_ls2 ** 2 + img_as2 ** 2 + img_bs2 ** 2) ** (1 / 2)

    L = img_ls2 - img_ls1
    A = img_as2 - img_as1
    B = img_bs2 - img_bs1
    E = (L**2 + A**2 + B**2)**(1/2)

    return  E

#img1 = cv2.imread("1.jpg")
#img2 = cv2.imread("2.jpg")
#a = secha(img1,img2)
#print(a)