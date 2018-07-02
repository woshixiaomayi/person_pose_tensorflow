'''
功能：在图片上画出线条（测试代码）
作者：老艾
时间：2018年06月22日16:31:30
'''

import scipy.io
import cv2
import numpy as np


# 载入所有图片路径
img_path = scipy.io.loadmat("../dataset/mydata/matlab.mat")

img_list = []
for i in range(len(img_path['dataset'][0])):
    img_list.append(img_path['dataset'][0][i][0][0])

# 载入每一张图片的关键点
joints = scipy.io.loadmat("../models/coco/test/predictions.mat")

joints_list = []
for i in range(len(joints['joints'][0])):
    joints_list.append(joints['joints'][0][i])

# 循环所有图片，为每一张图片划线，同时保存。
k = 0
for i, j in zip(img_list, joints_list):
    img_path = i
    joints = j

    img = cv2.imread(img_path)


    cv2.line(img, (int(joints[0][0]),int(joints[0][1])), (int(joints[1][0]),int(joints[1][1])), 255, 5)
    cv2.line(img, (int(joints[1][0]),int(joints[1][1])), (int(joints[2][0]),int(joints[2][1])), 255, 5)
    cv2.line(img, (int(joints[2][0]),int(joints[2][1])), (int(joints[3][0]),int(joints[3][1])), 255, 5)
    cv2.line(img, (int(joints[2][0]),int(joints[2][1])), (int(joints[4][0]),int(joints[4][1])), 255, 5)
    cv2.line(img, (int(joints[3][0]),int(joints[3][1])), (int(joints[5][0]),int(joints[5][1])), 255, 5)
    cv2.line(img, (int(joints[4][0]),int(joints[4][1])), (int(joints[6][0]),int(joints[6][1])), 255, 5)

    cv2.circle(img, (int(joints[0][0]),int(joints[0][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)
    cv2.circle(img, (int(joints[1][0]),int(joints[1][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)
    cv2.circle(img, (int(joints[2][0]),int(joints[2][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)
    cv2.circle(img, (int(joints[3][0]),int(joints[3][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)
    cv2.circle(img, (int(joints[4][0]),int(joints[4][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)
    cv2.circle(img, (int(joints[5][0]),int(joints[5][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)
    cv2.circle(img, (int(joints[6][0]),int(joints[6][1])), 7, (np.random.randint(256),np.random.randint(256),np.random.randint(256)), -1)

    # 保存图片
    cv2.imwrite( "./img_line/"+str(k)+".jpg", img)

    # 暂时图片
    cv2.imshow('image',img)
    cv2.waitKey(0)
    k += 1
