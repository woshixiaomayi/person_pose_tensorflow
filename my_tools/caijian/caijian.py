'''
功能：实现图片的裁剪功能
作者：老艾
联系：aixinyanchn@163.com
时间：2018年06月23日15:05:31
'''
from tqdm import tqdm
import numpy as np
import os
from xml2dict import XML2Dict
import cv2

fileList = os.listdir("./biaoji")

'''
    imag_path_list  一维
    keypoints_list 三维
'''
img_path_list = []
keypoints_list = []

print( fileList )
for j in range(len(fileList)):
    xml = XML2Dict()
    content_dict = xml.parse("./biaoji/" + fileList[j])["annotation"]

    # 拿到单个图像的路径
    img_path = "./biaozhu/" + content_dict.filename

    # 获取整张图片的宽高
    height = int(content_dict.size.height)
    width = int(content_dict.size.width)

    # 拿到 第二个人 的坐标
    person_bndbox = content_dict.object[8].bndbox
    # 只需要 最小坐标  xmin
    xmin = int(person_bndbox.xmin)
    # 读入图像
    img = cv2.imread(img_path)
    img = img[0:height,xmin:width]

    # 将裁剪下来的图像保存
    cv2.imwrite("./yi_caijian/"+str(j)+".jpg",img)
    print("第%d张图片裁剪完成"%(j))

