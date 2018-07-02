'''
功能：生成测试列表的.mat文件
作者：老艾
联系：aixinyanchn@163.com
时间：2018-06-23 09:14:18
'''

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import scipy.io as sio
from PIL import Image

# 获取 每一个文件 名称，并且组成一个列表
new_list = ["./wu_biaozhu/"+x for x in os.listdir("./wu_biaozhu")]

# 生成 size 列表
size_list = []
for path in new_list:
    img_path = os.path.join(os.getcwd(), path)
    img = Image.open(img_path)
    one_size = [3, img.size[1], img.size[0]]

    size_list.append(one_size)

# 将两个列表生成 series
s_path = Series(new_list)
s_size = Series(size_list)

df = DataFrame( {"image":s_path, "size":s_size} )

print(df)
#print(s_path)
#print(s_size)

sio.savemat( "./create_test_dataset.mat",{"dataset":df} )

data = sio.loadmat( "./create_test_dataset.mat" )
print(data)