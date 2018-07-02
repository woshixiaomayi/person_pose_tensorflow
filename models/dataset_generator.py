# -*- coding: utf-8 -*-


from tqdm import tqdm
import scipy.io as sio
import numpy as np
import copy


def dataset_generator(img_path,img_size,keypoints):
    img_path_type = np.array([img_path],dtype = '<U%s'%(len(img_path)))
    img_size_type = np.array([img_size],dtype = 'uint8')
    # very special
    module_array_transfer = copy.copy(module_array)
    module_array_transfer[0][0] = np.array(keypoints,dtype = 'int32')
    sample = (img_path_type,img_size_type,module_array_transfer)
    return sample

def mat_generator(img_path_list,img_size_list,keypoints_list):
    data_list = [dataset_generator(data[0],data[1],data[2]) for data in tqdm(zip(img_path_list,img_size_list,keypoints_list) ) if data[2]!=[]]
    result = {'__globals__': [],
              '__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Thu Mar  9 19:30:01 2017',
              '__version__': '1.0',
              'dataset':np.array([data_list],dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')])}
    return result


if __name__ == '__main__':
    # demo
    # there are two pictures infomations.
    img_path_list = ['./dir/dataset/im00006_3.png','./dir/dataset/im00005_1.png']
    img_size_list = [[  3, 252, 170],[  3, 391, 295]]  # [num_channels, image_height, image_width]
    keypoints_list = [  [[  3,  80, 187],              # 0-indexed joint ID, X coordinate, Y coordinate
           [  3,  26, 187],
           [  6,  77, 130],
           [  7, 104, 159],
           [  8,  81,  88],
           [  9,   9,  88],
           [ 10,   9, 148],
           [ 12,  43,  76],
           [ 13,  31,  11]],[[  0, 175, 261],
           [  1, 173, 178],
           [  2, 144, 122],
           [  3, 193, 124],
           [  4, 203, 146],
           [  5, 199, 153],
           [  6, 166, 144],
           [  7, 131, 107],
           [  8, 163, 111],
           [  9, 223, 122],
           [ 10, 224, 159],
           [ 11, 220, 207],
           [ 12, 187, 126],
           [ 13, 226,  72]]  ]
      
    # data templates
    module_name = '/models/dataset_example.mat'
    mlab = sio.loadmat(module_name)['dataset']
    module_array = mlab[0][0][2]
    print(module_array)
    
    # save
    save_path = '/models/train_dataset.mat'
    sio.savemat(save_path,mat_generator(img_path_list,img_size_list,keypoints_list))
