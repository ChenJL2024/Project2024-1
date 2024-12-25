import numpy as np  
import random
from random_samples import *

# 翻转操作的函数（这里直接实现翻转，因为翻转不依赖于缩放和平移的具体值）  
def flip_at_x960(points):  
    return 2 * 960 - points[..., 0], points[..., 1]  # 只翻转x坐标  
  
# 变换函数  
def transform_frame(index, frame, scale_x, scale_y, translate_x, translate_y):  
    # 缩放和平移  
    sucessful = True
    scaled_translated = frame * [scale_x, scale_y] + [translate_x, translate_y]  
    
    flipped_x, flipped_y = scaled_translated[...,0],scaled_translated[...,1]
    # 翻转（在x=960处）  
    if index % 5 == 0:
        flipped_x, flipped_y = flip_at_x960(scaled_translated)  
      
    # 合并翻转后的x和y坐标  
    flipped_frame = np.stack((flipped_x, flipped_y), axis=-1)  
      
    # 检查是否超出范围，并打印警告（或根据需要进行其他处理）  
    if np.any(flipped_frame[..., 0] < 0) or np.any(flipped_frame[..., 0] > 1920) or np.any(flipped_frame[..., 1] < 0) or np.any(flipped_frame[..., 1] > 1080): 
        # print("Warning: Some points are out of the canvas bounds after transformation.")
        sucessful = False
    
    return flipped_frame, sucessful

if __name__ == '__main__':

    initial_data = np.load('00000.npy')
    samples = get_samples()
    print(samples)
    for index, sample in enumerate (samples):
        scale_x = sample[0]
        scale_y = sample[0]
        translate_x, translate_y = sample[1], sample[2]
        # 对每一帧应用变换  
        transformed_data = np.empty_like(initial_data)
        sucess_list = []  
        for i in range(transformed_data.shape[0]):  
            transformed_data[i], sucessful = transform_frame(index, initial_data[i], scale_x, scale_y, translate_x, translate_y)
            sucess_list.append(sucessful)  
        # print(sucess_list)
        if False in sucess_list:
            print(sucess_list)
            continue
        else:
            np.save(f'{index+1}.npy',transformed_data)
        
