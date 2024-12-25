import os
import shutil
import numpy as np
from select_validdata import _check_peep_back_data


# files_path = '/home/goldsun/data/genData/out_peep'
# for root, dirs, files in os.walk(files_path):
#     for file in files:
#         if file.endswith('mp4'):
#             continue
#         npy_file = os.path.join(root, file)
#         npy_data = np.load(npy_file)
#         valid_data = _check_peep_back_data(npy_data)
#         child_path = npy_file.replace(file, '')
#         if valid_data:
#             positive_path = os.path.join(child_path, 'positive')
#             print(positive_path)
#             os.makedirs(positive_path, exist_ok=True)
#             shutil.move(npy_file, positive_path)
#             shutil.move(npy_file.replace('.npy', '.mp4'), positive_path)
#         else:
#             negative_path = os.path.join(child_path, 'negative')
#             os.makedirs(negative_path, exist_ok=True)
#             shutil.move(npy_file, negative_path)
#             shutil.move(npy_file.replace('.npy', '.mp4'), negative_path)

files_path = '/home/goldsun/data/genData/out_stand'
for root, dirs, files in os.walk(files_path):
    for file in files:
        npy_file = os.path.join(root, file)
        file_name = file.split('.')[0]
        child_path = npy_file.replace(file, '')
        if int(file_name) % 2 == 0:
            #sit
            sit_path = os.path.join(child_path, 'stand')
            os.makedirs(sit_path, exist_ok=True)
            shutil.move(npy_file, sit_path)

        else:
            #stand
            stand_path = os.path.join(child_path, 'sit')
            os.makedirs(stand_path, exist_ok=True)
            shutil.move(npy_file, stand_path)

        
