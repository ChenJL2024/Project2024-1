import numpy as np
import re
from collections import Counter
import random
import tqdm
import os



totalTimeLength=45
intervalLength_perSample=45 #生成最终用于训练的样本时，每个样本在原始视频样本中对应的帧总数
interval_inSample=1 #生成最终用于训练的样本时，对原始视频样本进行的采样间隔
intervalSample=15 ##生成最终用于训练的样本时，相邻两个起始帧在原始视频样本中的间隔
threshold_percentNonZero=0.95
threshold_percentStdInMeanTotal=0.1
ceilNumAmpliPerStu=1 #最小1
floorClipPercent=0.9
ceilAmpliClip=1 #最小0

#以下内容无需更改，仅需设置上边参数即可
#######################################
def doRandomClipAndAmpli(data,validLen,dataNew): #shape:T,N,V,C
    validIndex=[]
    for i in range(data.shape[1]):
       if validLen<totalTimeLength*0.2:
           continue
       if validLen>=totalTimeLength:
           validIndex.append(i)
           start = random.randint(0, validLen-totalTimeLength)
           dataNew[:, i, :, :] = data[start:start+totalTimeLength,:,:,:][:,i,:,:]
           continue

       validIndex.append(i)
       clipLen=random.randint(int(validLen*floorClipPercent),validLen)
       if clipLen<totalTimeLength*0.2:
          clipLen=validLen
       redundLen=validLen-clipLen
       start=random.randint(0,redundLen)
       dataCurrentStu=data[start:start+clipLen,:,:,:][:,i,:,:] #T,V,C
       voidLen=min(totalTimeLength-clipLen, ceilAmpliClip)
       leftLenAmpli=random.randint(0,voidLen)
       rightLenAmpli=random.randint(0,voidLen-leftLenAmpli)
       totalLen=leftLenAmpli+clipLen+rightLenAmpli
       startSave=random.randint(0,totalTimeLength-totalLen)
       dataNew[startSave:startSave+totalLen,:,:,:][:,i,:,:]= np.concatenate(
               [np.repeat(dataCurrentStu[0,:,:][None,:,:], leftLenAmpli, axis=0),\
                dataCurrentStu,\
                np.repeat(dataCurrentStu[-1,:,:][None,:,:], rightLenAmpli, axis=0)],\
               axis=0)        
    return dataNew[:,validIndex,:,:]           


def retrieveLable(normalSeries,series):
    timeLength=series.shape[0]
    start=0
    # start = 360 #去掉前面两分钟的数据
    while start<timeLength:
        end=start+intervalLength_perSample
        if end>timeLength:
            end=timeLength
        dataValid=series[start:end:interval_inSample]
        validLen=dataValid.shape[0]

        #下边过滤零值学生
        numNonZeros=Counter(np.where((dataValid**2).sum(axis=-1).sum(axis=-1)>0.0)[-1])
        indexValid=[]
        # 这里去除漏掉超过3帧的目标
        threshold_numNonZero=(validLen)*threshold_percentNonZero
        for i in numNonZeros.keys():
            if numNonZeros[i]>=threshold_numNonZero:
                indexValid.append(i)
        # 下边过滤动作幅度异常小的学生，
        stdStus=np.mean(dataValid[:,indexValid,:,:][:,:,:,0].std(axis=0) + dataValid[:,indexValid,:,:][:,:,:,1].std(axis=0),axis=-1) #len(indexValid)
    #   print(stdStus)
        indexValid_2=np.where(stdStus>threshold_percentStdInMeanTotal)[0]
    #   print(indexValid_2)
        indexValid=[indexValid[i] for i in indexValid_2]
        # print(len(indexValid))
        if len(indexValid)>0:
            dataValid=dataValid[:,indexValid,:,:] #timeStepLength, numStudents, 10, 2
            # #对学生进行复制，达到扩增数据集的目的
            # dataValidTemp=[]
            # numPerson=0
            # for i in range(len(indexValid)):
            #     numAmpli=random.randint(1,ceilNumAmpliPerStu)
            #     numPerson+=numAmpli
            #     aa = dataValid[:,i,:,:][:,None,:,:]
            #     dataValidTemp.append(np.repeat(dataValid[:,i,:,:][:,None,:,:],numAmpli,axis=1))
            # dataValid=np.concatenate(dataValidTemp,axis=1)
            # sample_data=np.zeros((totalTimeLength,numPerson,10,2))
            # #在时间维度进行随机切片并对头尾帧进行随机长度的扩增复制，最后输出的时间长度为`totalTimeLength`.
            # sample_data = doRandomClipAndAmpli(data=dataValid,validLen=validLen,dataNew=sample_data)
            if dataValid.shape[0]!= totalTimeLength:
               all_zeros_np = np.zeros((totalTimeLength-dataValid.shape[0],dataValid.shape[1],10,2))
               dataValid = np.concatenate((dataValid,all_zeros_np),axis=0)
            print(dataValid.shape)
            frames, targets, _, _ = dataValid.shape
            for target in range(targets):
                data_info = dataValid[:,target,:,:]
                data_info = np.expand_dims(data_info,axis=1)
                target_numNonZeros = Counter(np.where((data_info ** 2).sum(axis=-1).sum(axis=-1) > 0.0)[-1])[0]
                if target_numNonZeros >= totalTimeLength - 2:  ## 中间可能出现几帧的坐标都为0,两头也可能出现0，超过两帧目标丢失，就把此目标丢掉
                ## 这里的逻辑需要变一下，最理想的状况是15帧都不为0，
                ## 最多会出现2帧为0的状况，先考虑只有一帧为0的情况，那么target_numNonZeros==14
                    if target_numNonZeros == totalTimeLength - 1:
                        for frame in range(frames):
                            if (np.expand_dims(data_info[frame,:,:,:], axis=0) ** 2).sum(axis=-1).sum(axis=-1) == 0.0:
                                if frame == 0:
                                    data_info[frame,:,:,:] = data_info[frame+1,:,:,:]

                                elif frame == totalTimeLength - 1:
                                    data_info[frame,:,:,:] = data_info[frame-1,:,:,:]# 如果为0的位置出现在两头，也要给他补齐

                                else:
                                    data_info[frame,:,:,:] = (data_info[frame-1,:,:,:]+data_info[frame+1,:,:,:])/2 #如果出现在中间的话，需要结合前后帧取均值

                    ## 不等于14，就等于13
                    elif target_numNonZeros == totalTimeLength - 2:
                        frame_index_zero = [] # 寻找为0的索引
                        for frame in range(frames):
                            if (np.expand_dims(data_info[frame,:,:,:], axis=0) ** 2).sum(axis=-1).sum(axis=-1) == 0.0:
                                frame_index_zero.append(frame)
                        assert len(frame_index_zero) == 2, "Error: len(frame_index_zero) != 2."
                        if frame_index_zero == [0,1]:
                            data_info[0,:,:,:] = data_info[2,:,:,:]
                            data_info[1,:,:,:] = data_info[2,:,:,:]

                        elif frame_index_zero == [totalTimeLength - 2, totalTimeLength - 1]:
                            data_info[totalTimeLength - 2,:,:,:] = data_info[totalTimeLength - 3,:,:,:]
                            data_info[totalTimeLength - 1,:,:,:] = data_info[totalTimeLength - 3,:,:,:]

                        elif frame_index_zero == [0, totalTimeLength - 1]: 
                            data_info[0,:,:,:] = data_info[1,:,:,:]
                            data_info[totalTimeLength - 1,:,:,:] = data_info[totalTimeLength - 2,:,:,:]

                        elif frame_index_zero[0] == 0: # 有一帧出现在开始，另一帧肯定在中间，取均值
                            data_info[frame_index_zero[0],:,:,:] = data_info[1,:,:,:]
                            data_info[frame_index_zero[1],:,:,:] = (data_info[frame_index_zero[1]-1,:,:,:]+data_info[frame_index_zero[1]+1,:,:,:])/2

                        elif frame_index_zero[1] == totalTimeLength - 1: # 有一帧出现在结尾，另一帧肯定在中间，取均值
                            data_info[frame_index_zero[1],:,:,:] = data_info[-2,:,:,:]
                            data_info[frame_index_zero[0],:,:,:] = (data_info[frame_index_zero[0]-1,:,:,:]+data_info[frame_index_zero[0]+1,:,:,:])/2

                        elif frame_index_zero[1] - frame_index_zero[0] != 1: #漏检两帧在中间且不连续，两帧都取均值
                            data_info[frame_index_zero[0],:,:,:] = (data_info[frame_index_zero[0]-1,:,:,:]+data_info[frame_index_zero[0]+1,:,:,:])/2
                            data_info[frame_index_zero[1],:,:,:] = (data_info[frame_index_zero[1]-1,:,:,:]+data_info[frame_index_zero[1]+1,:,:,:])/2

                        else: # 漏检的两帧在中间且连续，则两边赋值
                            data_info[frame_index_zero[0],:,:,:] = data_info[frame_index_zero[0]-1,:,:,:]
                            data_info[frame_index_zero[1],:,:,:] = data_info[frame_index_zero[0]+1,:,:,:]

                    normalSeries.append(data_info)

            # 进行归一化
            # 注意在这个位置进行是为了保持与实际运行时一致，实际运行时是最后才归一化的
            # 最好的情况是EGCN的输入长度基本没有冗余，也就是补0的情况尽量少
            # xyMinT = np.min(data, axis=2, keepdims=True)
            # xyLenT = np.max(data, axis=2, keepdims=True) - xyMinT
            # # (timeStepLength, numStudents, 1, 2)
            # xyMax = np.max(xyLenT, axis=0, keepdims=True)  # (1, numStudents, 1, 2)
            # padT = (xyMax - xyLenT) * 0.5  # (timeStepLength, numStudents, 1, 2)
            # data = (padT + data - xyMinT) / (xyMax + 1e-6)
            #存入标签库
            
        #   start+=intervalSample
        start = end
    return normalSeries

def _not_stand_data(data):
    frames, targets, keypoints, coordinate = data.shape
    target_indices = []
    for target in range(targets):
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        target_y = np.diff(data[nonzero_frame_indice, target, 5, :], axis=0)  # 5号点y轴坐标依次做差
        neg_indices = np.where(target_y[:, 1] < 0)[0].tolist()
        neck_point_y = data[nonzero_frame_indice, target, 5, 1]
        spine_point_y = data[nonzero_frame_indice, target, 6, 1]
        mean_shoulder_distance = np.sum(spine_point_y - neck_point_y) / len(nonzero_frame_indice)
        max_neck_y = max(data[nonzero_frame_indice, target, 5, 1])
        min_neck_y = min(data[nonzero_frame_indice, target, 5, 1])
        move_neck = max_neck_y - min_neck_y
        if len(neg_indices) > 2:
            for i in range(len(neg_indices) - 1):
                if neg_indices[i + 1] - neg_indices[i] == 1 and move_neck > mean_shoulder_distance:
                    target_indices.append(target)
                    break
    
    target_indices = [x for x in range(targets) if x not in target_indices]
    valid_data = data[:, target_indices, :, :]
def _not_raise_data():
   pass
 
if __name__=='__main__':
  
    #这里不排除错误Box，所有内容都需要是负样本不被检出，因此都应当归到`正常动作`中。
    normalSeries=[]
    npy_root_path = './npy/normal/'
    for root, dirs,files in os.walk(npy_root_path):
        for file in files:
            npy_file = os.path.join(root,file)
            print(npy_file)
            if npy_file.split('.')[-1] == 'mp4':
                os.remove(npy_file)
            if npy_file.split('.')[-1] == 'npy':
                series = np.load(npy_file)
                normalSeries=retrieveLable(normalSeries,series)
        #   for normal_data in normalSeries:
            #  normal_data = _not_stand_data(normal_data)
            
        
    # for file in tqdm.tqdm(listFiles):
    #     series=np.load('../npy/suda_npy/'+file+'_series.npy') #timeStepLength, numStudents, 10, 2
    #     normalSeries=retrieveLable(normalSeries,series)
    #Save

    data = np.concatenate(normalSeries,axis=1)
    np.save('./out/normal_original.npy', data)

    data_copy = data.copy()
    data_x = data_copy[...,0]
    data_y = data_copy[...,1]
    data_with_position = np.stack((data_x / 1920, data_y / 1080), axis= -1)
    np.save('./out/normal_global_normalize.npy', data_with_position)  # shape: N,C,T,V

    # 进行归一化
    # 注意在这个位置进行是为了保持与实际运行时一致，实际运行时是最后才归一化的
    # 最好的情况是EGCN的输入长度基本没有冗余，也就是补0的情况尽量少
    xyMinT = np.min(data, axis=2, keepdims=True)
    xyLenT = np.max(data, axis=2, keepdims=True) - xyMinT
    # (timeStepLength, numStudents, 1, 2)
    xyMax = np.max(xyLenT, axis=0, keepdims=True)  # (1, numStudents, 1, 2)
    padT = (xyMax - xyLenT) * 0.5  # (timeStepLength, numStudents, 1, 2)
    data = (padT + data - xyMinT) / (xyMax + 1e-6)

    #   np.save('../out/normal.npy',np.concatenate(normalSeries,axis=1).transpose(1,3,0,2)) #shape: N,C,T,V axis=1 学生维度的拼接
    np.save('./out/normal_single_normalize.npy',data.transpose(1, 3, 0, 2))
    print('All Done.')
