import numpy as np
import re
from collections import Counter
import random
import tqdm

listFiles=['00L-无动作1115','00L-右边非作弊20min','00R-无动作1115','00R-右边非作弊20min','60min','100min','120min']
totalTimeLength=45
intervalLength_perSample=45 #生成最终用于训练的样本时，每个样本在原始视频样本中对应的帧总数
interval_inSample=1 #生成最终用于训练的样本时，对原始视频样本进行的采样间隔
intervalSample=15 ##生成最终用于训练的样本时，相邻两个起始帧在原始视频样本中的间隔
threshold_percentNonZero=0.0
threshold_percentStdInMeanTotal=0.0
ceilNumAmpliPerStu=4 #最小1
floorClipPercent=0.9
ceilAmpliClip=2 #最小0

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
    numPerson=series.shape[1]
    start=0
    while start<timeLength:
          end=start+intervalLength_perSample
          if end>timeLength:
             end=timeLength
          dataValid=series[start:end:interval_inSample]
          validLen = dataValid.shape[0]
          
          data=np.zeros((totalTimeLength,numPerson,10,2))
          #在时间维度进行随机切片并对头尾帧进行随机长度的扩增复制，最后输出的时间长度为`totalTimeLength`.
          data = doRandomClipAndAmpli(data=dataValid,validLen=validLen,dataNew=data)
          # 进行归一化
          # 注意在这个位置进行是为了保持与实际运行时一致，实际运行时是最后才归一化的
          # 最好的情况是EGCN的输入长度基本没有冗余，也就是补0的情况尽量少
          xyMinT = np.min(data, axis=2, keepdims=True)
          xyLenT = np.max(data, axis=2, keepdims=True) - xyMinT
          # (timeStepLength, numStudents, 1, 2)
          xyMax = np.max(xyLenT, axis=0, keepdims=True)  # (1, numStudents, 1, 2)
          padT = (xyMax - xyLenT) * 0.5  # (timeStepLength, numStudents, 1, 2)
          data = (padT + data - xyMinT) / (xyMax + 1e-6)
          #存入标签库
          normalSeries.append(data)
          start+=intervalSample
    return normalSeries


if __name__=='__main__':
  #这里不排除错误Box，所有内容都需要是负样本不被检出，因此都应当归到`正常动作`中。
  normalSeries=[]
  for file in tqdm.tqdm(listFiles):
      series=np.load('../npy/1001/'+file+'_series.npy') #timeStepLength, numStudents, 10, 2
      normalSeries=retrieveLable(normalSeries,series)

  #Save
  np.save('../out/normal.npy',np.concatenate(normalSeries,axis=1).transpose(1,3,0,2)) #shape: N,C,T,V
  print('All Done.')
