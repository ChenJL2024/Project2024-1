import cv2
import numpy as np
import re
import datetime
import math
from collections import Counter
import random
import tqdm

listFiles=['4','5','6','7','8','9','10','11','12','13','14','15','16','17']
patrnTitle=re.compile('(\d*).mp4')
FPS_extractedVideo=3 #生成原始视频样本时设置的每秒抽帧数。注意这里要按照真正的设置来填，比如对于30帧率的视频，生成
                      #样本时设置的20，导致实际抽帧间隔是1帧（30/20取整导致），则这里还是应该填写为20.
interval_inSample=1 #生成最终用于训练的样本时，对原始视频样本进行的采样间隔.(即中间空`interval_inSample-1`帧)
totalTimeLength=45
threshold_percentNonZero=0.95
threshold_percentStdInMeanTotal=0.05
ceilNumAmpliPerStu=3 #最小1
floorClipPercent=0.95
ceilAmpliClip=3 #最小0

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


def read_srt_file_gen(file):
  with open(file, "r") as fs:
    for data in fs.readlines():
      yield data

def retrieveLable(lableDict,file,series,frameInterval,fps):
    fileGen=read_srt_file_gen('./'+file+'.srt')
    maxLenTime=0.0
    while True:
      try:
        item=next(fileGen)
        if "--> " in item:
          time_arr = item.split('--> ')
          start_time = time_arr[0].replace(" ", "")
          end_time = time_arr[1].replace(" ", "").replace("\n", "")
          start_time = datetime.datetime.strptime(start_time + "0", "%H:%M:%S,%f")
          end_time = datetime.datetime.strptime(end_time + "0", "%H:%M:%S,%f")
          start=start_time.hour*3600 + start_time.minute*60 + start_time.second + start_time.microsecond*0.000001
          end=end_time.hour*3600 + end_time.minute*60 + end_time.second + end_time.microsecond*0.000001
          lenTimeTemp=end-start
          if lenTimeTemp>maxLenTime:
             maxLenTime=lenTimeTemp
          start=int(max(int(start*fps)-1, 0)/frameInterval)
          end=math.ceil(max(int(end*fps)-1, 0)/frameInterval)
          assert end>start,'Wrong: `end` dosen\'t large than `start`.'
          currentLabel=next(fileGen)
          currentSeriNum=next(fileGen)
          #series: timeStepLength, numStudents, 10, 2
          assert interval_inSample==1,'Wrong: only `1` support for `interval_inSample`.'
          deviationStart = start % totalTimeLength
          
          
          
          dataValid=series[start:end+1:interval_inSample][:,[currentSeriNum],:,:]
          assert len(dataValid.shape)==4,'Wrong: the shape of dataValid is incorrect.'
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          validLen=dataValid.shape[0]
          
          #下边过滤零值学生
          numNonZeros=Counter(np.where((dataValid**2).sum(axis=-1).sum(axis=-1)>0.0)[-1])
          indexValid=[]
          threshold_numNonZero=(validLen)*threshold_percentNonZero
          for i in numNonZeros.keys():
            if numNonZeros[i]>=threshold_numNonZero:
              indexValid.append(i)
          #下边过滤动作幅度异常小的学生
          stdStus=np.mean(dataValid[:,indexValid,:,:][:,:,:,0].std(axis=0)+dataValid[:,indexValid,:,:][:,:,:,1].std(axis=0),\
                          axis=-1) #len(indexValid)
          indexValid_2=np.where(stdStus>np.mean(stdStus)*threshold_percentStdInMeanTotal)[0]
          indexValid=[indexValid[i] for i in indexValid_2]
          if len(indexValid)>0:
            dataValid=dataValid[:,indexValid,:,:] #timeStepLength, numStudents, 10, 2
            #对学生进行复制，达到扩增数据集的目的
            dataValidTemp=[]   
            numPerson=0
            for i in range(len(indexValid)):  
                numAmpli=random.randint(1,ceilNumAmpliPerStu)
                numPerson+=numAmpli
                dataValidTemp.append(np.repeat(dataValid[:,i,:,:][:,None,:,:],numAmpli,axis=1))
            dataValid=np.concatenate(dataValidTemp,axis=1)
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
            lable=next(fileGen).replace(" ", "").replace("\n", "")
            if lable not in lableDict:
               lableDict[lable]=[]
            lableDict[lable].append(data) 
      except StopIteration:
            break
    return lableDict,maxLenTime       



if __name__=='__main__':
  #获得标签及对应内容
  lableDict={}
  maxLenTime=0.0
  for file in tqdm.tqdm(listFiles):
      cap=cv2.VideoCapture('./rawVideo/'+file+'.mp4')
      fps = cap.get(cv2.CAP_PROP_FPS)
      frameInterval = fps /FPS_extractedVideo
      frameInterval = 1 if frameInterval <= 1.0 else int(frameInterval)
      cap.release()
      series=np.load('./'+file+'_series.npy') #timeStepLength, numStudents, 10, 2
      lableDict,lenTimeTemp=retrieveLable(lableDict,file,series,frameInterval,fps)
      if lenTimeTemp>maxLenTime:
         maxLenTime=lenTimeTemp

  #Save
  for lable in lableDict.keys():
     np.save('./out/'+lable+'.npy',np.concatenate(lableDict[lable],axis=1).transpose(1,3,0,2)) #shape: N,C,T,V
  print('All Done.')   
  print(f'Max time len: {maxLenTime}s.')
