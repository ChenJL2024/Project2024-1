import math

import numpy as np
import re
import datetime
from collections import Counter
import random
import tqdm
from math import ceil

rootPath="../"
# listFiles = ['传递中','传递右','传递左', '01R-旁窥', '03R-传递纸条','03R-传递纸条1115', '03L-传递纸条','03L-传递纸条1115']
# listFiles = ['03R-传递纸条1115', '03L-传递纸条1115']
listFiles = ['01L-旁窥', '01L-旁窥1115', '01R-旁窥', '01R-旁窥1115', '02L-回头1', '02L-回头2','02L-回头1115','02R-回头1','02R-回头2','02R-回头1115','03L-传递纸条','03L-传递纸条1115',\
'03R-传递纸条','03R-传递纸条1115','传递右','传递左','传递中','回头右','回头左','回头中','旁窥右','旁窥左','旁窥中']
FPS_extractedVideo = 3 #所标注样本的帧率
interval_inSample = 1.0 #浮点数，生成最终用于训练的样本时，对series的采样间隔.(即中间空`interval_inSample-1`帧)
totalTimeLength = 45 #EGCN模型的时间维度大小
threshold_percentNonZero = 0.9 #容许的最低有效识别率（有效帧数(非零帧)占总EGCN的长度）
threshold_stdMeanTotal = -0.1 #这里取负值是为了对任意幅度的都采样，即不动的样本(std为0的)也不过滤。
ceilNumSamplesPerStu = 6 #对一个学生样本所作的总采样个数上限，最小1
floorClipPercent = 0.9 #一个作弊片段最少采样帧数比例为多少时，该样本还认为有效。
num_zeroSamples = 64 #对于非预警类别(normal类别)中，全零样本的构造个数。
whether_existNoneValidTime = False #是否需要校验有无起、终帧标签。对于存在非考试时段的视频，需要作此校验。

# 以下内容无需更改，仅需设置上边参数即可
#######################################
class oneSample:
    def __init__(self,lenTotal,behav_timeSlot,series,startBias):
        self.lenTotal=lenTotal #series的总长度(帧数)
        self.series=series #timeStepLength,numStudents(1),10,2
        self.behav_timeSlot = [] #[(label,start,end),...], end是按照python的规则，即end=最后一帧标号+1
        for timeSlot in behav_timeSlot:
            self.behav_timeSlot.append((timeSlot[0],timeSlot[1]-startBias,timeSlot[2]-startBias))
        self.behav_noLabel=self.cal_complementarySet() # [(start,end),...]

    def cal_complementarySet(self):
        behav_noLabel=[]
        ptr=0
        for item in self.behav_timeSlot:
            start=item[1]
            if start-1>ptr:
                behav_noLabel.append((ptr,start))
            ptr=item[2]
        if self.lenTotal-1 > ptr:
            behav_noLabel.append((ptr, self.lenTotal))
        return behav_noLabel

    def getData(self):
        data=dict()
        numLabels=len(self.behav_timeSlot)
        for i,timeSlot in enumerate(self.behav_timeSlot):
            start = timeSlot[1] if timeSlot[1]>=0 else 0
            end = timeSlot[2] if timeSlot[2]<=self.lenTotal else self.lenTotal
            assert end>start, "Error: end <= start."
            if i==0:
                availStart = 0
            else:
                availStart = self.behav_timeSlot[i-1][2]
                availStart = availStart if availStart<start else start
            if i==numLabels-1:
                availEnd = self.lenTotal
            else:
                availEnd = self.behav_timeSlot[i+1][1]
                availEnd = availEnd if availEnd<=self.lenTotal else self.lenTotal
                availEnd = availEnd if availEnd>end else end
            # 正常来说，对一段标注好的数据当前标注的起点到下一个标注的起点为一个有效数据
            # 前一个标注的终点到当前标注的终点为一段待处理数据
            # 上面的处理逻辑会包含中间有一段无动作的数据，两端有动作的数据中间不能空闲太长，否则会造成无效数据
            # 这里需要加一段是否大于15s/45帧时长的判断，如果大于的话就靠边抽取样本数据，当前是否包含？
            # if (availEnd-availStart)>45:
            #     print('这段样本大于15秒,如果太大的话，有可能造成采样到无效样本',(availEnd-availStart))
            #     print('有动作的开始和采样间的开始差了做少',start-availStart)
            #     print('采样的结束和有动作的结束差了多少',availEnd-end)
            avail_interval_inSample=interval_inSample # 最终样本的采样间隔
            if interval_inSample-int(interval_inSample)>0.0:
               if random.randint(0,1)==0:
                   avail_interval_inSample = int(interval_inSample)
               else:
                   avail_interval_inSample = ceil(interval_inSample)
            else:
                avail_interval_inSample = int(avail_interval_inSample)
            start = int((start-availStart)/avail_interval_inSample)
            end = ceil((end-availStart)/avail_interval_inSample)+1
            dataTemp = self.series[availStart:availEnd:avail_interval_inSample]
            end = end if len(dataTemp)>=end else len(dataTemp)
            availStart_ori = availStart
            availEnd_ori = availEnd
            dataTemp = self.gen_onePieceData(dataTemp,start,end)
            # dataTemp为采样片段，将实际有动作的开始和结束(start,end)也传进去了
            if dataTemp is not None:
                label = timeSlot[0]
                if label not in data:
                    data[label]=[]
                data[label]+=dataTemp
        return data

    def getNormalData(self):
        data = []
        for i, timeSlot in enumerate(self.behav_noLabel):
            avail_interval_inSample = interval_inSample
            if interval_inSample - int(interval_inSample) > 0.0:
                if random.randint(0, 1) == 0:
                    avail_interval_inSample = int(interval_inSample)
                else:
                    avail_interval_inSample = ceil(interval_inSample)
            else:
                avail_interval_inSample = int(avail_interval_inSample)
            dataTemp = self.series[timeSlot[0]:timeSlot[1]:avail_interval_inSample]
            end = len(dataTemp)
            dataTemp = self.gen_onePieceData(dataTemp, 0, end)
            if dataTemp is not None:
                data += dataTemp
        #下边构造全零样本
        data.append(np.zeros((totalTimeLength, num_zeroSamples, 10, 2)))
        return data

    def gen_onePieceData(self,series,start,end):
        # series是包括正常动作的完整片段，可能在前后会包含无动作片段
        # start、end：有效动作片段的起、终帧标号（相对于series，从0开始. 终帧+1,符合python规则）
        dataValid = series[start:end]
        threshold_numNonZero = dataValid.shape[0] * threshold_percentNonZero
        # 下边检测零值占比
        numNonZeros = Counter(np.where((dataValid ** 2).sum(axis=-1).sum(axis=-1) > 0.0)[-1])[0]
        if numNonZeros < threshold_numNonZero:
           return None
        # 下边检测动作幅度
        # 这里按std实际值来过滤，非`V`字头的版本会按照所有学生的平均幅度来过滤，具体见相应代码
        stdStus = np.mean((dataValid.std(axis=0).mean(axis=-1)), axis=-1)[0]
        if stdStus <= threshold_stdMeanTotal:
            return None
        # 对学生进行复制，达到扩增数据集的目的
        numSamples=0
        for _ in range(ceil(dataValid.shape[0] / totalTimeLength)):
            numSamples += random.randint(1, ceilNumSamplesPerStu)
        data=[]
        for _ in range(numSamples):# 在时间维度进行采样.
            data.append(self.doRandomClipAndAmpli(data=series, start=start, end=end)) #实际上有动作的标注片段
        return data

    def doRandomClipAndAmpli(self, data, start, end):
        totalLen = data.shape[0]
        totalLen_ori = totalLen
        start_ori = start
        end_ori = end
        validLen = end - start
        clipLen = min(\
            random.randint(int(validLen * floorClipPercent), validLen), totalTimeLength)
        # floorClipPercent 越大代表有动作的完整片段包含的越多，实际上下面的start变化会越小
        # 因为validLen和clipLen基本一致
        start += random.randint(0, validLen-clipLen)
        validLen = clipLen # 更新有效的动作片段
        dataNew = np.zeros((totalTimeLength, 1, 10, 2)) # 初始化一段样本的大小，待往里面填充值，填充值的部分需要仔细考虑


        start_valindInNew = random.randint(0, totalTimeLength-validLen) #有动作片段和最终样本长度的差距，随机整数采样
        #有`InNew`标识的表示是在dataNew中的标号。
        availStart = start - min(start, start_valindInNew)
        totalLen = min(totalTimeLength-start_valindInNew, totalLen-availStart)
        totalLen_InNew=totalLen
        availstart_InNew = start_valindInNew-start+availStart

        leftLenAmpli=0
        rightLenAmpli=0
        if totalLen < totalTimeLength:
            voidLen = totalTimeLength - totalLen
            leftLenAmpli = random.randint(0, voidLen)
            rightLenAmpli = random.randint(0, voidLen - leftLenAmpli)
            totalLen_InNew = leftLenAmpli + totalLen + rightLenAmpli
            availstart_InNew = random.randint(0, totalTimeLength - totalLen_InNew)
        # 第一种采样方式，和第二种搭配使用
        # 如果采样数据的总长度在45帧以上，为什么还要补0呢，因为在大部分情况下，模型在真实推导的时候补0的情况很少
        # 所以接下来的采样操作，在45帧以上，用两边的空余帧进行补齐，如果不够45帧，边缘帧进行repeat操作
        if totalLen_ori == totalTimeLength:
            dataNew = data # 待采样序列等于45的话，直接赋值
        elif totalLen_ori > totalTimeLength:
            print('totalLen_ori,start_ori,end_ori:',totalLen_ori,start_ori,end_ori)
            oneLen_ori = math.ceil((totalTimeLength-totalLen_InNew)/2) #向上取整
            anotherLen_ori = totalTimeLength-totalLen_InNew-oneLen_ori
            # 如果在45以上，那么上面两个参数必定一个在左边，一个在右边
            leftLenAmpli_ori, rightLenAmpli_ori = (oneLen_ori, anotherLen_ori) if availstart_InNew >= (totalLen_ori-totalLen_InNew-availstart_InNew) else (anotherLen_ori, oneLen_ori)
            print('leftLenAmpli_ori,rightLenAmpli_ori',leftLenAmpli_ori,rightLenAmpli_ori)
            print('*******availstart_InNew,leftLenAmpli_ori:', availstart_InNew, leftLenAmpli_ori)
            if availstart_InNew < leftLenAmpli_ori:
                leftLenAmpli_ori_back = leftLenAmpli_ori
                leftLenAmpli_ori = availstart_InNew
                rightLenAmpli_ori += (leftLenAmpli_ori_back-availstart_InNew)
            elif totalLen_ori-totalLen_InNew-availstart_InNew < rightLenAmpli_ori:
                print('totalLen_ori-totalLen_InNew-2*availstart_InNew , rightLenAmpli_ori:',\
                      totalLen_ori - totalLen_InNew - 2 * availstart_InNew, rightLenAmpli_ori)
                rightLenAmpli_ori_back = rightLenAmpli_ori
                rightLenAmpli_ori = totalLen_ori-totalLen_InNew-availstart_InNew
                leftLenAmpli_ori = leftLenAmpli_ori+(rightLenAmpli_ori_back-(totalLen_ori-totalLen_InNew-availstart_InNew))
            totalTimeLength_45 = leftLenAmpli_ori+totalLen_InNew+rightLenAmpli_ori
            print('availstart_InNew,leftLenAmpli_ori:',availstart_InNew,leftLenAmpli_ori)
            print('totalLen_ori-totalLen_InNew-2*availstart_InNew , rightLenAmpli_ori:',\
                  totalLen_ori-totalLen_InNew-2*availstart_InNew , rightLenAmpli_ori)
            print('totalLen_InNew:',totalLen_InNew)
            print('rightLenAmpli_ori:',rightLenAmpli_ori)
            dataNew = np.concatenate([data[availstart_InNew-leftLenAmpli_ori:availstart_InNew], \
                 data[availstart_InNew:availstart_InNew + totalLen_InNew], \
                 data[availstart_InNew + totalLen_InNew:availstart_InNew + totalLen_InNew+rightLenAmpli_ori]],axis=0)
                        # 这个dataNew 是没有0的采样数据，即采样片段在45以上，在左右两边扩展非零和非静止状态
            print('7777777777777777777777777777777',dataNew.shape[0])

        # 第二种采样方式，较为合理的采样
        # 先判断左右两边是否还有空余帧，如果有的话就先扩增，然后在np.repeat，
        # 但是这样依然会是的最两边有补0操作，因为随机采样的totalLen_InNew < totalTimeLength_45,
        # 这种方法和下面的直接np.repeat然后补0操作，在原理上补0的样本数量是一样的，只是减少了np.repeat的数量
        else:
            if (availStart-leftLenAmpli)>=0 and (availStart + totalLen+rightLenAmpli)<=totalLen_ori:
                dataNew[availstart_InNew:availstart_InNew + totalLen_InNew] = np.concatenate(
                    [data[availStart-leftLenAmpli:availStart], \
                     data[availStart:availStart + totalLen], \
                     data[(availStart + totalLen):(availStart + totalLen+rightLenAmpli)]],axis=0)
            elif (availStart-leftLenAmpli)>=0 and (availStart + totalLen + rightLenAmpli)>totalLen_ori:
                dataNew[availstart_InNew:availstart_InNew + totalLen_InNew] = np.concatenate(
                    [data[availStart - leftLenAmpli:availStart], \
                    data[availStart:availStart + totalLen], \
                    data[(availStart + totalLen-1):(totalLen_ori)], \
                    np.repeat(data[[(totalLen_ori-1)]], rightLenAmpli-((totalLen_ori)-(availStart + totalLen-1)), axis=0)],axis=0)
            elif(availStart-leftLenAmpli)<0 and (availStart + totalLen + rightLenAmpli)<=totalLen_ori:
                dataNew[availstart_InNew:availstart_InNew + totalLen_InNew] = np.concatenate(
                    [np.repeat(data[[0]], (leftLenAmpli-(availStart-0)), axis=0),\
                    data[0:availStart], \
                    data[availStart:availStart + totalLen], \
                    data[(availStart + totalLen):(availStart + totalLen+rightLenAmpli)]],axis=0)
            else:
                dataNew[availstart_InNew:availstart_InNew + totalLen_InNew] = np.concatenate(
                    [np.repeat(data[[0]], (leftLenAmpli - (availStart - 0)), axis=0), \
                    data[0:availStart], \
                    data[availStart:availStart + totalLen], \
                    data[(availStart + totalLen-1):(totalLen_ori)],\
                    np.repeat(data[[(totalLen_ori-1)]], rightLenAmpli-((totalLen_ori)-(availStart + totalLen-1)), axis=0)],axis=0)

        # 第三种采样方式，比较简单粗暴
        # 直接在两边np.repeat,然后补0，np.repeat相当于静止的画面，从另一方面说，它和直接补0没有本质上的区别
        # 如果说repeat五到六帧以上的数据，相当于静止了2s左右，还不如直接补0，因为EGCN正式推理过程中静止的情况是绝对没有的
        # dataNew[availstart_InNew:availstart_InNew+totalLen_InNew] = np.concatenate(
        #     [np.repeat(data[[availStart]], leftLenAmpli, axis=0), \
        #      data[availStart:availStart+totalLen], \
        #      np.repeat(data[[availStart+totalLen-1]], rightLenAmpli, axis=0)], \
        #     axis=0) # 左右均复制最后边缘一帧，然后在拼接
        #为了避免大量样本都包括0，这里再额外产生一个无零存在的样本
        if totalLen_InNew<totalTimeLength:
            dataNew_nonZero =np.concatenate(
                [np.repeat(data[[availStart]], availstart_InNew+leftLenAmpli, axis=0), \
                 data[availStart:availStart + totalLen], \
                 np.repeat(data[[availStart + totalLen - 1]], \
                    totalTimeLength-totalLen_InNew-availstart_InNew+rightLenAmpli, axis=0)], \
                axis=0)
            dataNew = np.concatenate([dataNew, dataNew_nonZero], axis=1)
        # 下边以1/5的概率新增一个采样：非标注时段全补零。
        if random.randint(0, 4) == 0:
            ampliSample = np.zeros((totalTimeLength, 1, 10, 2))
            ampliSample[start_valindInNew:start_valindInNew + validLen] = \
                data[start:start + validLen]
            dataNew = np.concatenate([dataNew, ampliSample], axis=1)
        return dataNew


def read_srt_file_gen(file):
    with open(file, "r",encoding='gb18030',errors='ignore') as fs:
        for data in fs.readlines():
            yield data
def retrieveLabel(file, numSamples, fps):
    fileGen = read_srt_file_gen(rootPath+'srt/' + file + '.srt')
    lenTime = 0.0
    sampleDict = dict()
    patrnNumber = re.compile('(\d)')
    for i in range(numSamples):
        sampleDict[i]=[]
    startLabel = None #这是为了兼容第一次的标注，对于存在非考试时段的视频，需要标注起、终点
    endLabel = None
    while True:
        try:
            item = next(fileGen)
            if "--> " in item:
                time_arr = item.split('--> ')
                start_time = time_arr[0].replace(" ", "")
                end_time = time_arr[1].replace(" ", "").replace("\n", "")
                start_time = datetime.datetime.strptime(start_time + "0", "%H:%M:%S,%f")
                end_time = datetime.datetime.strptime(end_time + "0", "%H:%M:%S,%f")
                start = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond * 0.000001
                end = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond * 0.000001
                lenTimeTemp = end - start
                if lenTimeTemp > lenTime:
                    lenTime = lenTimeTemp
                start = max(int(start * fps)-1, 0)
                end = ceil(end * fps)
                assert end > start, f'Wrong: `end` dosen\'t large than `start`. {item}'
                label = next(fileGen).replace(" ", "").replace("\n", "")
                if label=="Start":
                    startLabel=(start,end)
                    continue
                elif label=="End":
                    endLabel=(start,end)
                    continue
                nextLine = next(fileGen).replace(" ", "").replace("\n", "")
                if nextLine=="" or nextLine != "":
                    for key in sampleDict.keys():
                        sampleDict[key].append((label, start, end))
                    continue
                # while nextLine!="":
                #     validFlag=False
                #     if nextLine.find("，")>=0:
                #         raise Exception(f'Chinese Code - {file}')
                #     if patrnNumber.match(nextLine[:1]):
                #         validFlag = True
                #         addList = nextLine[1:].split(",")
                #         for index in addList:
                #             if index == "":
                #                 continue
                #             obj = int(index)
                #             if obj in sampleDict:
                #                 sampleDict[obj].append((label, start, end))
                #     if not validFlag:
                #         raise Exception(f'Unknown pattrn: {nextLine}')
                #     nextLine = next(fileGen).replace(" ", "").replace("\n", "")
        except StopIteration:
            break
    return sampleDict, lenTime, startLabel, endLabel


if __name__ == '__main__':
    lenTime = 0.0
    dataDict=dict()
    dataDict['normal']=[]
    for file in listFiles:
        print(f"Deal with {file}:")
        series = np.load(rootPath+'npy/' + file + '_series.npy')  # timeStepLength, numStudents, 10, 2
        lenTotal=series.shape[0]
        sampleDict,lenTimeTemp,startLabel,endLabel = retrieveLabel(file, series.shape[1], FPS_extractedVideo)
        startBias=0
        if whether_existNoneValidTime:
            if (startLabel is None) or (endLabel is None):
                raise Exception(f'No Start or End label! {file}.srt')
            series = series[startLabel[0]:endLabel[1]]
            lenTotal = series.shape[0]
            startBias = startLabel[0]
        if lenTimeTemp > lenTime:
            lenTime = lenTimeTemp
        for key in tqdm.tqdm(sampleDict.keys()):
            sample=oneSample(lenTotal,sampleDict[key],series[:,[key],:,:],startBias)
            dataTemp = sample.getData()
            dataDict['normal'] += sample.getNormalData()
            for label in dataTemp.keys():
                if label not in dataDict:
                    dataDict[label]=[]
                dataDict[label]+=dataTemp[label]
    print("\nSaving...")
    for key in tqdm.tqdm(dataDict.keys()):
        data=np.concatenate(dataDict[key], axis=1)
        # 归一化
        xyMinT = np.min(data, axis=2, keepdims=True)
        xyLenT = np.max(data, axis=2, keepdims=True) - xyMinT
        # (timeStepLength, numStudents, 1, 2)
        xyMax = np.max(xyLenT, axis=0, keepdims=True)  # (1, numStudents, 1, 2)
        padT = (xyMax - xyLenT) * 0.5  # (timeStepLength, numStudents, 1, 2)
        data = (padT + data - xyMinT) / (xyMax + 1e-6)
        # Save
        np.save(rootPath+'out/' + key + '.npy', data.transpose(1, 3, 0, 2))  # shape: N,C,T,V
    print('\nAll Done.')
    print(f'Max time len: {lenTime}s.')
