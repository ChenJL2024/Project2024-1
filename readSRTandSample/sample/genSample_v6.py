import numpy as np
import re
import math
import datetime
from collections import Counter
import random
import tqdm
from math import ceil


# listFiles = ['suda_left_stand_1','suda_left_stand_2','suda_right_stand_1','suda_right_stand_2', 
#              '182_left_stand_1','182_left_stand_2','188_right_stand_1','188_right_stand_2',
#                 '04L_stand1','04L_stand2','04R_stand1','04R_stand2']


FPS_extractedVideo = 3  #所标注样本的帧率
interval_inSample = 1.0  #浮点数，生成最终用于训练的样本时，对series的采样间隔.(即中间空`interval_inSample-1`帧)
totalTimeLength = 45 #EGCN模型的时间维度大小
threshold_percentNonZero = 0.95 #容许的最低有效识别率（有效帧数(非零帧)占总EGCN的长度）
threshold_stdMeanTotal = -0.1 #这里取负值是为了对任意幅度的都采样，即不动的样本(std为0的)也不过滤。
ceilNumSamplesPerStu = 16   #对一个学生样本所作的总采样个数上限，最小1
floorClipPercent = 0.95 #一个作弊片段最少采样帧数比例为多少时，该样本还认为有效。
num_zeroSamples = 1 #对于非预警类别(normal类别)中，全零样本的构造个数。
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
            # print(i,start,end)
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
            avail_interval_inSample=interval_inSample
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
            dataTemp = self.gen_onePieceData(dataTemp,start,end)
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
        # series是包括正常动作的完整片段
        # start、end：标注的动作片段的起、终帧标号（相对于series，从0开始. 终帧+1,符合python规则）
        dataValid = series[start:end]
        threshold_numNonZero = dataValid.shape[0] * threshold_percentNonZero # 在有效的片段内含0的阈值
        # 下边检测零值占比
        numNonZeros = Counter(np.where((dataValid ** 2).sum(axis=-1).sum(axis=-1) > 0.0)[-1])[0]
        if numNonZeros < threshold_numNonZero: ## 中间可能出现几帧的坐标可能出现0
           return None
        # 下边检测动作幅度
        stdStus = np.mean((dataValid.std(axis=0).mean(axis=-1)), axis=-1)[0]
        if stdStus <= threshold_stdMeanTotal: # 不动的样本等于0的话，或者全0的样本去掉
            return None
        # 对学生进行复制，达到扩增数据集的目的
        numSamples=0
        for _ in range(ceil(dataValid.shape[0] / totalTimeLength)):
            numSamples += random.randint(1, ceilNumSamplesPerStu)
        data=[]
        print('--------',numSamples)
        for _ in range(numSamples):
            # 在时间维度进行采样.
            sample_data = self.doRandomClipAndAmpli(data=series, start=start, end=end)

            # 不需要前后有0的样本，跳过此类样本
            frames, targets, _, _ = sample_data.shape
            for target in range(targets):
                data_info = sample_data[:,target,:,:]
                data_info = np.expand_dims(data_info,axis=1)
                target_numNonZeros = Counter(np.where((data_info ** 2).sum(axis=-1).sum(axis=-1) > 0.0)[-1])[0]
                if target_numNonZeros >= 43:  ## 中间可能出现几帧的坐标都为0,两头也可能出现0
                    ## 这里的逻辑需要变一下，两边有0的样本不管，中间为0的需要给它补上，最理想的状况是45帧都不为0，
                    ## 最多会出现2帧为0的状况，先考虑只有一帧为0的情况，那么target_numNonZeros==44
                    if target_numNonZeros == 44:
                        for frame in range(frames):
                            if (np.expand_dims(data_info[frame,:,:,:], axis=0) ** 2).sum(axis=-1).sum(axis=-1) == 0.0:
                                if frame == 0 or frame == 44:# 如果为0的位置出现在两头，就不用关心
                                    continue
                                data_info[frame,:,:,:] = (data_info[frame-1,:,:,:]+data_info[frame+1,:,:,:])/2 #如果出现在中间的话，需要结合前后帧取均值
                    ## 不等于44，等于43
                    elif target_numNonZeros == 43:
                        frame_index_zero = []
                        for frame in range(frames):
                            if (np.expand_dims(data_info[frame,:,:,:], axis=0) ** 2).sum(axis=-1).sum(axis=-1) == 0.0:
                                frame_index_zero.append(frame)
                        if frame_index_zero == [0,1] or frame_index_zero == [43, 44]or frame_index_zero == [0, 44]: #出现在两头就不用关心
                            continue
                        elif frame_index_zero[0] == 0: # 有一帧出现在开始，另一帧肯定在中间，取均值
                            data_info[frame_index_zero[1],:,:,:] = (data_info[frame_index_zero[1]-1,:,:,:]+data_info[frame_index_zero[1]+1,:,:,:])/2
                        elif frame_index_zero[1] == 44: # 有一帧出现在结尾，另一帧肯定在中间，取均值
                            data_info[frame_index_zero[0],:,:,:] = (data_info[frame_index_zero[0]-1,:,:,:]+data_info[frame_index_zero[0]+1,:,:,:])/2
                        elif frame_index_zero[1] - frame_index_zero[0] != 1: #漏检两帧在中间且不连续，两帧都取均值
                            data_info[frame_index_zero[0],:,:,:] = (data_info[frame_index_zero[0]-1,:,:,:]+data_info[frame_index_zero[0]+1,:,:,:])/2
                            data_info[frame_index_zero[1],:,:,:] = (data_info[frame_index_zero[1]-1,:,:,:]+data_info[frame_index_zero[1]+1,:,:,:])/2
                        else: # 漏检的两帧在中间且连续，则两边赋值
                            data_info[frame_index_zero[0],:,:,:] = data_info[frame_index_zero[0]-1,:,:,:]
                            data_info[frame_index_zero[1],:,:,:] = data_info[frame_index_zero[0]+1,:,:,:]
                    data.append(data_info)

        return data

    def doRandomClipAndAmpli(self, data, start, end):

        totalLen = data.shape[0]
        data_len = totalLen
        validLen = end - start
        clipLen = min(\
            random.randint(int(validLen * floorClipPercent), validLen), totalTimeLength)
        data_start = start
        # if validLen > clipLen:
        #     data_start += random.randint(0, validLen-clipLen)
        validLen = clipLen
        dataNew = np.zeros((totalTimeLength, 1, 10, 2))
        random_min = end - totalTimeLength
        # print("#########", totalLen, random_min, validLen, start, end)
        if random_min < 0:
            random_min = 0

        availStart = 0
        if random_min < data_start:
            availStart = random.randint(random_min, data_start)

        start_valindInNew = 0
        # print("!!!!!!!!", availStart, random_min, validLen, start, end)
        if data_start - availStart <= totalTimeLength - validLen:
            start_valindInNew = data_start-availStart
        #有`InNew`标识的表示是在dataNew中的标号。
        # totalLen = min(totalTimeLength-start_valindInNew, totalLen-availStart)
        if totalLen < availStart + totalTimeLength:
            totalLen = totalLen-availStart
        else:
            totalLen = totalTimeLength

        totalLen_InNew = totalLen
        availstart_InNew = 0
        leftLenAmpli=0
        rightLenAmpli=0
        if totalLen < totalTimeLength:
            voidLen = totalTimeLength - totalLen
            leftLenAmpli = random.randint(0, voidLen)
            rightLenAmpli = random.randint(0, voidLen - leftLenAmpli)
            totalLen_InNew = leftLenAmpli + totalLen + rightLenAmpli
            availstart_InNew = random.randint(0, totalTimeLength - totalLen_InNew)
        print("----------", availStart, availStart + totalLen, data_len, start, end)
        dataNew[availstart_InNew:availstart_InNew+totalLen_InNew] = np.concatenate(
            [np.repeat(data[[availStart]], leftLenAmpli, axis=0), \
             data[availStart:availStart+totalLen], \
             np.repeat(data[[availStart+totalLen-1]], rightLenAmpli, axis=0)], \
            axis=0)
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
                data[data_start:data_start + validLen]
            print(start_valindInNew,data_start,validLen,data.shape[0])
            dataNew = np.concatenate([dataNew, ampliSample], axis=1)
        return dataNew


def read_srt_file_gen(file):
    with open(file, "r", encoding='gb18030', errors='ignore') as fs:
        for data in fs.readlines():
            # print(data)
            yield data

def retrieveLabel(file, numSamples, fps):
    fileGen = read_srt_file_gen('./npy/' + file + '.srt')
    lenTime = 0.0
    sampleDict = dict()
    patrnNumber = re.compile('(\d)') # \d 代表对数字的匹配
    for i in range(numSamples):
        sampleDict[i]=[]  # 人数
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
                # 以下根据编号要删除或者只标少数的部分注释掉代表所有的片段都要
                # while nextLine!="":
                #     validFlag=False
                #     if nextLine.find("，")>=0:
                #         raise Exception(f'Chinese Code - {file}')
                #     if nextLine[:1]=="a":
                #        validFlag = True
                #        addList=nextLine[1:].split(",")
                #        for index in addList:
                #            if index=="":
                #                continue
                #            obj=int(index)-1 # 添加跟踪后，跟踪id是从1开始的，实际对应sampleDict中的key应该做减一操作
                #            if obj in sampleDict:
                #                sampleDict[obj].append((label,start,end))
                #     elif nextLine[:1] == 'd':
                #         validFlag = True
                #         delList = nextLine[1:].split(",")
                #         for key in sampleDict.keys():
                #             delkey = key+1 # 添加跟踪后，目标id是从1开始，所以标签文件中的（d num）应该是要删除num-1的目标
                #             if str(delkey) in delList:
                #                 continue
                #             sampleDict[key].append((label, start, end))
                #     elif (nextLine[:1]=="d" and nextLine[:4]!="dall") or\
                #             patrnNumber.match(nextLine[:1]): #匹配数字，如果直接跟数字的话也代表要删除目标的序号
                #         validFlag = True
                #         delList=nextLine[1:].split(",")
                #         for key in sampleDict.keys():
                #             if str(key) in delList:
                #                 continue
                #             sampleDict[key].append((label, start, end))
                #     elif nextLine[:4]=="dall":
                #         validFlag = True
                #         delList=nextLine[4:].split(",")
                #         for index in delList:
                #             if index == "":
                #                 continue
                #             obj=int(index)
                #             if obj in sampleDict:
                #                 del sampleDict[obj]
                #     if not validFlag:
                #         raise Exception(f'Unknown pattrn: {nextLine}')
                #     nextLine = next(fileGen).replace(" ", "").replace("\n", "")
        except StopIteration:
            break
    return sampleDict, lenTime, startLabel, endLabel

def calculate_single(A,B,C):
    vector_AB = B-A
    vector_BC = C-B

    dot_product = np.dot(vector_AB,vector_BC)
    # cross_product = np.cross(vector_AB,vector_BC)
    # angld_rad = np.arctan2(np.linalg.norm(cross_product),dot_product)
    # 计算向量的范数
    norm_AB = np.linalg.norm(vector_AB)
    norm_BC = np.linalg.norm(vector_BC)

    # 计算余弦值
    if norm_AB * norm_BC != 0:
        cos_theta = dot_product / (norm_AB * norm_BC)
    else:
        cos_theta = 0

    # 计算夹角（弧度）
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 将弧度转换为角度
    angle_deg = np.degrees(angle_rad)

    # angld_deg = np.degrees(angld_rad)
    return angle_deg

def angle_between_points(a, b, c):

    try:
        # Calculate vectors AB and BC
        AB = [b[0] - a[0], b[1] - a[1]]
        BC = [c[0] - b[0], c[1] - b[1]]

        # Calculate dot product of AB and BC
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]

        # Calculate magnitudes of AB and BC
        magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
        magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

        # Calculate angle in radians
        angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))

        # Convert angle to degrees
        angle_degrees = math.degrees(angle_radians)
    except:
        angle_degrees = 60 # 返回不满足条件的角度

    return angle_degrees

def _valid_passon_data(data):
    frames, targets, keypoints, coordinates = data.shape
    # 初始化存储差值的列表
    diff_lists = []
    move_lists = []
    # 循环遍历每个目标的每个关键点
    # nonzero_indices = np.nonzero(data)
    # targets_nonzero_indices = list(zip(*[index.flatten() for index in nonzero_indices]))

    for target in range(data.shape[1]):
        target_diffs = []  # 存储当前目标的差值列表
        target_move = []
        # 找到第一个坐标点不为0的帧的索引
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        # inver_data = data[::-1]
        # start_frame = np.argmax(np.any(data[:, target, :, :] != 0, axis=(1, 2)))
        # start_frame = np.argmax(np.any(data != 0,axis=-1))
        # end_fram = 45 - np.argmax(np.any(inver_data[:, target, :, :] != 0,axis=(1, 2)))
        # 循环遍历每个关键点
        for keypoint in range(data.shape[2]):
            # 计算当前关键点在相邻两帧之间的差值，从第一个非零坐标点的帧开始计算
            # keypoint_diff = np.diff(data[start_frame:end_fram, target, keypoint, :], axis=0)
            aa = data[nonzero_frame_indice, target, keypoint, :]
            max_x = np.max(data[nonzero_frame_indice, target, keypoint, 0])
            min_x = np.min(data[nonzero_frame_indice, target, keypoint, 0])
            keypoint_diff_move = np.diff(data[nonzero_frame_indice, target, keypoint, :], axis=0)
            keypoint_diff = max_x - min_x
            keypoint_diff_move = np.sum(np.sqrt(np.sum(keypoint_diff_move ** 2, axis=1)))
            target_diffs.append(keypoint_diff)  # 将差值转换为列表并添加到当前关键点的差值列表中,不能用方差，用移动累加值
            target_move.append(keypoint_diff_move)
        diff_lists.append(target_diffs)  # 将当前目标的差值列表添加到总列表中
        move_lists.append(target_move)

    all_targets_angle = {}
    all_targets_angle_ori = {}
    for target in range(targets):
        # print(target)
        # if target != 58:
        #     continue
        all_angles = []
        for frame in range(frames):
            if np.all(data[frame, target, 0, :] == data[frame, target, 6, :], axis=-1):
                arm_angles = [50, 50]
                all_angles.append(arm_angles)
                continue
            A_2 = data[frame, target, 2, :]
            B_3 = data[frame, target, 3, :]
            C_4 = data[frame, target, 4, :]
            right_angle = angle_between_points(A_2, B_3, C_4)
            A_7 = data[frame, target, 7, :]
            B_8 = data[frame, target, 8, :]
            C_9 = data[frame, target, 9, :]
            left_angle = angle_between_points(A_7, B_8, C_9)
            arm_angles = [right_angle, left_angle]
            all_angles.append(arm_angles)
            # all_targets_angle[target].append(arm_angles)
        all_targets_angle[target] = np.array(all_angles)
        all_targets_angle_ori[target] = np.array(all_angles)
        # min_index = np.unravel_index(np.argmin(all_targets_angle[target]),all_targets_angle[target].shape)
        # min_value = all_targets_angle[target][min_index]

        # min_index_right = np.argmin(all_targets_angle[target][:,0])
        # min_value_right = all_targets_angle[target][min_index_right]

        # min_index_left = np.argmin(all_targets_angle[target][:, 1])
        # min_value_left = all_targets_angle[target][min_index_left]

        max_index = diff_lists[target].index(max(diff_lists[target]))
        max_move_index = move_lists[target].index(max(move_lists[target]))

        # 右边胳膊（2，3，4），看4号点的移动幅度
        # if max_index == 4:
        if max_index == 4 and max_move_index == 4:
            right_indexes = list(np.where(abs(data[:, target, 4, 1] - data[:, target, 2, 1]) < abs(
                data[:, target, 4, 0] - data[:, target, 2, 0])))[0].tolist()
            if len(right_indexes) == 0:
                continue
            min_index_right = np.argmin(all_targets_angle[target][right_indexes, 0])
            min_value_right = all_targets_angle[target][right_indexes[min_index_right]]
            if min_value_right[0] > 20:
                all_targets_angle[target] = None
                continue
            if abs(data[right_indexes[min_index_right], target, 4, :][1] -
                   data[right_indexes[min_index_right], target, 2, :][1]) > \
                    abs(data[right_indexes[min_index_right], target, 4, :][0] -
                        data[right_indexes[min_index_right], target, 2, :][0]):
                all_targets_angle[target] = None
                continue
        # if max_index == 9:
        if max_index == 9 and max_move_index == 9:
            left_indexes = list(np.where(abs(data[:, target, 9, 1] - data[:, target, 7, 1]) < abs(
                data[:, target, 9, 0] - data[:, target, 7, 0])))[0].tolist()
            if len(left_indexes) == 0:
                continue
            min_index_left = np.argmin(all_targets_angle[target][left_indexes, 1])
            min_value_left = all_targets_angle[target][left_indexes[min_index_left]]
            if min_value_left[1] > 20:
                all_targets_angle[target] = None
                continue

            if abs(data[left_indexes[min_index_left], target, 9, :][1] -
                   data[left_indexes[min_index_left], target, 7, :][1]) > \
                    abs(data[left_indexes[min_index_left], target, 9, :][0] -
                        data[left_indexes[min_index_left], target, 7, :][0]):
                print(data[min_index_left, target, 9, :], data[min_index_left, target, 7, :])
                all_targets_angle[target] = None
                continue
        if max_index != 4 and max_index != 9:
            all_targets_angle[target] = None
        if max_move_index != 4 and max_move_index != 9:
            all_targets_angle[target] = None

    key_not_none = [key for key, value in all_targets_angle.items() if value is not None]
    valid_data = data[:, key_not_none, :, :]
    return valid_data

def _valid_peep_data(data):
    frames, targets, keypoints, coordinates = data.shape
    diff_lists = []
    move_lists = []
    # 循环遍历每个目标的每个关键点
    # nonzero_indices = np.nonzero(data)
    # targets_nonzero_indices = list(zip(*[index.flatten() for index in nonzero_indices]))
    for target in range(data.shape[1]):
        target_diffs = []  # 存储当前目标的差值列表
        target_move = []
        # 找到第一个坐标点不为0的帧的索引
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        # inver_data = data[::-1]
        # start_frame = np.argmax(np.any(data[:, target, :, :] != 0, axis=(1, 2)))
        # start_frame = np.argmax(np.any(data != 0,axis=-1))
        # end_fram = 45 - np.argmax(np.any(inver_data[:, target, :, :] != 0,axis=(1, 2)))
        # 循环遍历每个关键点
        for keypoint in range(data.shape[2]):
            # 计算当前关键点在相邻两帧之间的差值，从第一个非零坐标点的帧开始计算
            # keypoint_diff = np.diff(data[start_frame:end_fram, target, keypoint, :], axis=0)
            aa = data[nonzero_frame_indice, target, keypoint, :]
            max_x = np.max(data[nonzero_frame_indice, target, keypoint, 0])
            min_x = np.min(data[nonzero_frame_indice, target, keypoint, 0])
            keypoint_diff_move = np.diff(data[nonzero_frame_indice, target, keypoint, :], axis=0)
            keypoint_diff = max_x - min_x
            keypoint_diff_move = np.sum(np.sqrt(np.sum(keypoint_diff_move ** 2, axis=1)))
            target_diffs.append(keypoint_diff)  # 将差值转换为列表并添加到当前关键点的差值列表中,不能用方差，用移动累加值
            target_move.append(keypoint_diff_move)
        diff_lists.append(target_diffs)  # 将当前目标的差值列表添加到总列表中
        move_lists.append(target_move)

    return data

def _valid_stand_data(data):
    frames, targets, keypoints, coordinate = data.shape
    target_indices = []
    for target in range(targets):
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        target_y = np.diff(data[nonzero_frame_indice, target, 5, :], axis=0)  # 0号点y轴坐标依次做差
        neg_indices = np.where(target_y[:, 1])[0].tolist()
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
    valid_data = data[:, target_indices, :, :]

    return valid_data

def _valid_sit_data(data):
    frames, targets, keypoints, coordinate = data.shape
    target_indices = []
    for target in range(targets):
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        target_y = np.diff(data[nonzero_frame_indice, target, 5, :], axis=0)  # 0号点y轴坐标依次做差
        positive_indices = np.where(target_y[:, 1] > 0)[0].tolist()
        neck_point_y = data[nonzero_frame_indice, target, 5, 1]
        spine_point_y = data[nonzero_frame_indice, target, 6, 1]
        mean_shoulder_distance = np.sum(spine_point_y - neck_point_y) / len(nonzero_frame_indice)
        max_neck_y = max(data[nonzero_frame_indice, target, 5, 1])
        min_neck_y = min(data[nonzero_frame_indice, target, 5, 1])
        move_neck = max_neck_y - min_neck_y
        if len(positive_indices) > 2:
            for i in range(len(positive_indices) - 1):
                if positive_indices[i + 1] - positive_indices[i] == 1 and move_neck > mean_shoulder_distance:
                    target_indices.append(target)
                    break
    valid_data = data[:, target_indices, :, :]
    return valid_data

def _valid_raise_data(data):

    frames, targets, keypoints, coordinate = data.shape
    target_indices = []
    for target in range(targets):
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        right_wrist_min = min(data[nonzero_frame_indice,target,4,1])
        right_wrist_min_index = np.argmin(data[nonzero_frame_indice,target,4,1])
        right_shoulder = data[right_wrist_min_index,target,2,1]
        left_wrist_min = min(data[nonzero_frame_indice,target,9,1])
        left_wrist_min_index = np.argmin(data[nonzero_frame_indice,target,9,1])
        left_shoulder = data[left_wrist_min_index,target,7,1]

        if right_shoulder >= right_wrist_min or left_shoulder >= left_wrist_min:
            target_indices.append(target)

    valid_data = data[:, target_indices, :, :]

    return valid_data

if __name__ == '__main__':
    lenTime = 0.0
    dataDict=dict()
    dataDict['normal']=[]
    for file in listFiles:
        print(f"Deal with {file}:")
        series = np.load('./npy/' + file + '_series.npy')  # timeStepLength, numStudents, 10, 2
        lenTotal=series.shape[0]
        sampleDict,lenTimeTemp,startLabel,endLabel = retrieveLabel(file, series.shape[1], FPS_extractedVideo)
        # print(sampleDict)
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
            # print(key)
            sample=oneSample(lenTotal,sampleDict[key],series[:,[key],:,:],startBias)
            dataTemp = sample.getData()
            dataDict['normal'] += sample.getNormalData()
            for label in dataTemp.keys():
                if label not in dataDict:
                    dataDict[label]=[]
                dataDict[label]+=dataTemp[label]
    print("\nSaving...")
    for key in tqdm.tqdm(dataDict.keys()):
        data = np.concatenate(dataDict[key], axis=1)
        # passon 对传递训练数据的处理
        # if key != 'normal':
        #     data = _valid_passon_data(data)
        # peep 对旁窥训练数据处理
        # if key != 'normal':
        #     data = _valid_peep_data(data)
        # stand 对站立训练数据处理
        if key == 'stand':
            data = _valid_stand_data(data)
        if key == 'sit':
            data = _valid_sit_data(data)
        if key == 'raise' or key == 'raise_up':
            data = _valid_raise_data(data)
        if key == 'normal':
            continue
        # # 归一化
        ## xyMinT: 所有帧每一个目标，分别取X_min，Y_min,
        ## xyMaxT: 所有帧每一个目标，分别取X_max，Y_max,
        ## xyLen: 所有帧每一个目标，x方向最大和最小的差值以及y方向上最大和最小的差值
        ## xyMax: 每个目标，在所有帧（45）中x方向最大和最小的差值以及y方向上最大和最小的差值，取最大的一帧，不同的目标，得到的最大值不一定是同一帧的，大概率属于不同的帧
        ## padT： 每个目标，最大的一帧xy方向的差值，分别与45帧，每个目标最大xy方向差值作差除以2，为每个目标的坐标点作padding
        # # 所有帧每一个目标，分别取X_min，Y_min, X_max, Y_max 保存成xyMinT，xyMaxT
        # # 相减之后得到，所有帧，每一个目标，x方向最大和最小的差值以及y方向上最大和最小的差值
        # xyMinT = np.min(data, axis=2, keepdims=True)
        # xyLenT = np.max(data, axis=2, keepdims=True) - xyMinT
        # # (timeStepLength, numStudents, 1, 2)
        # ## 每个目标在45帧中的最大差值
        # xyMax = np.max(xyLenT, axis=0, keepdims=True)  # (1, numStudents, 1, 2)
        # padT = (xyMax - xyLenT) * 0.5  # (timeStepLength, numStudents, 1, 2)
        # data = (padT + data - xyMinT) / (xyMax + 1e-6)
        # Save
        # np.save('./out/' + key + '.npy', data.transpose(1, 3, 0, 2))  # shape: N,C,T,V
        np.save('./out/' + key + '.npy', data)
    print('\nAll Done.')
    print(f'Max time len: {lenTime}s.')