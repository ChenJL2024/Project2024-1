import torch


def _multi_input(data, joint, velocity, bone, timeStepLength,connect_joint):
    for i in range(10):
        joint[:, 2:, :, i, :] = data[:, :, :, i, :] - data[:, :, :, 5, :]
    for i in range(timeStepLength-2):
        velocity[:, :2, i,:,:] = data[:,:,i+1,:,:] - data[:,:,i,:,:]
        velocity[:,2:,i,:,:] = data[:,:,i+2,:,:] - data[:,:,i,:,:]
    for i in range(len(connect_joint)):
        bone[:,:2,:,i,:] = data[:,:,:,i,:] - data[:,:,:,connect_joint[i],:]
    bone_length = 0
    for i in range(2):
        bone_length += bone[:,i,:,:,:] ** 2
    bone_length = torch.sqrt(bone_length) + 0.0001
    for i in range(2):
        bone[:,2+i,:,:,:] = torch.arccos(bone[:,i,:,:,:] / bone_length)
    return joint, velocity, bone

def L2Cache_up(config, queueL3NewData, pinNewData, queueRawData):
    numThreshSereisSamples=config.numThreshSereisSamples
    connect_joint = torch.tensor([1, 5, 5, 2, 3, 5, 5, 5, 7, 8])

    hashKeys=[]
    tLast=[]
    dataCache=[]
    frames=[]
    numTotalSumples=0
    while True:
        protcl,contents = queueL3NewData.get()
        if protcl==0:
            numSamples = len(contents[1])
            numTotalSumples += numSamples
            tLast.append(contents[1])
            hashKeys+=[contents[0] for _ in range(numSamples)]
            frames+=[contents[2] for _ in range(numSamples)]
            dataAllStus=torch.unsqueeze(contents[3],4) #shape：(timeStep, numStus, 10, 2, 1)
            dataAllStus=dataAllStus.permute(1,3,0,2,4) #shape：(samples,coord,frame,joint,num_person)
            joint = torch.zeros((numSamples, 4, timeStepLength, 10, 1))
            velocity = torch.zeros((numSamples, 4, timeStepLength, 10, 1))
            bone = torch.zeros((numSamples, 4, timeStepLength, 10, 1))
            joint[:, :2, :, :, :] = dataAllStus
            joint, velocity, bone = _multi_input(dataAllStus, joint, velocity, bone, timeStepLength, connect_joint)
            data_new = torch.stack([joint, velocity, bone], axis=1)
            #shape：(samples,3(branch),coord,frame,joint,num_person)
            dataCache.append(data_new)

        if numTotalSumples >= numThreshSereisSamples or sum(i<0 for i in frames):
            #待解决：即使在Linux下，也经常出现还有任务，但这里没有弹出数据的情况。最终程序阻塞，而第二个模型始终没运行。
            #最终也没有将所有任务的终止信息显示出来。
            queueRawData.put((1, None))

            tLast=torch.cat(tLast,axis=0)
            dataCache=torch.cat(dataCache,axis=0)
            iStart=0
            iEnd=batchSize
            while numTotalSumples>0:
                pinNewData.send((0,(hashKeys[iStart:iEnd],tLast[iStart:iEnd],\
                                    frames[iStart:iEnd],dataCache[iStart:iEnd])))
                # 这里不要装载到GPU，否则在用Pipe运送时会出现全部置0的情况。
                iStart=iEnd
                iEnd+=batchSize
                numTotalSumples -= batchSize
            pinNewData.send((1, None))
            hashKeys = []
            tLast = []
            dataCache = []
            frames = []
            numTotalSumples = 0
    return