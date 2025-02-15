import torch
import math

def postProc(config, pipe, queueResults):
    indicesAnormal = set([i for i in range(48)]) # EGCN网络输出中被考虑为作弊的标签序号
    numConsiderAnormal = config.numConsiderAnormal
    threshConfidAnormal = config.threshConfidAnormal

    hashKeys = []
    coordCache = []
    frames=[]
    outs = []
    toBeDel=set()
    resultAnormal={}
    numStusAnormal = 0
    while True:
        protcl, contents = pipe.recv()
        if protcl == 0:
            hashKeys += contents[0]
            coordCache += contents[1]
            frames += contents[2]
            outs.append(contents[3])
        elif protcl == 1:
            if outs:
                outs = torch.cat(outs, dim=0)
                _, indices = torch.sort(outs, axis=1, descending=True)
                numStus=len(hashKeys)
                for i in range(numStus):
                    indexAnormal = list(
                                   set(torch.where(indices[i] < numConsiderAnormal)[0].tolist()) & \
                                   set(torch.where(outs[i] > threshConfidAnormal)[0].tolist()) & \
                                   indicesAnormal)
                    numAnormal=len(indexAnormal)
                    if numAnormal>0:
                        numStusAnormal+=1
                        coord = coordCache[i].tolist()
                        dictCoord={"x0":int(coord[0]),"y0":int(coord[1]),\
                                   "x1":math.ceil(coord[2]),"y1":math.ceil(coord[3])}
                        infoAnormalStu={"Coord":dictCoord,"Anormal Info":{"num":numAnormal}}
                        for j in range(numAnormal):
                            infoAnormalStu["Anormal Info"][str(j)]={"Class":indexAnormal[j],\
                                                                    "Confid":outs[i][indexAnormal[j]].item()}
                        hashKey = hashKeys[i]
                        frame = frames[i]
                        if frame < 0:
                            toBeDel.add(hashKey)
                            frame = -frame
                        if hashKey not in resultAnormal:
                            timeSlot={}
                            timeSlot["End/s"] = math.ceil(frame/5)
                            remainder = frame % 300
                            if remainder == 0:
                                timeSlot["Start/s"]=(int(frame/300)-1) * 60
                            else:
                                timeSlot["Start/s"]=(int(frame/300)) * 60
                            resultAnormal[hashKey] = {"Time Slot":timeSlot,\
                                                      "num":0,\
                                                      "Anormal Students":[]}
                        resultAnormal[hashKey]["Anormal Students"].append(infoAnormalStu)
                        resultAnormal[hashKey]["num"] = numStusAnormal
            queueResults.put((0,resultAnormal))
            queueResults.put((1,toBeDel))
            hashKeys = []
            coordCache = []
            frames = []
            outs = []
            toBeDel = set()
            resultAnormal = {}
            numStusAnormal = 0
    return



    '''
            indices, _ = torch.where(indices[:, indicesAnormal] < numConsiderAnormal)
            indices = torch.unique(indices)
            recordsAnormal.append(indices)

        if flag == 1:
            data = data.to('cpu')
            _, indices = torch.sort(data, axis=1, descending=True)
            indices, _ = torch.where(indices[:, indicesAnormal] < numConsiderAnormal)
            indices = torch.unique(indices)
            recordsAnormal.append(indices)
        elif flag == 0:
            records=torch.tensor(data[0],dtype=int)
            coordCache = data[1]
            indicesRecord = data[2]

            resultAnormal= {}
            for i in range(len(recordsAnormal)):
                indexOfCaches=records[i*lengthPerIter+startInPerIter+recordsAnormal[i]]
                for j in range(len(indexOfCaches)):
                    indexOfCache = indexOfCaches[j]
                    indexInCache = len(torch.where(
                        records[:(i * lengthPerIter + startInPerIter + recordsAnormal[i][j] +1)]\
                        ==indexOfCache)[0])
                    iBatch, iRoom, iStudents = indicesRecord[indexOfCache.item()][indexInCache-1]
                    indexSource = iBatch * sizeBatch + iRoom
                    if indexSource not in resultAnormal:
                        resultAnormal[indexSource]=[\
                            coordCache[indexOfCache.item()][iBatch][iRoom][iStudents].tolist()]
                    else:
                        resultAnormal[indexSource].append(
                            coordCache[indexOfCache.item()][iBatch][iRoom][iStudents].tolist())
            print('Detect Anormal Behavior:')
            print(resultAnormal)
            '''