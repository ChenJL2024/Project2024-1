import jump
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def L2Cache_down(numL3CacheProc,taskManagedata_queue, L3CacheQueue_list):
    dictHash={}
    while True:
        protcl,contents=taskManagedata_queue.get()
        if protcl==1:
            for item in contents:
                if item in dictHash:
                    del dictHash[item]
        elif protcl==0:
            preds=contents[1]
            hashKeys = contents[0]
            frames=contents[2]
            imgs = contents[3]
            sources= contents[4]

        


            # print(type(preds),type(hashKeys),type(frames),type(imgs),type(sources), img_)
            del contents
            for i in range(len(hashKeys)):
                # print('=======================',len(hashKeys))
                hashKey=hashKeys[i]
                if hashKey not in dictHash:
                    idCache = jump.hash(hashKey, numL3CacheProc)
                    dictHash[hashKey] = idCache
                    print('idCache:',hashKey,idCache)
                # print(hashKey,preds[i].shape,frames[i],imgs[i].shape,sources[i])
                L3CacheQueue_list[dictHash[hashKey]].put((0,(hashKey,preds[i].to('cpu'),frames[i],imgs[i],sources[i])))
    return

