import torch
import datetime
import time
#import multiprocessing
import torch.multiprocessing as multiprocessing
from ultralytics import YOLO
import numpy as np
import cv2
from kernel.Pose.utils.general import non_max_suppression_v8
import queue

def _startUpProc(config,device):
    # multiprocessing.set_start_method('spawn')
    from BLL.accessLayer import sourceProc
    queueRawData = multiprocessing.Queue(config.maxsizeQueue)
    queueResults = multiprocessing.Queue(config.maxsizeQueue)
    sourceP = []
    updateCmdPipeIn_list = []
    ioFlagList=[]
    ctx=multiprocessing.get_context('spawn')
    # genSourceFlag为0代表不需要sourceProc再产生数据了，直到主程序再发指令。
    genSourceFlag = ctx.Value("i", 1)
    for iProc in range(config.numSourceProc):
        sourceprocdata_queue = multiprocessing.Queue(config.maxsizeQueue)
        # pout, pin = multiprocessing.Pipe(duplex=False)
        updateCmdPipeIn_list.append(sourceprocdata_queue)
        ioFlag = ctx.Value("i", 0)
        # ioFlag为正数时代表需要sourceProc更新的新源请求数量（Pipe中的个数）；
        ioFlagList.append(ioFlag)
        temp = multiprocessing.Process(target=sourceProc, args=(config, device, sourceprocdata_queue, queueRawData,\
                                                                queueResults, ioFlag,genSourceFlag,))
        temp.start()
        time.sleep(0.3) #解决Linux系统上的问题，同时启动会出错
        sourceP.append(temp)

    from BLL.taskManage import taskManage
    # pout_L2Cache, pin_L2Cache = multiprocessing.Pipe(duplex=False)
    taskManagedata_queue = multiprocessing.Queue(config.maxsizeQueue)
    tk = multiprocessing.Process(target=taskManage, args=(config,updateCmdPipeIn_list,\
                                                          queueResults,taskManagedata_queue, ioFlagList))
    tk.start()
    time.sleep(0.3)

    from BLL.L3Cache import L3Cache
    queueL3NewData = multiprocessing.Queue()
    L3CacheP = []
    L3CacheQueue_list = []
    for iProc in range(config.numL3CacheProc):
        l3cachedata_queue = multiprocessing.Queue(config.maxsizeQueue)
        # pout, pin = multiprocessing.Pipe(duplex=False)
        L3CacheQueue_list.append(l3cachedata_queue)
        temp = multiprocessing.Process(target=L3Cache, args=(device, l3cachedata_queue, config, queueL3NewData,queueResults))
        temp.start()
        time.sleep(0.3)
        L3CacheP.append(temp)

    from BLL.L2Cache_down import L2Cache_down
    L2Cache_down_p = multiprocessing.Process(target=L2Cache_down, args=(config.numL3CacheProc, \
                                                                taskManagedata_queue, L3CacheQueue_list))
    L2Cache_down_p.start()
    time.sleep(0.3)
    return updateCmdPipeIn_list, queueRawData, queueL3NewData, queueResults, L3CacheQueue_list,\
           taskManagedata_queue, ioFlagList, genSourceFlag

def _stratUpModel(device,config):
    poseExtract = YOLO(config.weights)
    # from kernel.Pose.models.experimental import attempt_load
    # poseExtract = attempt_load(config.weights, map_location=device)  # load FP32 model
    #poseExtract.half()  # to FP16
    # if device.type != 'cpu':
    #     poseExtract(torch.zeros(1, 3, config.img_size, config.img_size).to(device).type_as(
    #         next(poseExtract.parameters())))  # run once
    del poseExtract.ckpt['date']
    del poseExtract.ckpt['version']
    del poseExtract.ckpt['license']
    del poseExtract.ckpt['docs']
    return poseExtract

def _runPoseExtract(device,config,updateCmdPipeIn_list,queueRawData,\
                    poseExtract,taskManagedata_queue, genSourceFlag, t0):
    batchSize = config.batchSize_Pose
    numSourceProc = config.numSourceProc

    sizeNow=0
    hashKeys=[]
    frames=[]
    imgs=[]
    imgsRaw=[]
    sources=[]
    with genSourceFlag.get_lock():
        genSourceFlag.value = 1
    # time.sleep(15)
    for iProc in range(numSourceProc):
        updateCmdPipeIn_list[iProc].put((1, None))
    timePrint=0
    while True:
        # 待解决，考虑到Pipe在运送GPU数据时，第一次会出现错误，所以考虑启动时设置小一点的时间窗口，对系统`预热`后再开始正式运行。
        protcl,contents = queueRawData.get()
        if protcl==1:
            '''
            with genSourceFlag.get_lock():
                genSourceFlag.value = 0
            '''
            if sizeNow>0:
                imgs = torch.stack(imgs, axis=0).to(device)
                #imgs = torch.stack(imgs, axis=0)
                pose_start = time.time()
                preds_pose_gpu = poseExtract.predict(imgs, task='pose', imgsz=(384,640),device = device)[0]
                preds_pose_gpu = non_max_suppression_v8(preds_pose_gpu, config.conf_thres, config.iou_thres, \
                                          agnostic=config.agnostic_nms, classes=config.classes, \
                                          nc=2)
                if len(preds_pose_gpu) == 0:
                    preds_pose_gpu = torch.zeros((1,36)).unsqueeze(0)
                pose_end = time.time()
                elapsed_time = (pose_end-pose_start)*1000
                # print(f'关键点检测耗时protcl=1:{elapsed_time:.2f} ms')
                #pin_L2Cache.send((0, (hashKeys, preds.to('cpu'))))
                taskManagedata_queue.put((0, (hashKeys, preds_pose_gpu, frames, imgsRaw,sources)))
                t1 = datetime.datetime.now()
                # print('{} timeConsuming of Step1-Pose: {:.2f}ms'.format(timePrint,0.001 * (t1 - t0).microseconds))
                t0 = t1
                timePrint+=1
                sizeNow = 0
                hashKeys = []
                frames = []
                imgs = []
                imgsRaw = []
                sources = []
            #break
        if protcl==0:
            hashKeys.append(contents[0])
            imgs.append(contents[1])
            frames.append(contents[2])
            imgsRaw.append(contents[3])
            sources.append(contents[4])
            sizeNow+=1
            if sizeNow==batchSize:
                imgs=torch.stack(imgs, axis=0).to(device)
                # imgs_ = torch.stack(imgs, axis=0)
                # preds = poseExtract(imgs.to(device), augment=augmentPoseNet)[0]
                pose_start = time.time()
                preds_pose_gpu = poseExtract.predict(imgs, task='pose', imgsz=(384,640),device = device)[0]
                # print(type(preds_pose_gpu))
                preds_pose_gpu = non_max_suppression_v8(preds_pose_gpu, config.conf_thres, config.iou_thres, \
                                          agnostic=config.agnostic_nms, classes=config.classes, \
                                          nc=2)
                # print(type(preds_pose_gpu))
                # preds_ultralytics = poseExtract.predict(imgsRaw, conf = 0.5)[0]
                # preds_box = preds_ultralytics.boxes.data
                # preds_keypoints = preds_ultralytics.keypoints.data.view(preds_ultralytics.keypoints.data.shape[0], (preds_ultralytics.keypoints.data.shape[1] * preds_ultralytics.keypoints.data.shape[2]))
                # print(preds_box,preds_keypoints)
                # print('框和关键点尺度：',preds_box.shape,preds_keypoints.shape, frames)
                if len(preds_pose_gpu) > 0:
                    # preds = torch.cat((preds_box,preds_keypoints), dim = 1)
                    # preds = preds.unsqueeze(0)
                    # # preds = preds.view(config.batchSize_Pose,preds.shape[0],preds.shape[1])
                    # pose_end = time.time()
                    # elapsed_time = (pose_end - pose_start) * 1000
                    # print(f'关键点检测耗时protcl=1:{elapsed_time:.2f} ms')
                    # #pin_L2Cache.send((0,(hashKeys,preds.to('cpu'))))
                    taskManagedata_queue.put((0, (hashKeys, preds_pose_gpu, frames, imgsRaw,sources)))
                    sizeNow = 0
                    hashKeys = []
                    frames=[]
                    imgs = []
                    imgsRaw = []
                    sources=[]
                    t1 = datetime.datetime.now()
                    # print('{} timeConsuming of Step1-Pose: {:.2f}ms'.format(timePrint,0.001 * (t1 - t0).microseconds))
                    t0 = t1
                    timePrint += 1
                else:
                    taskManagedata_queue.put((0, (hashKeys, torch.zeros((1,36)).unsqueeze(0), frames, imgsRaw,sources)))
                    sizeNow = 0
                    hashKeys = []
                    frames=[]
                    imgs = []
                    imgsRaw = []
                    sources=[]
    print('Step1-Pose suspended.\n')
    return t0


def main(config):
    from kernel.Pose.utils.torch_utils import select_device
    device = select_device(config.device)

    updateCmdPipeIn_list, queueRawData, queueL3NewData, queue_result, L3CacheQueue_list, \
        taskManagedata_queue,ioFlagList, genSourceFlag =_startUpProc(config,device)
    poseExtract = _stratUpModel(device,config)

    t0=datetime.datetime.now()
    print('\nStart detecting...')
    augmentPoseNet=config.augment
    time.sleep(1)
    while True:
        #显存资源是核心问题，因此不将两个模型独立在不同进程中，而是采取串行策略，以此保证显存安全性。
        #效率问题是通过增加多级缓存来尽量减少模型进程中的非张量操作。
        t0 = _runPoseExtract(device,config,updateCmdPipeIn_list,queueRawData,\
                    poseExtract,taskManagedata_queue,genSourceFlag, t0)
        #待解决，测试中有如下情况发生：在第三个时间窗口检测时，当poseExtract结束后，程序卡死了，内存占用达到100%。
    print('\nprocess exit')
