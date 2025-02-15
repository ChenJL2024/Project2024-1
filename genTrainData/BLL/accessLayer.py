import datetime

from kernel.Pose.utils.general import check_img_size
from kernel.Pose.utils.datasets import LoadStreams, LoadImages
import torch.backends.cudnn as cudnn
import torch
import time
import cv2

def _initial(stride,source,imgsz):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    # (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if isinstance(imgsz, (list, tuple)):
        assert len(imgsz) == 2;
        "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        dataset.count = -1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        dataset.count = 0
    return (dataset,source) # source : content[1]——》file


def _ejectData(device,half,numSources,datasets,listHashKey,queueRawData,queueResults):
    toBeDel = []
    for i in range(numSources):
        t1 = time.perf_counter()
        # path, img, img0, self.cap, self.frame
        flag, img, img0, _, frame = next(datasets[i][0])
        t2 = time.perf_counter()
        elapsed_time = (t2-t1)*1000
        # print(f'获得一个图片的时间:{elapsed_time:.2f} ms')
        # cv2.imwrite('accesslayer_img0.png',img0)
        if flag == -1:
            toBeDel.append(i)
            if frame == 0:
                print('无法读取文件.')
                queueResults.put((1, set([listHashKey[i]])))
                continue
            else:
                frame = -frame
        imgForModel = torch.from_numpy(img)#.to(device)
        #待解决：确定img的shape，判断flag为-1时添加的img是否一致。
        # imgForModel = torch.from_numpy(img)
        # imgForModel = imgForModel.half() if half else imgForModel.float()  # uint8 to fp16/32
        imgForModel = imgForModel.float()
        imgForModel /= 255  # 0 - 255 to 0.0 - 1.0
        # if imgForModel.ndimension() == 3:
        #     imgForModel = imgForModel.unsqueeze(0)
        # print('listHashKey[i]:',listHashKey[i])
        queueRawData.put((0,(listHashKey[i], imgForModel, frame,\
                          img0,datasets[i][1])))
        if frame<0:
            queueRawData.put((1,None))


    if toBeDel:
        newIndex= set(i for i in range(numSources)) - set(toBeDel)
        datasets = [datasets[i] for i in newIndex]
        listHashKey = [listHashKey[i] for i in newIndex]
        numSources -= len(toBeDel)
    return numSources,datasets,listHashKey

def sourceProc(config, device, sourceprocdata_queue, queueRawData,queueResults,ioFlag,genSourceFlag):
    stride = config.model_stride_max
    maxPerSourceProcImgCache = config.maxPerSourceProcImgCache
    imgsz, save_txt_tidl = config.img_size, config.save_txt_tidl
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    listHashKey=[]
    datasets=[]
    numSources=0
    continueFlag=False
    while True:
        protcl,contents=sourceprocdata_queue.get()  # content:hashkey和file
        if protcl==-1:
            break
        elif protcl==0:
            datasets.append(_initial(stride, contents[1], imgsz))  # content[1]:file, imgz = config.img_size
            listHashKey.append(contents[0])
            numSources+=1
            with ioFlag.get_lock():
                ioFlag.value -= 1
            if ioFlag.value>0:
                continue
        elif protcl==1:
            continueFlag = True
        
        if continueFlag:
            iFrame=0
            while numSources>0:

                numSources,datasets,listHashKey = \
                    _ejectData(device, half, numSources, datasets, listHashKey, queueRawData, queueResults)
                iFrame+=1
                if iFrame==maxPerSourceProcImgCache: #如下两个if的顺序不可更改！
                    if genSourceFlag.value==0:
                        continueFlag = False
                        break
                    if ioFlag.value>0:
                        break
                    iFrame=0
    return