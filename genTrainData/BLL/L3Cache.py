import os
import time

import torch
import cv2
import numpy as np
import shutil
import torch.backends.cudnn as cudnn
from kernel.Pose.utils.datasets import LoadStreams, LoadImages
from kernel.Pose.utils.general import non_max_suppression_v8,scale_coords,check_imshow
from kernel.Pose_v7.utils.plots import plot_one_box
import matplotlib.pyplot as plt
# from kernel.Pose.bytetrack_utils.byte_tracker import BYTETracker
from kernel.Pose.bytetrack_utils.byte_tracker_new import BYTETracker
from .smooth_keypoint import KeypointSmoothing


# lideping

def plot_2d_array(array, save_path=None):
    unique_values = np.unique(array)
    color_map = plt.cm.get_cmap('tab20', len(unique_values))  # Choose a colormap with enough colors for unique values

    rows, cols = array.shape

    fig, ax = plt.subplots()
    ax.imshow(array, cmap=color_map)

    # Add colorbar to show the color mapping for unique values
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=color_map), ax=ax, ticks=unique_values)
    cbar.set_label('Unique Values')

    # Add text annotations to display the actual values in the array
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(int(array[i, j]*100)), ha='center', va='center', color='black')

    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('2D Array Visualization')

    # Save the plot to a file (if save_path is provided)
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the figure to release resources and avoid displaying the plot on screen

    plt.show()

# lideping
# 跟踪主函数
def track_main(tracker, detection_results,Result_kpts, frame_id, image_height, image_width, test_size):
    '''
    main function for tracking
    :param args: the input arguments, mainly about track_thresh, track_buffer, match_thresh
    :param detection_results: the detection bounds results, a list of [x1, y1, x2, y2, score]
    :param frame_id: the current frame id
    :param image_height: the height of the image
    :param image_width: the width of the image
    :param test_size: the size of the inference model
    '''
    online_targets = tracker.update(detection_results, Result_kpts, [image_height, image_width], test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    results = []
    aspect_ratio_thresh = 1.6  # +++++
    min_box_area = 10  # ++++
    online_keypoints = []

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        keypoints = target.kpts
        online_keypoints.append(keypoints)
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area or vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(target.score)
            # save results
            results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )

    return online_tlwhs, online_ids, online_keypoints

# lideping
## 跟踪后xywh——》xyxy
def tlwh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y

# lideping
# keypoints、bbox、id匹配
def keypoint_idmatch(kpts,bboxs,steps):
    tracked_dict = {}
    for bbox in bboxs:
        count = []
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        id = bbox[4]
        tracked_key = (id,x1,y1,x2,y2)
        new_kpts = kpts[:,2:]
        for row in new_kpts:
            n = 0
            for i in range(len(row)-steps):
                if row[i]>=x1 and row[i]<=x2 and row[i+1]>=y1 and row[i+1]<=y2:
                    n += 1
                i+=steps
            count.append(n)
        count_tensor = torch.tensor(count)
        max_value, max_index = torch.max(count_tensor, dim=0)
        tracked_dict[tracked_key] = kpts[max_index,:]
    return tracked_dict

def _arrangeStudents(series,stepNow,last,now):
    # series对应一路，也就是一个教室的信息。shape：(timeStep,numStus,10,2)
    # last对应一路, shape：(numStus,4) 最后一维依次是：左上角点x、y，右下角点x、y
    # now对应一路, shape: (numStudents,57) 注：numStudents是新时刻的，可能会与last中的不同，
    # 57：(cx，cy，w，h，置信度，识别框类别，关键点1_x1，关键点1_y1，关键点1_类别，关键点2_x2，关键点2_y2，关键点2_类别...)
    series[stepNow,:,:,:] = series[stepNow-1,:,:,:]
    numStudents=now.shape[0]
    if numStudents == 0:
        return series,last,[None,None]
    #halfW = torch.ceil(now[:, 2] / 2)
    #halfH = torch.ceil(now[:, 3] / 2)
    existed_last = []
    existed_now = []
    newed = []
    # now_id = now[:,:1].tolist()
    # last_id = last[:,:1].tolist()
    now_id = [now_element for now_sublist in now[:,:1].tolist() for now_element in now_sublist]
    last_id = [last_element for last_sublist in last[:,:1].tolist() for last_element in last_sublist]
    # for id_index,id in enumerate(now_id):
    #     if id in last_id:
    #         existed.append(id_index)
    #     else:
    #         newed.append(id_index)
    for id_index, id in enumerate(now_id):
        if id in last_id:
            existed_last.append(last_id.index(id))
            existed_now.append(now_id.index(id))
        else:
            newed.append(now_id.index(id))

    #出于性能的考虑，这里不考虑之前学生在新结果中被两个或两个以上学生对应的情况
    #最终对应于：新结果中的位于最后的那个学生（tensorA[[0,0,...],:]=tensorB[[0,1,...],:] A中的0最终对应的是B中的1）
    #如果处理上边的情况，必将会引入判断与循环。

    newIndices=[None,None]
    if len(existed_last) > 0:
        series[stepNow, existed_last, :, 0] = now[existed_now,:][:,[7, 10, 13, 16, 19, 22, 25, 28, 31, 34]] #当前帧存储的顺序和上一帧存储的顺序是一样的，所以直接使用existed
        series[stepNow, existed_last, :, 1] = now[existed_now,:][:,[8, 11, 14, 17, 20, 23, 26, 29, 32, 35]]
        last[existed_last,:] = now[existed_now,:5]
    if len(newed)>0:
        numStus = series.shape[1]
        newIndices[1]=numStus
        series = torch.cat([series, torch.zeros(series.shape[0], len(newed), 10, 2)], dim=1)
        series[stepNow, numStus:, :, 0] = now[newed, :][:,[7, 10, 13, 16, 19, 22, 25, 28, 31, 34]]
        series[stepNow, numStus:, :, 1] = now[newed, :][:,[8, 11, 14, 17, 20, 23, 26, 29, 32, 35]]
        last = torch.cat([last, torch.zeros(len(newed), 5)], dim=0)
        last[numStus:, :] = now[newed, :5].clone()
    return series,last,newIndices

def del_tensor_ele(arr,index_a,index_b):
    arr1 = arr[0:index_a]
    arr2 = arr[index_b:30]
    return torch.cat((arr1,arr2),dim = 0)

smooth_keypoint = {}
# def L3Cache(device, poutStep1, config, queuePush, queueResults):
def L3Cache(device, l3cachedata_queue, config, queueL3NewData,queueResults):
    global smooth_keypoint
    save_img = not config.sources.endswith('.txt')  # save inference images
    timeStepLength = config.timeStepCache
    tSeries = {}
    tLast={}
    tBoxHist={}
    videoWriters={}
    tracker={}


    numBoxSave=config.numBoxSave
    # lideping 初始化跟踪器 0.5  30  0.8

    while True:
        protcl,contents = l3cachedata_queue.get()
        if protcl==0:
            frame_id = 0
            hashKey = contents[0]
            pred = contents[1]  #一帧图片的检测结果
            frame = contents[2]
            print('-----------------------------------------------L3cache:',frame)
            img = contents[3]
            source = contents[4]
            # video_name = source.split('\\')[-1][:-4]
            # childvideo_dir = source.split('\\')[1]
            # os.makedirs(config.outputDir + 'Done/' + childvideo_dir, exist_ok=True)
            img0 = LetterBox((640, 640), auto=True, stride=32)(image=img)
            if frame < 0:
                if hashKey in tSeries:
                    # for iTime in range(numBoxSave):
                    # np.save(config.outputDir + source[17:-4] + f'_lastSeriesBox_{iTime}.npy', \
                    #         tBoxHist[hashKey][-1 - iTime])
                    npy_output_path = config.outputDir + source.split('/')[-2]
                    os.makedirs(npy_output_path, exist_ok = True)
                    np.save(npy_output_path + '/' + source.split('/')[-1].split('.')[0] + f'_series.npy', \
                            np.array(tSeries[hashKey][:(-frame), :, :, :]))  ## 保存的时候只保存到最后一帧视频部分的数据
                    videoWriters[hashKey].release()

                    # for iTime in range(numBoxSave):
                    #     shutil.move(config.outputDir + source[17:-4] + f'_lastSeriesBox_{iTime}.npy', \
                    #            config.outputDir + 'Done/' + source[17:-4] + f'_lastSeriesBox_{iTime}.npy')
                    # shutil.move(config.outputDir + source[17:-4] + f'_series.npy', \
                    #             config.outputDir + 'Done/' + source[17:-4] + f'_series.npy')
                    # shutil.move(config.outputDir + source[17:-4] + '.mp4', \
                    #             config.outputDir + 'Done/' + source[17:-4] + '.mp4')
                    # shutil.move(source, source[:17] + 'Done/' + source[17:-4] + '.mp4')
                    del tSeries[hashKey]
                    # del tLast[hashKey]
                    # del tBoxHist[hashKey]
                    del videoWriters[hashKey]
                    del tracker[hashKey]
                    del smooth_keypoint[hashKey]
                queueResults.put((1, set([hashKey])))
                continue

            Height = img.shape[0]
            Width = img.shape[1]
            #preds = torch.stack(preds, axis=0)
            # pred = pred.unsqueeze(0)
            # pred = non_max_suppression_v8(pred, config.conf_thres, config.iou_thres, \
            #                               agnostic=config.agnostic_nms, classes=config.classes, \
            #                               nc=2)[0]
            # 删除掉老师的信息，老师不参与行为分析的训练
            # pred = np.delete(pred.cpu(), np.where(pred.cpu()[:, 5] == 1), axis=0)
            # pred = pred.numpy()
            # pred = np.delete(pred, np.where(pred[:, 5] == 1), axis=0) 
            #57: (x0，y0，x1，y1，置信度，识别框类别，关键点1_x1，关键点1_y1，关键点1_置信度，关键点2_x2，关键点2_y2，关键点2_置信度...)
            if hashKey not in tSeries:
                # tLast[hashKey] = pred[:, :4].clone() #（x,y,w,h）or(x1,y1,x2,y2)
                # tBoxHist[hashKey]=[]
                # newIndices=[None,0]

                video_output_path = config.outputDir + source.split('/')[-2]
                os.makedirs(video_output_path, exist_ok=True)
                videoWriters[hashKey] = cv2.VideoWriter(video_output_path + '/' + source.split('/')[-1], \
                                                        cv2.VideoWriter_fourcc(*"mp4v"), 3, \
                                                        (img.shape[1], img.shape[0])) # mp4v 、HEVC
                print(config.outputDir + source[17:-4] + '.mp4')
                tracker[hashKey] = BYTETracker(config.track_thresh, config.track_buffer, config.match_thresh)  # track
                smooth_keypoint[hashKey] = {} # 每个关键点坐标平滑器字典对应1路视频文件，每路视频文件中的目标id个数对应每个关键点的平滑器
            # Process detections
            t0 = time.time()
            det = pred  # detections per image
            tensor_tracker_matrix = None
            if len(det):
                # det = torch.from_numpy(det)
                # Rescale boxes from img_size to im0 size
                #384,640
                scale_coords([img0.shape[0],img0.shape[1]], det[:, :4], img.shape, kpt_label=False)
                scale_coords([img0.shape[0],img0.shape[1]], det[:, 6:], img.shape, kpt_label=config.kpt_label, step=3)
                Results = []  #
                Results_kpts = []
                # Write results
                # for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                for det_index, (*xyxy, conf) in enumerate(det[:, :5]):
                    Results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])  #
                    #[[x1,y1,c1],[x2,y2,c2]...]
                    Results_kpts.append(pred[det_index, 5:36])
                    # Results_kpts.append(del_tensor_ele(det[det_index,6:],3,3).reshape(-1,3)) # 存储10个关键点
                track_t0 = time.time()
                online_tlwhs, online_ids, online_keypoints = track_main(tracker[hashKey], np.array(Results),Results_kpts, frame_id, img0.shape[0], img0.shape[1],(img0.shape[0], img0.shape[1]))
                track_t1 = time.time()
                track_elapsed_time = (track_t1-track_t0)*1000
                print(f'跟踪耗时{track_elapsed_time:.2f} ms')
                online_tlwhs = np.array(online_tlwhs).reshape(-1, 4)
                online_xyxys = tlwh2xyxy(online_tlwhs)
                outputs = np.concatenate((online_xyxys, np.array(online_ids).reshape(-1, 1)), axis=1)
                tracked_coordinate = [] #框的坐标，关键点的坐标
                # from draw_rectangle import draw_rectangle
                # desk_info = draw_rectangle('./457.json', img) # 将方框画到图片上，并获取方框id和方框右下角坐标点，方框id为key,右下角坐标点为值，值是个元组
                """
                重新分配考生id, 将课桌的id给考生
                分配标准：此视角下每一个课桌右下角距离考生中心点
                """
                for det_index_,(output, conf,keypoint) in enumerate(zip(outputs,det[:,4],online_keypoints)):
                    xyxy = output[0:4]
                    id = output[4]
                    # if id not in smooth_keypoint[hashKey]:
                    #     smooth_keypoint[hashKey][id] = KeypointSmoothing(1920, 1080, filter_type='OneEuro', beta=0.05)
                    # print(tracked_dict)
                    con_cla = []
                    if save_img or config.save_img or config.save_crop or config.view_img:
                        # label = f'{int(id)} {conf:.2f}'
                        label = f'{int(id)}'
                        cla = keypoint[0]
                        # kpts = value[2:]
                        # 找到关键点坐标为0的索引
                        kpts = online_keypoints[det_index_][1:]
                        # kpt_copy = kpts.copy()
                        # # smooth_kpts = smooth_keypoint[hashKey][id].smooth_process(kpt_copy.reshape(-1,3)).reshape(-1)
                        # # with open('kpts.txt','a') as kpts_file:
                        # #     kpts_file.write(str(kpts)+'\n')
                        # # with open('smooth_keypoint.txt','a') as smooth_kpts_:
                        # #     smooth_kpts_.write(str(smooth_kpts)+'\n')
                        # kpts_conf_less_than5 = kpts.copy() # 将置信度小于0.5的关键点置为0

                        # kpts_zeros = np.zeros(30, dtype = float)
                        # ## 找到当前目标关键点为0的关键点索引
                        # num_kpts = len(kpts_conf_less_than5) // 3
                        # nodes_zero = []
                        # for node in range(num_kpts):
                        #     conf = kpts_conf_less_than5[3 * node + 2]
                        #     if conf < 0.3: # 小于0.3，所有关键点置为0
                        #         # kpts_conf_less_than5[3 * node], kpts_conf_less_than5[3 * node + 1] = 0.0, 0.0
                        #         nodes_zero.append(node)
                        # if len(nodes_zero) > 0:
                        #     kpts_conf_less_than5 = kpts_zeros

                        # 每个文件一个平滑器字典，每个目标id分配一个平滑器，平滑器保留整个视频的时间
                        #todo 关键点坐标平滑

                        
                        id_xyxy = np.concatenate((np.array([id]), xyxy), axis=0).tolist()
                        con_cla.append(float(conf))
                        con_cla.append(float(cla))
                        plot_one_box(xyxy,cla, img, label=label, color=None,
                                     line_thickness=config.line_thickness, kpt_label=config.kpt_label, kpts=kpts, steps=3,
                                     orig_shape=img.shape[:2])
                        
                        one_student_info = id_xyxy + con_cla + kpts.tolist() # 如果考虑关键点置信度的话，将kpts改为 kpts_conf_less_than5
                        tracked_coordinate.append(one_student_info)
                tracked_matrix = np.array(np.reshape(tracked_coordinate,(-1,37)),dtype=np.float32)
                tensor_tracker_matrix = torch.tensor(tracked_matrix)

                        # cv2.imwrite('1.jpg',img)
            if tensor_tracker_matrix is None:
                continue
            if hashKey not in tSeries:
                numStudents = tensor_tracker_matrix.shape[0]  # 一帧图像中的学生个数
                if numStudents == 0:
                    continue
                series = torch.zeros(timeStepLength, numStudents, 10, 2)  # （60000，numStudents，10，2）10个关键点
                series[frame, :, :, 0] = tensor_tracker_matrix[:, [7, 10, 13, 16, 19, 22, 25, 28, 31, 34]]  ## 骨骼x坐标点
                series[frame, :, :, 1] = tensor_tracker_matrix[:, [8, 11, 14, 17, 20, 23, 26, 29, 32, 35]]  ## 骨骼y坐标点
                tSeries[hashKey] = series  # 字典：tSeries{hashKey：series}
                tLast[hashKey] = tensor_tracker_matrix[:,:5].clone()
                tBoxHist[hashKey] = []
            else:
                series, last, newIndices = _arrangeStudents(tSeries[hashKey], frame, tLast[hashKey], \
                                                            tensor_tracker_matrix)  ## 这里是新的series，应该是经过重叠面积处理后的，一个学生对应一个id
                tSeries[hashKey] = series
                tLast[hashKey] = last
            tBoxHist[hashKey].append(tLast)
            # img0 = LetterBox((640,640), auto=True, stride=32)(image=img)
            videoWriters[hashKey].write(img)
            t1 = time.time()
            elapsed_time = (t1-t0)*1000
            # print(f'L3Chache 耗时{elapsed_time:.2f}ms')
    return

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self,new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32, dw = 0, dh = 0, new_unpad = (640,640)):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.dw = dw
        self.dh = dh
        self.new_unpad = new_unpad

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        #
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # new_unpad = self.new_unpad
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            # dw = self.dw
            # dh = self.dh
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img
    