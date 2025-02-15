import argparse
import os
import torch
from BLL.mainStream import main
import multiprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./kernel/Pose/parameters/pre_model_v8.pt', help='model.pt path(s)')
    parser.add_argument('--sources', type=str, default='/home/goldsun/data/Data/1103_后面有测试数据/right_passon_test/', help='sources')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default= False,action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave',action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', default=True,action='store_true', help='use keypoint labels')
    parser.add_argument('--model-stride-max', default=64, type=int, help='int(model.stride.max())')
    parser.add_argument('--maxsizeQueue', default=16, type=int, help='maxsizeQueue')
    parser.add_argument('--batchSize-Pose', default=48, type=int, help='batchSize-Pose')
    parser.add_argument('--numSourceProc', default=3, type=int, help='numSourceProc')
    parser.add_argument('--maxPerSourceProcImgCache', default=1, type=int, help='maxPerSourceProcImgCache')
    parser.add_argument('--numL3CacheProc', default=3, type=int, help='numL3CacheProc')
    parser.add_argument('--numThreshSereisSamples', default=1, type=int, help='numThreshSereisSamples')
    parser.add_argument('--numPostProc', default=3, type=int, help='numPostProc')
    parser.add_argument('--numConsiderAnormal', default=5, type=int, help='numConsiderAnormal')
    parser.add_argument('--threshConfidAnormal', type=float, default=0.00001, help='threshConfidAnormal')

    parser.add_argument('--outputDir', type=str, default='./out/', help='outputDir')
    parser.add_argument('--numBoxSave', default=1, type=int, help='numBoxSave(Last)')
    parser.add_argument('--timeStepCache', default=60000, type=int, help='timeStepCache')

    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")  # track
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks, usually as same with FPS")  # track
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")  # track
    config = parser.parse_args()

    multiprocessing.set_start_method('spawn')
    with torch.no_grad():
       main(config)

