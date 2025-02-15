import numpy as np
from collections import deque
import os
import os.path as osp
import copy

from .kalman_filter import KalmanFilter
from kernel.Pose.bytetrack_utils import matching
from .basetrack import BaseTrack, TrackState

import torch
import time
import logging


logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)



# console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(f'mylog.log')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
# console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
logger.addHandler(file_handler)



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, kpts):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.kpts = kpts
        self.score = score
        self.tracklet_len = 0
        self.tracker = None ##BYTETracker的对象

        # keypoints list for use in Actions prediction.
        self.keypoints_list = deque(maxlen=30)

    def set_Byte_tracker(self,tracker): ## 给BYTETracker的对象赋值
        self.tracker = tracker

    def get_track_id(self):
        return self.tracker.next_id() #通过BYTETracker的对象调用next()方法 track_id + 1

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.get_track_id() # 获取目标的id
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.get_track_id() #获取目标的id
        self.score = new_track.score

    def update(self, new_track, Result_kpts, frame_id):  # ct+++Result_kpts
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.kpts = Result_kpts
        self.score = new_track.score

        self.keypoints_list.append(Result_kpts)  # ct++++

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

inner_bytrack_i = 0
inner_bytrack_time1 = 0
inner_bytrack_time2 = 0
inner_bytrack_time3 = 0
inner_bytrack_time4 = 0
inner_bytrack_time5 = 0
inner_bytrack_time6 = 0
inner_bytrack_time7 = 0
inner_bytrack_time8 = 0
inner_bytrack_time9 = 0
inner_bytrack_time10 = 0
inner_bytrack_time11 = 0
inner_bytrack_time12 = 0
inner_bytrack_time13 = 0
inner_bytrack_time14 = 0
inner_bytrack_time15 = 0
inner_bytrack_time16 = 0
inner_bytrack_time17 = 0
inner_bytrack_time18 = 0
inner_bytrack_time19 = 0
inner_bytrack_time20 = 0
inner_bytrack_time21 = 0
inner_bytrack_time22 = 0
inner_bytrack_time23 = 0
inner_bytrack_time24 = 0
inner_bytrack_time25 = 0
inner_bytrack_time26 = 0
inner_bytrack_time27 = 0
inner_bytrack_time28 = 0
inner_bytrack_time29 = 0
inner_bytrack_time30 = 0
inner_bytrack_time31 = 0
inner_bytrack_time32 = 0
inner_bytrack_time33 = 0
inner_bytrack_time34 = 0
inner_bytrack_time35 = 0
inner_bytrack_time36 = 0
inner_bytrack_time37 = 0
inner_bytrack_time38 = 0
inner_bytrack_time39 = 0
inner_bytrack_time40 = 0

class BYTETracker(object):
    def __init__(self, track_thresh, track_buffer, match_thresh, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        # self.args = args
        # self.det_thresh = args.track_thresh
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self._count = 0


    def next_id(self):
        self._count += 1
        return self._count

    def update(self, output_results, Result_kpts, img_info, img_size):   # ct++++
        logger.debug('=============================frame_id:=============================== '+str(self.frame_id))
        """"
        tlwh: 代表左上角坐标+宽高
        tlbr: 代表左上角坐标+右下角坐标
        xyah: 代表中心坐标+宽高比+高
        """
        global inner_bytrack_i
        global inner_bytrack_time1, inner_bytrack_time2, inner_bytrack_time3, inner_bytrack_time4
        global inner_bytrack_time5, inner_bytrack_time6, inner_bytrack_time7, inner_bytrack_time8
        global inner_bytrack_time9, inner_bytrack_time10, inner_bytrack_time11, inner_bytrack_time12
        global inner_bytrack_time13, inner_bytrack_time14, inner_bytrack_time15, inner_bytrack_time16
        global inner_bytrack_time17, inner_bytrack_time18, inner_bytrack_time19
        global inner_bytrack_time20, inner_bytrack_time21, inner_bytrack_time22
        global inner_bytrack_time23, inner_bytrack_time24, inner_bytrack_time25, inner_bytrack_time26
        global inner_bytrack_time27, inner_bytrack_time28, inner_bytrack_time29
        global inner_bytrack_time30, inner_bytrack_time31, inner_bytrack_time32
        global inner_bytrack_time33, inner_bytrack_time34, inner_bytrack_time35, inner_bytrack_time36
        global inner_bytrack_time37, inner_bytrack_time38, inner_bytrack_time39

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        
        t1 = time.time()
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        t2 = time.time()
        inner_bytrack_time1 += t2 - t1

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        
        t3 = time.time()
        inner_bytrack_time2 += t3 - t2
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        Result_kpts = torch.stack(Result_kpts, 0)  # 将list[tensor]，转为tensor[tesor]
        Result_kpts = Result_kpts.cpu().numpy()  # ct++++
        Result_kpts_first = Result_kpts[remain_inds]  # ct++++
        Result_kpts_second = Result_kpts[inds_second]  # ct++++
        t4 = time.time()
        inner_bytrack_time3 += t4 - t3
        logger.debug('============================Step1: 分配高分段和低分段信息============================')
        logger.debug('坐标信息：'+str(output_results))
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, kpts) for
                          (tlbr, s, kpts) in zip(dets, scores_keep, Result_kpts_first)]
        else:
            detections = []

        logger.debug('高置信度的detections：'+str(detections))
        t5 = time.time()
        inner_bytrack_time4 += t5 - t4
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        logger.debug('unconfirmed:'+str(unconfirmed))
        logger.debug('tracked_stracks:'+str(tracked_stracks))
        logger.debug('self.tracked_stracks:'+str(self.tracked_stracks))

        logger.debug('============================Step2:First association, with high score detection boxes============================')
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        logger.debug('self.lost_stracks:'+str(self.lost_stracks))
        logger.debug('strack_pool:'+str(strack_pool))
        t6 = time.time()
        inner_bytrack_time5 += t6 - t5
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        t7 = time.time()
        inner_bytrack_time6 += t7 - t6
        dists = matching.iou_distance(strack_pool, detections)
        logger.debug('matching.iou_distance:'+str(dists))
        t8 = time.time()
        inner_bytrack_time7 += t8 - t7
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        logger.debug('matching.linear_assignment(dists, thresh=0.8),matches:'+str(matches))
        logger.debug('matching.linear_assignment(dists, thresh=0.8),u_track:'+str(u_track))
        logger.debug('matching.linear_assignment(dists, thresh=0.8),u_detection:'+str(u_detection))
        """
        matches是配对了的，比如 [(1,2)] 就表示strack_pool中第1个【STrack对象】和detections中第2个检测框匹配上了。
        显然，u_track就是剩下的没匹配成功的【STrack对象】序号，比如[4]。
        u_detection 就是剩下的没匹配成功的detections中的检测框序号，比如[0]。

        """
        t9 = time.time()
        inner_bytrack_time8 += t9 - t8
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            det.set_Byte_tracker(self)
            if track.state == TrackState.Tracked:
                t10 = time.time()
                track.update(detections[idet], Result_kpts_first[idet], self.frame_id)  # ct++++Result_kpts_first[idet]
                t11 = time.time()
                inner_bytrack_time9 += t11 - t10
                activated_starcks.append(track)
            else:
                t12 = time.time()
                track.re_activate(det, self.frame_id, new_id=False)
                t13 = time.time()
                inner_bytrack_time10 += t13 - t12
                refind_stracks.append(track)
        logger.debug('activate_stracks:'+str(activated_starcks))
        logger.debug('refind_stracks:'+str(refind_stracks))
        logger.debug('lost_stracks:'+str(lost_stracks))
        logger.debug('removed_stracks:'+str(removed_stracks))
        logger.debug('============================Step 3: Second association, with low score detection boxes============================')
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            t14 = time.time()
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, kpts_second) for
                          (tlbr, s, kpts_second) in zip(dets_second, scores_second, Result_kpts_second)]
            t15 = time.time()
            inner_bytrack_time11 += t15 - t14
        else:
            detections_second = []
        logger.debug('低置信度的detections_second：'+str(detections_second))
        t16 = time.time()
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        logger.debug('r_tracked_stracks:'+str(r_tracked_stracks))
        logger.debug('strack_pool:'+str(strack_pool))
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        logger.debug('matching.iou_distance:'+str(dists))
        t17 = time.time()
        inner_bytrack_time12 += t17 - t16
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        logger.debug('matching.linear_assignment(dists, thresh=0.5),matches:'+str(matches))
        logger.debug('matching.linear_assignment(dists, thresh=0.5),u_track:'+str(u_track))
        logger.debug('matching.linear_assignment(dists, thresh=0.5),u_detection:'+str(u_detection_second))
        t18 = time.time()
        inner_bytrack_time13 += t18 - t17
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            det.set_Byte_tracker(self)
            if track.state == TrackState.Tracked:
                t19 = time.time()
                track.update(det, Result_kpts_second[idet], self.frame_id)  # ct++++Result_kpts_second[idet]
                t20 = time.time()
                inner_bytrack_time14 += t20 - t19
                activated_starcks.append(track)
            else:
                t21 = time.time()
                track.re_activate(det, self.frame_id, new_id=False)
                t22 = time.time()
                inner_bytrack_time15 += t22 - t21
                refind_stracks.append(track)
        
        t23 = time.time()
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        logger.debug('activate_stracks:'+str(activated_starcks))
        logger.debug('refind_stracks:'+str(refind_stracks))
        logger.debug('lost_stracks:'+str(lost_stracks))
        logger.debug('removed_stracks:'+str(removed_stracks))
        t24 = time.time()
        inner_bytrack_time16 += t24 - t23
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        logger.debug('==========Deal with unconfirmed tracks, usually tracks with only one beginning frame==========')
        detections = [detections[i] for i in u_detection]
        logger.debug('detection:'+str(detections))
        logger.debug('u_detection:'+str(u_detection))
        Result_kpts_first = [Result_kpts_first[i] for i in u_detection]  # ct++++++

        dists = matching.iou_distance(unconfirmed, detections)
        logger.debug('unconfirmed:'+str(unconfirmed))
        logger.debug('matching.iou_distance:'+str(dists))
        # center_dist = 
        t25 = time.time()
        inner_bytrack_time17 += t25 - t24
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        logger.debug('matching.linear_assignment(dists, thresh=0.7),matches:'+str(matches))
        logger.debug('matching.linear_assignment(dists, thresh=0.7),u_track:'+str(u_track))
        logger.debug('matching.linear_assignment(dists, thresh=0.7),u_detection:'+str(u_detection))
        t26 = time.time()
        inner_bytrack_time18 += t26 - t25
        for itracked, idet in matches:
            t27 = time.time()
            unconfirmed[itracked].update(detections[idet], Result_kpts_first[idet], self.frame_id)  # ct++++
            activated_starcks.append(unconfirmed[itracked])
            t28 = time.time()
            inner_bytrack_time19 += t28 - t27
        t29 = time.time()    
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        logger.debug('activate_stracks:'+str(activated_starcks))
        logger.debug('refind_stracks:'+str(refind_stracks))
        logger.debug('lost_stracks:'+str(lost_stracks))
        logger.debug('removed_stracks:'+str(removed_stracks))
        t30 = time.time()
        inner_bytrack_time20 += t30 - t29
        logger.debug('==================Step 4: Init new stracks==================')
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            logger.debug("u_detection:"+str(u_detection))
            logger.debug("detections:"+str(detections))
            logger.debug("track.score:"+str(detections[inew].score))
            track = detections[inew]
            track.set_Byte_tracker(self)
            if track.score < self.det_thresh:
                continue
            t31 = time.time()
            track.activate(self.kalman_filter, self.frame_id)
            t32 = time.time()
            inner_bytrack_time21 += t32 - t31
            activated_starcks.append(track)
        logger.debug('activate_stracks:'+str(activated_starcks))
        logger.debug('refind_stracks:'+str(refind_stracks))
        logger.debug('lost_stracks:'+str(lost_stracks))
        logger.debug('removed_stracks:'+str(removed_stracks))
        """ Step 5: Update state"""
        logger.debug('==================Step 5: Update state===================')
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        t33 = time.time()
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        t34 = time.time()
        inner_bytrack_time22 += t34 - t33
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        t35 = time.time()
        inner_bytrack_time23 += t35 - t34
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        t36 = time.time()
        inner_bytrack_time24 += t36 - t35
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        t37 = time.time()
        inner_bytrack_time25 += t37 - t36
        self.lost_stracks.extend(lost_stracks)
        t38 = time.time()
        inner_bytrack_time26 += t38 - t37
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        t39 = time.time()
        inner_bytrack_time27 += t39 - t38
        self.removed_stracks.extend(removed_stracks)
        t40 = time.time()
        inner_bytrack_time28 += t40 - t39
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        t41 = time.time()
        inner_bytrack_time29 += t41 - t40
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('activate_stracks:'+str(activated_starcks))
        logger.debug('refind_stracks:'+str(refind_stracks))
        logger.debug('lost_stracks:'+str(lost_stracks))
        logger.debug('removed_stracks:'+str(removed_stracks))

        logger.debug("self.tracked_stracks:"+str(self.tracked_stracks))
        logger.debug("self.lost_stracks:"+str(self.lost_stracks))
        logger.debug("self.removed_stracks:"+str(self.removed_stracks))
        logger.debug('output_stracks:'+str(output_stracks))

        inner_bytrack_i += 1
        if inner_bytrack_i == 48:
            # print('-----byte track:',  
            #         inner_bytrack_time1, 
            #         inner_bytrack_time2,
            #         inner_bytrack_time3, 
            #         inner_bytrack_time4,
            #         inner_bytrack_time5, 
            #         inner_bytrack_time6,
            #         inner_bytrack_time7, 
            #         inner_bytrack_time8,
            #         inner_bytrack_time9,
            #         inner_bytrack_time10,
            #         inner_bytrack_time11, 
            #         inner_bytrack_time12,
            #         inner_bytrack_time13, 
            #         inner_bytrack_time14,
            #         inner_bytrack_time15, 
            #         inner_bytrack_time16,
            #         inner_bytrack_time17, 
            #         inner_bytrack_time18,
            #         inner_bytrack_time19,
            #         inner_bytrack_time20,
            #         inner_bytrack_time21, 
            #         inner_bytrack_time22,
            #         inner_bytrack_time23, 
            #         inner_bytrack_time24,
            #         inner_bytrack_time25, 
            #         inner_bytrack_time26,
            #         inner_bytrack_time27, 
            #         inner_bytrack_time28,
            #         inner_bytrack_time29,
            #         inner_bytrack_time29
            #         )
            
            inner_bytrack_i = 0
            inner_bytrack_time1 = 0
            inner_bytrack_time2 = 0
            inner_bytrack_time3 = 0
            inner_bytrack_time4 = 0
            inner_bytrack_time5 = 0
            inner_bytrack_time6 = 0
            inner_bytrack_time7 = 0
            inner_bytrack_time8 = 0
            inner_bytrack_time9 = 0
            inner_bytrack_time10 = 0
            inner_bytrack_time11 = 0
            inner_bytrack_time12 = 0
            inner_bytrack_time13 = 0
            inner_bytrack_time14 = 0
            inner_bytrack_time15 = 0
            inner_bytrack_time16 = 0
            inner_bytrack_time17 = 0
            inner_bytrack_time18 = 0
            inner_bytrack_time19 = 0
            inner_bytrack_time20 = 0
            inner_bytrack_time21 = 0
            inner_bytrack_time22 = 0
            inner_bytrack_time23 = 0
            inner_bytrack_time24 = 0
            inner_bytrack_time25 = 0
            inner_bytrack_time26 = 0
            inner_bytrack_time27 = 0
            inner_bytrack_time28 = 0
            inner_bytrack_time29 = 0

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
