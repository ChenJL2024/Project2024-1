import numpy as np
import shutil
import math
import cv2
import os


FPS_extractedVideo = 3

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# def angle_between_points(a, b, c):

#     try:
#         # Calculate vectors AB and BC
#         distance_ab = calculate_distance(a, b)
#         distance_ac = calculate_distance(a, c)
#         distance_bc = calculate_distance(b, c)

#         # Calculate dot product of AB and BC
#         a_2 = distance_bc * distance_bc
#         c_2 = distance_ab * distance_ab
#         b_2 = distance_ac * distance_ac

#         # Calculate angle in radians
#         angle_radians = math.acos((a_2+c_2-b_2)/ (2*distance_bc*distance_ab))
#         # Convert angle to degrees
#         angle_degrees = math.degrees(angle_radians)
#     except:
#         angle_degrees = 60 # 返回不满足条件的角度

#     return angle_degrees

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
    frames, targets, num_points, _ = data.shape
    diff_lists = []
    move_lists = []
    # 循环遍历每个目标的每个关键点

    for target in range(targets):
        target_diffs = []  # 存储当前目标的差值列表
        target_move = []
        # 找到第一个坐标点不为0的帧的索引
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1) # 得到不为0的帧索引
        # 循环遍历每个关键点
        for keypoint in range(num_points):
            # 计算当前关键点在相邻两帧之间的差值，从第一个非零坐标点的帧开始计算
            max_x = np.max(data[nonzero_frame_indice, target, keypoint, 0])
            print(data[nonzero_frame_indice, target, keypoint, 0])
            max_x_index = np.argmax(data[nonzero_frame_indice, target, keypoint, 0])
            min_x = np.min(data[nonzero_frame_indice, target, keypoint, 0])
            min_x_index = np.argmin(data[nonzero_frame_indice, target, keypoint, 0])
            print(max_x_index, min_x_index)
            keypoint_diff_move = np.diff(data[nonzero_frame_indice, target, keypoint, :], axis=0)
            print(keypoint_diff_move)
            keypoint_diff = max_x - min_x
            keypoint_diff_move_sum = np.sum(np.sqrt(np.sum(keypoint_diff_move ** 2, axis=1)))
            print(keypoint_diff_move_sum)
            target_diffs.append(keypoint_diff)      # 在frames时间序列内，10个关键点在x轴方向上最大和最小值的差值
            target_move.append(keypoint_diff_move_sum)  # 在frames时间序列内，10个关键点运动距离之和
        diff_lists.append(target_diffs) 
        move_lists.append(target_move)

    all_targets_angle = {}
    all_targets_angle_ori = {}
    for target in range(targets):
        all_angles = []
        for frame in range(frames):
            if np.all(data[frame, target, 0, :] == data[frame, target, 6, :], axis=-1):
                arm_angles = [50, 50] # 如果当前帧元素都是0的话，左右胳膊角度设置为50
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
        max_index = diff_lists[target].index(max(diff_lists[target])) # 获取10个关键点中最大值的索引，看几号关键点运动距离最大
        max_move_index = move_lists[target].index(max(move_lists[target])) # 同上，

        # 理论上认为，发生传递动作，手部点位动作运动是最大的，默认传递时只有一个手动作较大
        # 右边胳膊（2，3，4），看4号点的移动幅度
        # if max_index == 4:
        if max_index == 4 and max_move_index == 4:
            right_indexes = list(np.where(abs(data[:, target, 4, 1] - data[:, target, 2, 1]) < abs(
                data[:, target, 4, 0] - data[:, target, 2, 0])))[0].tolist()
            if len(right_indexes) == 0:
                continue
            min_index_right = np.argmin(all_targets_angle[target][right_indexes, 0]) 
            min_value_right = all_targets_angle[target][right_indexes[min_index_right]] # 获取右手角度最小的值
            if min_value_right[0] < 25:
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
            if min_value_left[1] < 25:
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
    key_none = [i for i, j in all_targets_angle.items() if j is None]
    valid_data, invalid_data = None, None
    if len(key_not_none) > 0:
        valid_data = data[:, key_not_none, :, :]
    if len(key_none) > 0:
        invalid_data = data[:, key_none, :, :]
    return valid_data, invalid_data

def __check_pass_on(data):
    check_status = True
    frame_size, examinee_size, _, _ = data.shape
    for target in range(examinee_size):
        shoulder_width = 0
        check_distance = False

        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)

        coord_wrist_right_min = min(data[nonzero_frame_indice, target, 4, 0])
        coord_wrist_right_max = max(data[nonzero_frame_indice, target, 4, 0])
        # 右手腕移动距离以右肩部为边界, 若已经在右肩右侧则以右侧点计算
        move_distance_wrist_right = min(coord_wrist_right_max, np.mean(data[nonzero_frame_indice, target, 2, 0])) - coord_wrist_right_min

        coord_wrist_left_min = min(data[nonzero_frame_indice, target, 9, 0])
        coord_wrist_left_max = max(data[nonzero_frame_indice, target, 9, 0])
        # 左手腕移动距离计算方式参照右腕
        move_distance_wrist_left = coord_wrist_left_max - max(coord_wrist_left_min, np.mean(data[nonzero_frame_indice, target, 7, 0]))

        check_persistence_time_left = True
        check_persistence_time_right = True
        if move_distance_wrist_left > move_distance_wrist_right:
            if move_distance_wrist_left > shoulder_width:
                # 通过传递时手腕最远距离定住一定时间判断是否传递
                hit = 0
                history_hit_max = 0
                for frame_index in range(frame_size):
                    # print('left_history_hitdistance:', coord_wrist_left_max - data[frame_index, target, 9, :][0])
                    # todo:根据框位置及分辨率更新阈值
                    if coord_wrist_left_max - data[frame_index, target, 9, :][0] < 15:  # 达到传递位移最大处
                        # print('left_history_hit hit distance:',
                        #         coord_wrist_left_max - data[frame_index, target, 9, :][0])
                        hit += 1
                        if history_hit_max < hit:
                            history_hit_max = hit
                    else:
                        if history_hit_max < hit:
                            history_hit_max = hit
                        hit = 0
                # 排除手臂挥动动作与手臂长时间处于相同形状
                # print('left_history_hit_max:', history_hit_max)
                if 1 < history_hit_max < 8:
                    check_persistence_time_left = True
                else:
                    check_persistence_time_left = False
            else:
                check_persistence_time_left = False
        else:
            if move_distance_wrist_right > shoulder_width:
                # 通过传递时手腕最远距离定住一定时间判断是否传递
                hit = 0
                history_hit_max = 0
                for frame_index in range(frame_size):
                    # print('right_history_hit distance:', data[frame_index, target, 4, :][0] - coord_wrist_right_min)
                    # todo:根据框位置及分辨率更新阈值
                    if data[frame_index, target, 4, :][0] - coord_wrist_right_min < 15:  # 达到传递位移最大处
                        # print('right_history_hit hit distance:',
                        #         data[frame_index, target, 4, :][0] - coord_wrist_right_min)
                        hit += 1
                        if history_hit_max < hit:
                            history_hit_max = hit
                    else:
                        if history_hit_max < hit:
                            history_hit_max = hit
                        hit = 0
                # 排除手臂挥动动作与手臂长时间处于相同形状
                # print('right_history_hit_max:', history_hit_max)
                if 1 < history_hit_max < 8:
                    check_persistence_time_right = True
                else:
                    check_persistence_time_right = False
            else:
                check_persistence_time_right = False
        
        if check_persistence_time_left == False or check_persistence_time_right == False:
            return False

        condition_ok_statistic = 0
        for frame_index in range(frame_size):
            
            point_forehead = data[frame_index, target, 0, :]
            
            point_shoulder_right = data[frame_index, target, 2, :]
            point_elbow_right = data[frame_index, target, 3, :]
            point_wrist_right = data[frame_index, target, 4, :]

            point_shoulder_left = data[frame_index, target, 7, :]
            point_elbow_left = data[frame_index, target, 8, :]
            point_wrist_left = data[frame_index, target, 9, :]

            shoulder_width = abs(point_shoulder_left[0] - point_shoulder_right[0]) * 0.8
            arm_length_x_right = point_shoulder_right[0] - point_wrist_right[0]
            arm_length_x_left = point_wrist_left[0] - point_shoulder_left[0]

            arm_length_y_right = point_shoulder_right[1] - point_wrist_right[1]
            arm_length_y_left = point_wrist_left[1] - point_shoulder_left[1]

            if arm_length_x_right > arm_length_x_left:
                if arm_length_x_right > shoulder_width:  # and arm_length_x_right > arm_length_y_right:
                    angle_elbow_right = angle_between_points(point_shoulder_right,point_elbow_right,point_wrist_right)
                    # print('right_angle.', angle_elbow_right)
                    if angle_elbow_right < 25:
                        forearm_length_right = np.sqrt((point_wrist_right[0] - point_elbow_right[0])**2 + (point_wrist_right[1] - point_elbow_right[1])**2)
                        bigarm_length_right = np.sqrt((point_elbow_right[0] - point_shoulder_right[0])**2 + (point_elbow_right[1] - point_shoulder_right[1])**2)

                        if forearm_length_right >= bigarm_length_right*3/4:
                            hand_head_distance = np.linalg.norm(point_forehead-point_wrist_right)
                            if hand_head_distance > shoulder_width:
                                condition_ok_statistic += 1
                                if condition_ok_statistic > 1:
                                    # print('add check condition right OK')
                                    check_distance = True
                                    break
                        else:
                            condition_ok_statistic = 0
                    else:
                        condition_ok_statistic = 0
                else:
                    condition_ok_statistic = 0
            else:
                if arm_length_x_left > shoulder_width :  # and arm_length_x_left>arm_length_y_left:
                    angle_elbow_left = angle_between_points(point_shoulder_left, point_elbow_left, point_wrist_left)
                    # print('left_angle.', angle_elbow_left)
                    if angle_elbow_left < 25:
                        forearm_length_left = np.linalg.norm(point_wrist_left-point_elbow_left)
                        bigarm_length_left = np.linalg.norm(point_elbow_left-point_shoulder_left)

                        if forearm_length_left >= bigarm_length_left * 3 / 4:
                            
                            hand_head_distance = np.linalg.norm(point_wrist_left-point_forehead)
                            # print("hand_head_distance", hand_head_distance)
                            if hand_head_distance > shoulder_width:
                                condition_ok_statistic += 1
                                if condition_ok_statistic > 1:
                                    # print('add check condition left OK')
                                    check_distance = True
                                    break
                        else:
                            condition_ok_statistic = 0
                    else:
                        condition_ok_statistic = 0
                else:
                    condition_ok_statistic = 0

        if check_distance == False:
            check_status = False
    
    return check_status

def _valid_peep_data(data):

    frame_size, examinee_size, keypoints, coordinates = data.shape
    diff_lists = []
    move_lists = []
    target_indices = []
    for target in range(examinee_size):
        target_diffs = []
        target_move = []
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        for keypoint in range(keypoints):
            max_x = np.max(data[nonzero_frame_indice, target, keypoint, 0])
            min_x = np.min(data[nonzero_frame_indice, target, keypoint, 0])
            keypoint_diff_move = np.diff(data[nonzero_frame_indice, target, keypoint, :], axis=0)
            keypoint_diff = max_x - min_x
            keypoint_diff_move = np.sum(np.sqrt(np.sum(keypoint_diff_move ** 2, axis=1)))
            target_diffs.append(keypoint_diff)  # 将差值转换为列表并添加到当前关键点的差值列表中,不能用方差，用移动累加值
            target_move.append(keypoint_diff_move)
        diff_lists.append(target_diffs)  # 将当前目标的差值列表添加到总列表中
        move_lists.append(target_move)

        max_index = diff_lists[target].index(max(diff_lists[target]))
        max_move_index = move_lists[target].index(max(move_lists[target]))
        # 头部关键点
        if max_move_index == 0 or max_move_index == 1 or max_move_index == 2 or max_move_index == 7: 
            # if data[max_index,target,4,1] < data[max_index,target,0,1] + abs(data[max_index,target,2,1] - data[max_index,target,7,1])*0.1:
                target_indices.append(target)
    valid_data = data[:, target_indices, :, :]

    return valid_data

def _check_peep_back_data(data):
    check_status = False
    frame_size, examinee_size, keypoints, coordinates = data.shape
    diff_lists = []
    move_lists = []
    target_indices = []
    for target in range(examinee_size):
        target_diffs = []
        target_move = []
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        for keypoint in range(keypoints):
            max_x = np.max(data[nonzero_frame_indice, target, keypoint, 0])
            min_x = np.min(data[nonzero_frame_indice, target, keypoint, 0])
            keypoint_diff_move = np.diff(data[nonzero_frame_indice, target, keypoint, :], axis=0)
            keypoint_diff = max_x - min_x
            keypoint_diff_move = np.sum(np.sqrt(np.sum(keypoint_diff_move ** 2, axis=1)))
            target_diffs.append(keypoint_diff)  # 将差值转换为列表并添加到当前关键点的差值列表中,不能用方差，用移动累加值
            target_move.append(keypoint_diff_move)
        diff_lists.append(target_diffs)  # 将当前目标的差值列表添加到总列表中
        move_lists.append(target_move)

        max_index = diff_lists[target].index(max(diff_lists[target]))
        max_move_index = move_lists[target].index(max(move_lists[target]))
        # 头部关键点
        if max_move_index == 0 or max_move_index == 1 or max_move_index == 2 or max_move_index == 7: 
           check_status = True

    return check_status

def _valid_stand_data(data):
    target_indices = []
    frames, targets, keypoints, coordinate = data.shape
    for target in range(targets):

        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)

        shoulder_distance = np.mean(data[nonzero_frame_indice, target, 7, 0] - data[nonzero_frame_indice, target, 2, 0])
        
        coord_shoulder_right_min = min(data[nonzero_frame_indice, target, 2, 1])
        coord_shoulder_right_max = max(data[nonzero_frame_indice, target, 2, 1])

        coord_shoulder_left_min = min(data[nonzero_frame_indice, target, 7, 1])
        coord_shoulder_left_max = max(data[nonzero_frame_indice, target, 7, 1])

        right_shoulder_stand_distance = coord_shoulder_right_max - coord_shoulder_right_min
        left_shoulder_stand_distance = coord_shoulder_left_max - coord_shoulder_left_min

        if right_shoulder_stand_distance > shoulder_distance*2/3 and left_shoulder_stand_distance > shoulder_distance*2/3:
            target_indices.append(target)
    valid_data = data[:,target_indices,:,:]

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

    frame_size, examinee_size, keypoints, coordinates = data.shape
    diff_lists = []
    move_lists = []
    target_indices = []
    for target in range(examinee_size):
        target_diffs = []
        target_move = []
        nonzero_frame_indice = np.any(data[:, target, 0, :] != data[:, target, 6, :], axis=1)
        for keypoint in range(keypoints):
            max_y = np.max(data[nonzero_frame_indice, target, keypoint, 1])
            min_y = np.min(data[nonzero_frame_indice, target, keypoint, 1])
            keypoint_diff_move = np.diff(data[nonzero_frame_indice, target, keypoint, :], axis=0)
            keypoint_diff = max_y - min_y
            keypoint_diff_move = np.sum(np.sqrt(np.sum(keypoint_diff_move ** 2, axis=1)))
            target_diffs.append(keypoint_diff)  # 将差值转换为列表并添加到当前关键点的差值列表中,不能用方差，用移动累加值
            target_move.append(keypoint_diff_move)
        diff_lists.append(target_diffs)  # 将当前目标的差值列表添加到总列表中
        move_lists.append(target_move)

        max_index = diff_lists[target].index(max(diff_lists[target]))
        max_move_index = move_lists[target].index(max(move_lists[target]))
        # 右手
        if max_index == 4 and max_move_index == 4: 
            # if data[max_index,target,4,1] < data[max_index,target,0,1] + abs(data[max_index,target,2,1] - data[max_index,target,7,1])*0.1:
                target_indices.append(target)
        elif max_index == 9 and max_move_index == 9:
            # if data[max_index,target,9,1] < data[max_index,target,0,1] + abs(data[max_index,target,7,1] - data[max_index,target,7,1])*0.1:
                target_indices.append(target)
    valid_data = data[:, target_indices, :, :]

    return valid_data

def plot_target_video(file_path, data):
    frames, targets, _, _ = data.shape
    output_filename = file_path.replace('npy', 'mp4')
    if targets == 1:
        for target_index in range(targets):
        # 创建视频写入对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_filename, fourcc, FPS_extractedVideo, (1920, 1080))
            # 创建一个全白的背景
            background = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            # 遍历每一帧
            for frame_index in range(frames):
                background_copy = background.copy()  # 每一帧都需要在新的背景上绘制
                # 获取当前目标的关键点坐标
                frame_keypoints = data[frame_index, target_index]
                # 在背景上绘制每个关键点
                radius = 5  # 圆的半径
                color = (0, 0, 0)  # 黑色 (BGR格式)
                for keypoint in frame_keypoints:
                    keypoint_x, keypoint_y = keypoint
                    cv2.circle(background_copy, (int(keypoint_x), int(keypoint_y)), radius, color, -1)
                    # 连接关键点
                    connections = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 7), (2, 3), (3, 4), (7, 8), (8, 9)]
                    for connection in connections:
                        start_point = frame_keypoints[connection[0]]
                        end_point = frame_keypoints[connection[1]]
                        cv2.line(background_copy, (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])), color, 2)
                video_writer.write(background_copy)
            # 释放视频写入对象
            video_writer.release()

    if targets > 1:
        background = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, FPS_extractedVideo, (1920, 1080))
        for frame_index in range(frames):
            background_copy = background.copy()  # 每一帧都需要在新的背景上绘制
            for target_index in range(targets):
                # 获取当前目标的关键点坐标
                frame_keypoints = data[frame_index, target_index]
                # print(frame_keypoints)
                if frame_keypoints[0][0] < 1 or frame_keypoints[0][1] < 1:
                    continue
                radius = 5  # 圆的半径
                color = (0, 0, 0)  # 黑色 (BGR格式)

                for keypoint in frame_keypoints:
                    keypoint_x, keypoint_y = keypoint
                    cv2.circle(background_copy, (int(keypoint_x), int(keypoint_y)), radius, color, -1)
                    # 连接关键点
                    connections = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 7), (2, 3), (3, 4), (7, 8), (8, 9)]
                    for connection in connections:
                        start_point = frame_keypoints[connection[0]]
                        end_point = frame_keypoints[connection[1]]
                        cv2.line(background_copy, (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])), color, 2)
            video_writer.write(background_copy)
            # 释放视频写入对象
        video_writer.release()

def process_save_datavalid(save_path, dataValid, object_order):
    # 以下用于保存标注区间样本（仅保存一个，缩减人工筛选个数）
    os.makedirs(save_path, exist_ok= True)
    npy_path = f'{save_path}/{object_order}.npy'
    # np.save(npy_path, dataValid)
    # plot_target_video(npy_path, dataValid)
    # # # passon
    # valid_data = __check_pass_on(dataValid)
    # if valid_data: # 判断为传递的正样本
    #     passon_positive_path = os.path.join(save_path, 'positive')
    #     os.makedirs(passon_positive_path, exist_ok = True)
    #     if not os.path.exists(passon_positive_path + '/' +npy_path.split('/')[-1]):
    #         shutil.move(npy_path, passon_positive_path)
    #         shutil.move(npy_path.replace('npy', 'mp4'), passon_positive_path)
    # else:
    #     negative_path = os.path.join(save_path, 'negative')
    #     os.makedirs(negative_path, exist_ok=True)
    #     if not os.path.exists(negative_path + '/' +npy_path.split('/')[-1]):
    #         shutil.move(npy_path, negative_path)
    #         shutil.move(npy_path.replace('npy', 'mp4'), negative_path)

    return npy_path