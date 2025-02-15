import cv2
import json


classes = {
    'desk':0  # 课桌标签映射
}

## 处理单个json文件
def draw_rectangle(json_file='./457.json',frame = None):
    desk_info = {}
    # 载入 labelme格式的 json 标注文件
    with open(json_file, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
    img_width = labelme['imageWidth']  # 图像宽度
    img_height = labelme['imageHeight']  # 图像高度

    desk_count = 0
    for each_ann in labelme['shapes']:  # 遍历每个框

        if each_ann['shape_type'] == 'rectangle':  # 筛选出框
            if each_ann['label'] != 'desk':
                continue
            # 获取类别 ID
            bbox_class_id = classes[each_ann['label']]

            # 左上角和右下角的 XY 像素坐标
            bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
            bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
            bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
            bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))

            # 框中心点的 XY 像素坐标
            bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
            bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)

            # 框宽度
            bbox_width = abs(bbox_bottom_right_x - bbox_top_left_x)

            # 框高度
            bbox_height = abs(bbox_bottom_right_y - bbox_top_left_y)
            c1 = (int(bbox_top_left_x),bbox_top_left_y)
            c2 = (int(bbox_bottom_right_x),int(bbox_bottom_right_y))
            cv2.rectangle(frame, c1, c2, (0,0,255),1)
            label = 'desk:' + str(desk_count)
            cv2.putText(frame,label,(c1[0],c1[1]-2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),1)
            desk_info[desk_count] = [bbox_bottom_right_x,bbox_bottom_right_y]
            desk_count += 1
    return desk_info

if __name__=='__main__':

    json_file = './457.json'
    img_file = './457.jpg'
    frame = cv2.imread(img_file)
    desk_info = draw_rectangle(json_file,frame)
    cv2.imshow('desk',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
