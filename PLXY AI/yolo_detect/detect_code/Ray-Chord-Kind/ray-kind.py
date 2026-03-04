import os
import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects(image_path, weights, iou, conf):
    """
    使用YOLOv8模型检测射线种类，并返回检测后的图片和每个类别的检测数目。

    参数:
        image_path (str): 输入图片路径。
        weights (str): 模型权重文件路径。
        iou (float): IoU阈值。
        conf (float): 置信度阈值。

    返回:
        output_image (numpy.ndarray): 检测后的图片。
        class_counts (list): 每种类别的检测数量，格式为 [Class 1 Count, Class 2 Count, Class 3 Count, Class 4 Count]。
    """
    # 检查路径有效性
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"权重文件不存在：{weights}")

    # 加载模型
    model = YOLO(weights)

    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片：{image_path}")

    # 进行预测
    results = model.predict(image, iou=iou, conf=conf)
    detections = results[0].boxes.data.cpu().numpy()  # 获取检测框数据

    # 初始化类别计数
    class_counts = [0, 0, 0, 0]  # 假设类别为 0, 1, 2, 3

    # 绘制检测框和标签
    output_image = image.copy()
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 更新类别计数
        if 0 <= int(class_id) < 4:
            class_counts[int(class_id)] += 1

        # 绘制边界框和标签
        label = f"Class {int(class_id) + 1}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        label_x = x1 + (x2 - x1 - label_size[0]) // 2
        label_y = y1 + (y2 - y1 + label_size[1]) // 2
        cv2.putText(output_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # # 在图片左上角显示每种类别的检测数量
    # font_scale = 4  # 设置字体放大比例
    # font_thickness = 15  # 设置字体粗细
    # offset_x = 50  # 水平偏移，防止文本过于靠近左边界
    # offset_y = 300  # 垂直偏移
    # summary_text = (f"Class 1: {class_counts[0]}, "
    #                 f"Class 2: {class_counts[1]}, "
    #                 f"Class 3: {class_counts[2]}, "
    #                 f"Class 4: {class_counts[3]}")
    # cv2.putText(output_image, summary_text, (offset_x, offset_y),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

    return output_image, class_counts

