import os
import cv2
from ultralytics import YOLO

def detect_objects(image_path, weights, iou, conf, max_objects=3000):
    """
    使用YOLOv8模型进行对象检测，并返回检测后的图片和检测到的对象数目。

    参数:
        image_path (str): 要检测的图片路径。
        weights (str): YOLOv8模型的权重文件路径。
        iou (float): IoU阈值（如0.5）。
        conf (float): 置信度阈值（如0.25）。
        max_objects (int): 检测的最大对象数目（默认为3000）。

    返回:
        output_image (numpy.ndarray): 检测后的图像。
        object_count (int): 检测到的对象数目。
    """
    # 检查图片路径是否有效
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在：{image_path}")

    # 检查权重文件是否有效
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"权重文件不存在：{weights}")

    # 加载YOLOv8模型
    model = YOLO(weights)

    # 加载图片
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法加载图片：{image_path}")

    # 使用YOLOv8进行预测
    results = model.predict(image_path, iou=iou, conf=conf, max_det=max_objects)
    predictions = results[0].boxes

    # 获取检测对象数目
    object_count = len(predictions)

    # 遍历所有检测框并进行标注
    for box in predictions:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
        width, height = x2 - x1, y2 - y1

        # 计算星号大小为检测框最短边的五分之三
        star_size = int(min(width, height) * 3 / 5)

        # 在框中心位置标记黑色星号（*），大小为 star_size
        center_x, center_y = x1 + width // 2, y1 + height // 2
        cv2.putText(img, '*', (center_x - star_size // 2, center_y + star_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, star_size / 15, (0, 0, 0), 2)

    # # 在图片左上角显示检测对象的数目
    # font_scale = 10  # 放大字体大小
    # font_thickness = 15  # 增大字体粗细
    # offset_x = 50  # 水平偏移，防止文本过于靠近左边界
    # offset_y = 300  # 垂直偏移，确保文本完整显示
    #
    # # 确保文本在图片左上角有足够的空间
    # cv2.putText(img, f'Count: {object_count}', (offset_x, offset_y),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)  # 字体颜色改为红色

    return img, object_count
