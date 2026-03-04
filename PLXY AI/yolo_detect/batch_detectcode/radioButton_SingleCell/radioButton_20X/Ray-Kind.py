import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# 细胞类型颜色映射
cell_colors = {
    'A': (0, 255, 0),  # 绿色
    'B': (0, 0, 255),  # 红色
    'C': (255, 0, 0),  # 蓝色
    'D': (0, 255, 255),  # 黄色
}


# 细胞检测及结果保存
def detect_cells_with_labels(image_folder, weight_file, output_folder, progress_callback=None):
    # 加载YOLOv8模型
    model = YOLO(weight_file)

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 输出Excel文件路径
    excel_output = os.path.join(output_folder, 'Ray-Kind20X_detection_results.xlsx')

    # 初始化Excel数据列表
    excel_data = []

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    total_files = len(image_files)

    # 微米转换比例
    pixel_to_micron = 50 / 293  # 50像素 = 293微米

    # 遍历每个图像文件
    for idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, img_file)
        image = cv2.imread(img_path)

        # 使用YOLO模型进行检测
        results = model(img_path, max_det=1000)

        # 获取检测结果（xyxy格式：[x_min, y_min, x_max, y_max, confidence, class_id]）
        detections = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        # 细胞类型标签（假设模型的类别顺序对应于A、B、C、D）
        class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        # 初始化每种细胞类型的编号
        cell_type_counter = {'A': 1, 'B': 1, 'C': 1, 'D': 1}

        # 遍历所有检测结果
        for det_idx, (detection, class_id) in enumerate(zip(detections, class_ids)):
            # 提取检测框坐标
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            width = x_max - x_min
            height = y_max - y_min
            area = width * height * (pixel_to_micron ** 2)  # 计算面积，单位为平方微米

            # 获取细胞类型和颜色
            cell_type = class_labels[int(class_id)]
            color = cell_colors.get(cell_type, (255, 255, 255))  # 默认白色

            # 生成唯一标识符，按照每个细胞类型的编号递增
            cell_id = f"{cell_type}_{cell_type_counter[cell_type]}"

            # 更新每种细胞类型的编号
            cell_type_counter[cell_type] += 1

            # 计算中心点坐标
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # 绘制检测框和唯一标识符（ID）
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            font_scale = 1.25  # 字体缩放
            thickness = 2  # 文字厚度
            text_size, _ = cv2.getTextSize(cell_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(image, cell_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

            # 将检测信息保存到Excel数据列表
            excel_data.append([
                img_file, cell_id, cell_type,
                width * pixel_to_micron, height * pixel_to_micron, area
            ])

        # 保存检测结果图像
        output_img_path = os.path.join(output_folder, img_file)
        cv2.putText(image, f'Detections: {len(detections)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(output_img_path, image)

        # 更新进度条
        if progress_callback:
            progress = int((idx / total_files) * 100)
            progress_callback(progress)

    # 保存数据到Excel
    df = pd.DataFrame(excel_data, columns=[
        'Image Name', 'Cell ID', 'Cell Type', 'Width (microns)', 'Height (microns)', 'Area (square microns)'
    ])
    df.to_excel(excel_output, index=False)
    print(f"Results saved to {output_folder} and Excel file {excel_output}")


# 主函数
def main(image_folder, weight_file, output_folder, progress_callback=None):
    """主程序启动细胞检测并保存结果"""
    detect_cells_with_labels(image_folder, weight_file, output_folder, progress_callback)
