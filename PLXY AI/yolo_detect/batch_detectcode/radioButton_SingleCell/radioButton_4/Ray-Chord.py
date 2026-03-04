import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# 检查是否有可用的GPU，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 检测射线横切面个数函数
def detect_ray_cross(image_folder, weight_file, output_folder, progress_callback=None):
    # 加载YOLOv8模型，并将其移动到GPU（如果可用）
    model = YOLO(weight_file).to(device)

    # 输入和输出路径
    output_excel_path = os.path.join(output_folder, "detection_results.xlsx")
    output_image_folder = os.path.join(output_folder, "processed_images")

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    # 修改为每个像素对应的微米数：234个像素 = 20微米
    pixels_per_micron = 20 / 234  # 每个像素对应的微米数
    cm_in_pixels = 0.1 / 0.002  # 0.3cm in pixels (1 cm = 0.002 mm)

    # 初始化一个空的DataFrame来存储所有图片的细胞信息
    all_cell_info = pd.DataFrame(columns=["图像名", "细胞编号", "细胞宽度 (μm)", "细胞高度 (μm)", "细胞面积 (μm²)"])

    # 设置每次处理的图片数量
    batch_size = 10  # 每次处理的图片数

    # 获取所有待处理的图片列表
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg", ".tiff", ".tif"))]

    # 记录已经处理过的图片数量
    processed_images = set(all_cell_info['图像名'].tolist())

    # 以batch_size为批次处理图片
    for batch_start in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[batch_start:batch_start + batch_size]

        # 处理每个图像文件
        for i, image_filename in enumerate(batch_files):
            if image_filename in processed_images:
                continue  # 跳过已经处理的图片

            # 读取图像
            image_path = os.path.join(image_folder, image_filename)
            image = cv2.imread(image_path)

            # 检查图像是否成功读取
            if image is None:
                print(f"Warning: Failed to read image {image_filename}. Skipping this image.")
                continue  # 跳过无法读取的图像

            # 使用YOLOv8模型进行推理，设置max_det参数为1000
            results = model(image, device=device, max_det=1000)

            # 解析检测结果
            detections = results[0].boxes.data.cpu().numpy()

            # 如果没有检测到框，跳过此图像
            if len(detections) == 0:
                print(f"No detections for image {image_filename}. Skipping this image.")
                continue

            # 总检测框数量
            total_boxes = len(detections)

            # 按y1进行初步排序
            detections = sorted(detections, key=lambda x: x[1])

            # 分行处理，忽略检测框少于5个的行
            rows = []
            current_row = []
            for detection in detections:
                if current_row and detection[1] - current_row[-1][1] > cm_in_pixels:
                    if len(current_row) >= 5:  # 只有检测框数目大于等于5个才认为是一行
                        rows.append(current_row)
                    current_row = []
                current_row.append(detection)
            if len(current_row) >= 5:
                rows.append(current_row)

            # 在图像上绘制检测框和编号
            output_image = image.copy()
            cell_info = []  # 存储每个细胞的信息：编号、长、宽、面积

            for row_idx, row in enumerate(rows):
                # 对每一行内的检测框按x1从左到右排序
                row = sorted(row, key=lambda x: x[0])

                for idx, detection in enumerate(row):
                    x1, y1, x2, y2, score, class_id = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    width_pixels = x2 - x1
                    height_pixels = y2 - y1
                    width_microns = width_pixels * pixels_per_micron
                    height_microns = height_pixels * pixels_per_micron
                    area_microns = width_microns * height_microns

                    # 绘制边界框，按类别区分颜色
                    color = (0, 255, 0)  # 默认绿色
                    if class_id == 1:
                        color = (255, 0, 0)  # 红色
                    elif class_id == 2:
                        color = (0, 0, 255)  # 蓝色
                    elif class_id == 3:
                        color = (0, 255, 255)  # 黄色

                    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

                    # 生成细胞编号，格式：Row-Index
                    label = f"{row_idx + 1}-{idx + 1}"

                    # 在边界框中间绘制编号
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    label_x = x1 + (width_pixels - label_size[0]) // 2
                    label_y = y1 + (height_pixels + label_size[1]) // 2
                    cv2.putText(output_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 保存细胞信息
                    cell_info.append({
                        "图像名": image_filename,
                        "细胞编号": label,
                        "细胞宽度 (μm)": width_microns,
                        "细胞高度 (μm)": height_microns,
                        "细胞面积 (μm²)": area_microns
                    })

            # 将当前图片的细胞信息添加到所有细胞信息列表中
            cell_info_df = pd.DataFrame(cell_info)
            all_cell_info = pd.concat([all_cell_info, cell_info_df], ignore_index=True)

            # 保存处理后的图像
            output_image_path = os.path.join(output_image_folder, image_filename)
            success = cv2.imwrite(output_image_path, output_image)

            # 检查图像是否成功保存
            if success:
                print(f"Saved processed image: {output_image_path}")
            else:
                print(f"Failed to save image: {output_image_path}")

            # 更新进度条
            if progress_callback:
                progress = int(((batch_start + i) / len(image_files)) * 100)
                progress_callback(progress)

    # 保存最终的细胞信息到Excel文件
    all_cell_info.to_excel(output_excel_path, index=False)
    print(f"Detection results saved to {output_excel_path}")

def main(image_folder, weight_file, output_folder, progress_callback=None):
    """Main function to start ray-cross detection."""
    detect_ray_cross(image_folder, weight_file, output_folder, progress_callback)
