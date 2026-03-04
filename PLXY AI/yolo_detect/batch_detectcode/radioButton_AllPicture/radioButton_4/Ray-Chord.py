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

    # 初始化一个空的DataFrame来存储所有图片的统计信息
    if os.path.exists(output_excel_path):
        all_results = pd.read_excel(output_excel_path)  # 读取之前保存的Excel
    else:
        all_results = pd.DataFrame(
            columns=["图片名", "细胞总数", "Row总数", "每Row平均细胞数", "平均检测框长 (μm)", "平均检测框宽 (μm)",
                     "平均检测框面积 (μm²)", "检测框总面积 (μm²)", "检测框面积占比 (%)", "细胞编号", "细胞长 (μm)", "细胞宽 (μm)", "细胞面积 (μm²)"])

    # 设置每次处理的图片数量
    batch_size = 10  # 每次处理的图片数

    # 获取所有待处理的图片列表
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg", ".tiff", ".tif"))]

    # 记录已经处理过的图片数量
    processed_images = set(all_results['图片名'].tolist())

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

            image_area = image.shape[0] * image.shape[1]  # 计算整张图像的面积

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
            row_stats = []  # 存储每一行的统计信息
            total_area = 0
            total_width = 0
            total_height = 0
            cell_info = []  # 存储每个细胞的信息：编号、长、宽、面积

            for row_idx, row in enumerate(rows):
                # 对每一行内的检测框按x1从左到右排序
                row = sorted(row, key=lambda x: x[0])

                row_length = 0
                row_width = 0
                row_area = 0

                for idx, detection in enumerate(row):
                    x1, y1, x2, y2, score, class_id = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    width_pixels = x2 - x1
                    height_pixels = y2 - y1
                    width_microns = width_pixels * pixels_per_micron
                    height_microns = height_pixels * pixels_per_micron
                    area_microns = width_microns * height_microns

                    # 累加当前行的长度、宽度和面积
                    row_length += width_microns
                    row_width += height_microns
                    row_area += area_microns

                    # 累加图片的总长、宽和面积
                    total_width += width_microns
                    total_height += height_microns
                    total_area += area_microns

                    # 绘制边界框，按类别区分颜色
                    color = (0, 0, 255)  # 默认绿色
                    if class_id == 1:
                        color = (0, 0, 255)  # 红色
                    elif class_id == 2:
                        color = (0, 0, 255)  # 蓝色
                    elif class_id == 3:
                        color = (0, 0, 255)  # 黄色

                    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)

                    # 生成细胞编号，格式：Row-Index
                    label = f"{row_idx + 1}-{idx + 1}"

                    # 在边界框中间绘制编号
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
                    label_x = x1 + (width_pixels - label_size[0]) // 2
                    label_y = y1 + (height_pixels + label_size[1]) // 2
                    cv2.putText(output_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # 保存细胞信息
                    cell_info.append({
                        "细胞编号": label,
                        "细胞长 (μm)": width_microns,
                        "细胞宽 (μm)": height_microns,
                        "细胞面积 (μm²)": area_microns
                    })

            # 计算图片的统计信息
            row_count = len(rows)
            average_cells_per_row = total_boxes / row_count if row_count > 0 else 0
            average_width = total_width / total_boxes if total_boxes > 0 else 0
            average_height = total_height / total_boxes if total_boxes > 0 else 0
            average_area = total_area / total_boxes if total_boxes > 0 else 0
            area_percentage = (total_area / (image_area * pixels_per_micron * pixels_per_micron)) * 100

            # 添加统计信息到DataFrame
            temp_df = pd.DataFrame({
                "图片名": [image_filename],
                "细胞总数": [total_boxes],
                "Row总数": [row_count],
                "每Row平均细胞数": [average_cells_per_row],
                "平均检测框长 (μm)": [average_width],
                "平均检测框宽 (μm)": [average_height],
                "平均检测框面积 (μm²)": [average_area],
                "检测框总面积 (μm²)": [total_area],
                "检测框面积占比 (%)": [area_percentage],
            })

            all_results = pd.concat([all_results, temp_df], ignore_index=True)

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

    # 保存最终结果
    all_results.to_excel(output_excel_path, index=False)
    print(f"Detection results saved to {output_excel_path}")

def main(image_folder, weight_file, output_folder, progress_callback=None):
    """Main function to start ray-cross detection."""
    detect_ray_cross(image_folder, weight_file, output_folder, progress_callback)
