import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def _compute_font_scale_only(text, target_px, thickness=3, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    仅根据目标像素高度/宽度计算 fontScale，使得文字的 max(宽,高) ≈ target_px。
    注意：不改变 thickness，保持传入的 thickness 不变。
    """
    # 基于 fontScale=1.0 先测量基准尺寸（厚度固定）
    (w0, h0), _ = cv2.getTextSize(text, font, 1.0, thickness)
    base_max = max(w0, h0)
    if base_max == 0:
        return 1.0  # 兜底
    scale = float(target_px) / float(base_max)
    return max(scale, 0.01)  # 防止极小值


def detect_vessels(image_folder, weight_file, output_folder, progress_callback=None):
    # Load YOLOv8 model
    model = YOLO(weight_file)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Excel file path in the same output folder
    excel_output = os.path.join(output_folder, 'vessel40X_detection_results.xlsx')

    # Initialize an empty list to store data for the Excel sheet
    excel_data = []

    # Get all valid image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    total_files = len(image_files)

    # Conversion factor: 234 pixels = 20 micrometers
    pixel_to_micron = 20 / 234

    for idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, img_file)
        image = cv2.imread(img_path)
        results = model(image)

        # Get detections (xyxy format: [x_min, y_min, x_max, y_max, confidence, class_id])
        detections = results[0].boxes.xyxy.cpu().numpy()

        # Sort detections from left to right based on x_min
        detections = sorted(detections, key=lambda x: x[0])

        # Count the number of detected vessels
        vessel_count = len(detections)

        # Initialize sums for calculating averages
        total_width = 0
        total_height = 0
        total_diameter = 0
        total_area = 0

        # Draw bounding boxes and label them with their index
        for i, det in enumerate(detections):
            x_min, y_min, x_max, y_max = map(int, det[:4])
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            width = x_max - x_min
            height = y_max - y_min
            area = (width * height) * (pixel_to_micron ** 2)  # Calculate the area in square micrometers
            diameter = max(width, height) * pixel_to_micron  # Calculate the diameter in micrometers
            width_real = width * pixel_to_micron  # Convert width to micrometers
            height_real = height * pixel_to_micron  # Convert height to micrometers

            # Accumulate values for averages
            total_width += width_real
            total_height += height_real
            total_diameter += diameter
            total_area += area

            # Draw rectangle for each detected vessel （保持不变）
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

            # ==== 仅调整“数字大小”，不改粗细 ====
            # 目标尺寸：最短边的 0.8 倍
            min_side = max(1, min(width, height))
            target_len = 0.5 * min_side

            # 固定 thickness=3，只计算合适的 fontScale
            font = cv2.FONT_HERSHEY_SIMPLEX
            fixed_thickness = 3
            font_scale = _compute_font_scale_only(str(i + 1), target_len, thickness=fixed_thickness, font=font)

            # 重新测一次文字尺寸，用于居中
            (tw, th), _ = cv2.getTextSize(str(i + 1), font, font_scale, fixed_thickness)
            org_x = int(center_x - tw / 2)
            org_y = int(center_y + th / 2)

            # Put the detection number in the center of the bounding box（厚度保持 3）
            cv2.putText(image, str(i + 1), (org_x, org_y), font, font_scale, (0, 0, 255), fixed_thickness, cv2.LINE_AA)

        # Calculate averages
        if vessel_count > 0:
            avg_width = total_width / vessel_count
            avg_height = total_height / vessel_count
            avg_diameter = total_diameter / vessel_count
            avg_area = total_area / vessel_count
        else:
            avg_width = avg_height = avg_diameter = avg_area = 0

        # Append the image-level summary to excel_data
        excel_data.append([img_file, vessel_count, avg_width, avg_height, avg_diameter, avg_area])

        # Display the vessel count in the top-left corner（保持不变）
        cv2.putText(image, f'Count: {vessel_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Save the processed image to the output folder
        output_img_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_img_path, image)

        # Update progress if callback is provided
        if progress_callback:
            progress = int((idx / total_files) * 100)
            progress_callback(progress)  # Call the callback with progress value

    # Save Excel file in the same output folder
    df = pd.DataFrame(excel_data, columns=['Image Name', 'Vessel Count', 'Average Width (microns)', 'Average Height (microns)', 'Average Diameter (microns)', 'Average Area (square microns)'])
    df.to_excel(excel_output, index=False)
    print(f"Results saved to {output_folder} and Excel file {excel_output}")


def main(image_folder, weight_file, output_folder, progress_callback=None):
    """Main function to start vessel detection."""
    detect_vessels(image_folder, weight_file, output_folder, progress_callback)
