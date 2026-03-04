import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def _compute_font_scale_only(text, target_px, thickness=3, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    根据目标像素长度计算 fontScale，使得文字的 max(宽,高) ≈ target_px。
    thickness 固定（不改变），返回 fontScale（下限 0.01）。
    """
    (w0, h0), baseline = cv2.getTextSize(text, font, 1.0, thickness)
    base_max = max(w0, h0)
    if base_max == 0:
        return 1.0
    scale = float(target_px) / float(base_max)
    return max(scale, 0.01)


def detect_fibres(image_folder, weight_file, output_folder, progress_callback=None):
    # Load YOLOv8 model
    model = YOLO(weight_file)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Excel file path in the same output folder
    excel_output = os.path.join(output_folder, 'fibre40X_detection_results.xlsx')

    # Initialize an empty list to store data for the Excel sheet
    excel_data = []

    # Get all valid image files (case-insensitive)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    total_files = len(image_files)

    # Conversion factor: 20 pixels = 234 micrometers
    # （保持你原来的数值，如果需要 20/234，请手动修改）
    pixel_to_micron = 20.0 / 234.0

    for idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: failed to read {img_path}, skip.")
            continue

        # Perform detection with a maximum of 3000 detections
        raw_results = model.predict(img_path, max_det=3000)
        # 确保结果存在且格式正确
        if len(raw_results) == 0 or not hasattr(raw_results[0], 'boxes'):
            detections = []
        else:
            detections = raw_results[0].boxes.xyxy.cpu().numpy()

        # Sort detections from left to right based on x_min
        detections = sorted(detections, key=lambda x: float(x[0])) if len(detections) > 0 else []

        # Count the number of detected fibres
        fibre_count = len(detections)

        # Initialize sums for calculating averages
        total_width = 0.0
        total_height = 0.0
        total_diameter = 0.0
        total_area = 0.0

        # Draw bounding boxes and label them with their index
        for i, det in enumerate(detections):
            x_min, y_min, x_max, y_max = map(int, det[:4])

            # Clamp coordinates to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1] - 1, x_max)
            y_max = min(image.shape[0] - 1, y_max)

            # Ensure valid box
            width = max(1, x_max - x_min)
            height = max(1, y_max - y_min)

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            area = (width * height) * (pixel_to_micron ** 2)
            diameter = max(width, height) * pixel_to_micron
            width_real = width * pixel_to_micron
            height_real = height * pixel_to_micron

            # Accumulate values for averages
            total_width += width_real
            total_height += height_real
            total_diameter += diameter
            total_area += area

            # Draw rectangle for each detected fibre
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

            # =========== Put label centered in bbox ===========
            label = str(i + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fixed_thickness = 3

            # 目标文字尺寸：取最短边的 0.8 倍（你可以改成 0.5 或其他）
            min_side = min(width, height)
            target_len = max(1.0, 0.8 * min_side)

            # 计算合适的 font_scale（只调整 scale，不改变 thickness）
            font_scale = _compute_font_scale_only(label, target_len, thickness=fixed_thickness, font=font)

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, fixed_thickness)

            # 计算文本 origin（y 是 baseline）
            # 初始希望位置：水平中心，垂直中心（baseline 放在 center_y + th/2）
            org_x = int(center_x - tw / 2)
            org_y = int(center_y + th / 2)

            # Clamp to keep text inside box:
            # 水平：x_min <= org_x <= x_max - tw
            org_x = max(x_min, min(org_x, x_max - tw))

            # 垂直：确保 top >= y_min -> org_y >= y_min + th
            #         确保 bottom <= y_max -> org_y + baseline <= y_max  => org_y <= y_max - baseline
            min_org_y = y_min + th
            max_org_y = y_max - baseline
            # 防止 min_org_y > max_org_y（当框非常短时），退化为把 baseline 放在 y_min + th (或 y_max)
            if min_org_y > max_org_y:
                # 让 org_y 在 [y_min + th, y_max]
                org_y = max(y_min + th, min(org_y, y_max))
            else:
                org_y = max(min_org_y, min(org_y, max_org_y))

            # Finally put text (red, thickness fixed)
            cv2.putText(image, label, (org_x, org_y), font, font_scale, (0, 0, 255), fixed_thickness, cv2.LINE_AA)

        # Calculate averages
        if fibre_count > 0:
            avg_width = total_width / fibre_count
            avg_height = total_height / fibre_count
            avg_diameter = total_diameter / fibre_count
            avg_area = total_area / fibre_count
        else:
            avg_width = avg_height = avg_diameter = avg_area = 0.0

        # Append the image-level summary to excel_data
        excel_data.append([img_file, fibre_count, avg_width, avg_height, avg_diameter, avg_area])

        # Display the fibre count in the top-left corner
        cv2.putText(image, f'Count: {fibre_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Save the processed image to the output folder
        output_img_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_img_path, image)

        # Update progress if callback is provided
        if progress_callback and total_files > 0:
            progress = int((idx / total_files) * 100)
            progress_callback(progress)  # Call the callback with progress value

    # Save Excel file in the same output folder
    df = pd.DataFrame(excel_data, columns=[
        'Image Name', 'Fibre Count', 'Average Width (microns)', 'Average Height (microns)',
        'Average Diameter (microns)', 'Average Area (square microns)'
    ])
    df.to_excel(excel_output, index=False)
    print(f"Results saved to {output_folder} and Excel file {excel_output}")


def main(image_folder, weight_file, output_folder, progress_callback=None):
    """Main function to start fibre detection."""
    detect_fibres(image_folder, weight_file, output_folder, progress_callback)
