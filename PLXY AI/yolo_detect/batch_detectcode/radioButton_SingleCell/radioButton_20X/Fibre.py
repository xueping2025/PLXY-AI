import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def detect_fibres(image_folder, weight_file, output_folder, progress_callback=None):
    # Load YOLOv8 model
    model = YOLO(weight_file)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Excel file path in the same output folder
    excel_output = os.path.join(output_folder, 'fibre20X_detection_results.xlsx')

    # Initialize an empty list to store data for the Excel sheet
    excel_data = []

    # Get all valid image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    total_files = len(image_files)

    # Conversion factor: 293 pixels = 100 micrometers
    pixel_to_micron = 50 / 293

    for idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, img_file)
        image = cv2.imread(img_path)

        # Perform detection with a maximum of 3000 detections
        results = model.predict(img_path, max_det=3000)[0].boxes.xyxy.cpu().numpy()

        # Sort detections: first by vertical position (y_min), then by horizontal position (x_min)
        detections = sorted(results, key=lambda x: (x[1] + x[3]) / 2)  # Sort by the center y (vertical)
        detections = sorted(detections, key=lambda x: (x[0] + x[2]) / 2)  # Then by the center x (horizontal)

        # Count the number of detected fibres
        fibre_count = len(detections)

        # Prepare a list for each image to store fibre data
        image_fibre_data = []

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

            # Draw rectangle for each detected fibre
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Put the detection number in the center of the bounding box
            cv2.putText(image, str(i + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Store fibre number, real width, real height, diameter, and area in the excel data
            image_fibre_data.append([img_file, i + 1, width_real, height_real, diameter, area])

        # Display the fibre count in the top-left corner
        cv2.putText(image, f'Count: {fibre_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save the processed image to the output folder
        output_img_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_img_path, image)

        # Append the image-level summary to excel_data
        excel_data.append([img_file, 'Total', None, None, None, fibre_count])
        # Add detailed fibre data
        excel_data.extend(image_fibre_data)

        # Update progress if callback is provided
        if progress_callback:
            progress = int((idx / total_files) * 100)
            progress_callback(progress)  # Call the callback with progress value

    # Save Excel file in the same output folder
    df = pd.DataFrame(excel_data, columns=['Image Name', 'Fibre Number', 'Width (microns)', 'Height (microns)',
                                           'Diameter (microns)', 'Area (square microns)'])
    df.to_excel(excel_output, index=False)
    print(f"Results saved to {output_folder} and Excel file {excel_output}")


def main(image_folder, weight_file, output_folder, progress_callback=None):
    """Main function to start fibre detection."""
    detect_fibres(image_folder, weight_file, output_folder, progress_callback)
