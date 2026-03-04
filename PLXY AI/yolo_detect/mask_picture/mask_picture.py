import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QDialog, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QPoint
from PIL import Image
import os


class ROISelector(QDialog):
    def __init__(self, image, scale_factor, main_app=None):
        super().__init__()
        self.setWindowTitle("Select the area of interest (click 'Finish' to end the drawing)")

        self.original_image = image.copy()
        self.scale_factor = scale_factor
        self.main_app = main_app  # ✅ 保存主界面引用

        # 其余保持不变……


        # 缩放后的图像尺寸
        self.image_height, self.image_width, _ = self.original_image.shape
        self.setFixedSize(self.image_width, self.image_height + 50)  # 额外空间放按钮

        # 绘图相关变量
        self.drawing = False
        self.points = []
        self.current_path = []

        # 布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.image_width, self.image_height)
        self.image_label.setStyleSheet("QLabel { background-color : lightgray; }")
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.layout.addWidget(self.image_label)

        # 按钮布局
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        # 完成按钮
        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self.finish_drawing)
        self.button_layout.addWidget(self.finish_button)

        # 重置按钮
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_drawing)
        self.button_layout.addWidget(self.reset_button)

        # 显示初始图像
        self.show_image()

    def show_image(self):
        # 将BGR图像转换为RGB
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.update_display()

    def update_display(self):
        # Create a temporary pixmap to draw on
        temp_pixmap = self.pixmap.copy()
        painter = QPainter(temp_pixmap)
        pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
        painter.setPen(pen)

        if self.points:
            for i in range(1, len(self.points)):
                painter.drawLine(self.points[i - 1], self.points[i])

        painter.end()

        # Show on local image_label
        self.image_label.setPixmap(temp_pixmap)

        # ✅ Sync to main window's median_HPicture label
        if self.main_app is not None:
            try:
                self.main_app.ui.median_HPicture.setAlignment(Qt.AlignCenter)
                self.main_app.ui.median_HPicture.setScaledContents(False)
                self.main_app.ui.median_HPicture.setPixmap(
                    temp_pixmap.scaled(
                        self.main_app.ui.median_HPicture.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
                print("[SYNC] ROI image updated to median_HPicture.")
            except Exception as e:
                print(f"[SYNC ERROR] Failed to update median_HPicture: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            pos = event.position().toPoint()
            if 0 <= pos.x() < self.image_width and 0 <= pos.y() < self.image_height:
                self.points.append(pos)
                self.update_display()

    def mouseMoveEvent(self, event):
        if self.drawing:
            pos = event.position().toPoint()
            if 0 <= pos.x() < self.image_width and 0 <= pos.y() < self.image_height:
                self.points.append(pos)
                self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def finish_drawing(self):
        if len(self.points) < 3:
            QMessageBox.warning(self, "Warning", "At least three points are needed to form a region.")
            return
        self.accept()

    def reset_drawing(self):
        self.points = []
        self.update_display()

    def get_mask(self):
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        if len(self.points) >= 3:
            # 使用多边形填充
            polygon = [(point.x(), point.y()) for point in self.points]
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
        return mask


class MainWindow(QMainWindow):
    def __init__(self, main_app=None):
        super().__init__()
        self.main_app = main_app  # 👈 这里保存主程序传过来的窗口对象

        self.setWindowTitle("Cell Detection and Recognition - ROI Selection Tool")
        self.resize(1200, 900)

        # 布局
        self.layout = QVBoxLayout()

        # 图像显示标签
        self.image_label = QLabel("Please load an image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color : lightgray; }")
        self.layout.addWidget(self.image_label)

        # 按钮布局
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        # 加载图像按钮
        self.load_button = QPushButton("Load image")
        self.load_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.load_button)

        # 选择ROI并保存按钮
        self.select_button = QPushButton("Select ROI and save")
        self.select_button.clicked.connect(self.select_roi)
        self.select_button.setEnabled(False)
        self.button_layout.addWidget(self.select_button)

        # 设置主窗口的中心部件
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # 初始化图像变量
        self.original_image = None
        self.image_path = None
        self.display_image = None
        self.scale_factor = 1.0

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if file_name:
            # Use PIL to support Chinese path
            try:
                pil_image = Image.open(file_name)
                pil_image = pil_image.convert("RGB")
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                self.image_path = file_name
            except Exception as e:
                QMessageBox.critical(self, "Error", f"The image cannot be loaded: {e}")
                return

            self.prepare_display_image()
            self.show_image(self.display_image)
            self.select_button.setEnabled(True)

            # ✅ Sync to main app display (input_HPicture)
            if self.main_app is not None:
                try:
                    height, width, channel = self.original_image.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        self.original_image.data, width, height,
                        bytes_per_line, QImage.Format.Format_RGB888
                    ).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_image)
                    self.main_app.ui.input_HPicture.setPixmap(
                        pixmap.scaled(
                            self.main_app.ui.input_HPicture.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                    )
                    print("[SYNC] Image displayed on input_HPicture.")
                except Exception as e:
                    print(f"[SYNC ERROR] Failed to display image on input_HPicture: {e}")

    def prepare_display_image(self):
        if self.original_image is None:
            return

        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        max_width = screen_geometry.width() - 200
        max_height = screen_geometry.height() - 200

        orig_height, orig_width, _ = self.original_image.shape

        # 计算缩放比例
        width_ratio = max_width / orig_width
        height_ratio = max_height / orig_height
        self.scale_factor = min(width_ratio, height_ratio, 1.0)  # 不放大

        if self.scale_factor < 1.0:
            new_width = int(orig_width * self.scale_factor)
            new_height = int(orig_height * self.scale_factor)
            self.display_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            self.display_image = self.original_image.copy()

    def show_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def select_roi(self):
        if self.original_image is None:
            return

        # 打开ROI选择对话框
        roi_dialog = ROISelector(self.display_image, self.scale_factor, self.main_app)
        roi_dialog.exec()

        # 获取掩码
        mask_display = roi_dialog.get_mask()
        if mask_display is None or not np.any(mask_display):
            QMessageBox.warning(self, "Warning", "No valid ROI area has been selected")
            return

        # 将掩码映射回原始图像尺寸
        if self.scale_factor != 1.0:
            mask = cv2.resize(mask_display, (self.original_image.shape[1], self.original_image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_display.copy()

        # 应用掩码：保留ROI区域，非ROI区域设为黑色
        masked_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)

        # 显示掩盖后的图像
        self.prepare_display_image_for_mask(masked_image)
        self.show_image(self.display_image)
        # ✅ Sync masked image to main app's output_HPicture
        if self.main_app is not None:
            try:
                height, width, channel = masked_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    masked_image.data, width, height,
                    bytes_per_line, QImage.Format.Format_RGB888
                ).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)

                self.main_app.ui.output_HPicture.setAlignment(Qt.AlignCenter)
                self.main_app.ui.output_HPicture.setScaledContents(False)
                self.main_app.ui.output_HPicture.setPixmap(
                    pixmap.scaled(
                        self.main_app.ui.output_HPicture.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
                print("[SYNC] Final masked image updated to output_HPicture.")
            except Exception as e:
                print(f"[SYNC ERROR] Failed to update output_HPicture: {e}")

        # 选择保存路径
        input_ext = os.path.splitext(self.image_path)[1]
        save_filter = f"Images (*{input_ext})"
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save the processed image", "", save_filter
        )
        if output_path:
            # 使用PIL保存以支持多种格式和无损保存
            try:
                # 确保输出路径的文件夹存在
                if os.path.dirname(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 将BGR转换为RGB
                masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                pil_masked = Image.fromarray(masked_image_rgb)

                # 根据输入格式保存
                input_format = input_ext.lower()
                if input_format in ['.jpg', '.jpeg']:
                    pil_masked.save(output_path, format='JPEG', quality=95)  # 高质量JPEG
                elif input_format in ['.png']:
                    pil_masked.save(output_path, format='PNG', compress_level=1)  # 低压缩PNG
                elif input_format in ['.tiff', '.tif']:
                    pil_masked.save(output_path, format='TIFF')  # 无损TIFF
                else:
                    pil_masked.save(output_path)  # 默认保存

                QMessageBox.information(self, "Success", f"The image has been saved to: {output_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Unable to save the image：{e}")

    def prepare_display_image_for_mask(self, masked_image):
        if self.original_image is None:
            return

        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        max_width = screen_geometry.width() - 200
        max_height = screen_geometry.height() - 200

        orig_height, orig_width, _ = masked_image.shape

        # 计算缩放比例
        self.scale_factor = min(max_width / orig_width, max_height / orig_height, 1.0)

        if self.scale_factor < 1.0:
            new_width = int(orig_width * self.scale_factor)
            new_height = int(orig_height * self.scale_factor)
            self.display_image = cv2.resize(masked_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            self.display_image = masked_image.copy()


def main(main_app=None):
    window = MainWindow(main_app)  # 传入主界面引用
    window.show()
    return window



if __name__ == "__main__":
    main()
