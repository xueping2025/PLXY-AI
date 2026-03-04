import sys
import os
from PySide6.QtWidgets import QFileDialog, QApplication, QMainWindow
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import QMessageBox
from modules import *
from modules import Ui_MainWindow
from widgets import *
import importlib.util
import cv2
from PIL import Image  # 引入 Pillow 库
import numpy as np

os.environ["QT_FONT_DPI"] = "96"
widgets = None

# --- 路径与加载工具（放在 main.py 顶部） ---
from pathlib import Path
import sys, runpy, importlib, importlib.util

######################
# --- 放在 main.py 顶部（import 之后）---
from pathlib import Path
import os, sys



def set_cwd_to_app_root() -> Path:
    """
    将当前工作目录强制切到“应用根”：
    - 打包后：exe 所在目录（把 exe 放在项目根即可）
    - 开发期：main.py 所在目录
    返回：应用根 Path
    """
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) \
           else Path(__file__).resolve().parent
    os.chdir(root)                # 关键：切换 CWD
    if str(root) not in sys.path: # 方便 import 包
        sys.path.insert(0, str(root))
    return root

APP_ROOT = set_cwd_to_app_root()
print("Application root directory:", APP_ROOT)
print("Current working directory:", Path.cwd())








#######################


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # 设置标题和描述
        Settings.ENABLE_CUSTOM_TITLE_BAR = True
        self.setWindowTitle("PLXY AI")
        widgets.titleRightInfo.setText("PLXY AI")


        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"
        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)
            # SET HACKS
            AppFunctions.setThemeHack(self)



        # 左侧菜单按钮绑定
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)

        # 初始化按钮事件绑定
        widgets.btn_home.clicked.connect(self.buttonClick)
        #widgets.btn_widgets.clicked.connect(self.buttonClick)
        #widgets.btn_new.clicked.connect(self.buttonClick)
        #widgets.btn_save.clicked.connect(self.buttonClick)
        widgets.btn_detect2.clicked.connect(self.buttonClick)
        widgets.btn_batch.clicked.connect(self.buttonClick)
        widgets.ImageStitching.clicked.connect(self.buttonClick)
        widgets.btn_HandMask.clicked.connect(self.buttonClick)

        widgets.btn_openModel.clicked.connect(self.open_model_path)
        widgets.btn_openInput.clicked.connect(self.open_image_folder)
        widgets.pushButton_2.clicked.connect(self.run_detection)
        widgets.btn_next.clicked.connect(self.show_next_image)
        widgets.btn_Pre.clicked.connect(self.show_previous_image)

        #HandMask界面
        widgets.btn_handmask.clicked.connect(lambda: self.execute_handmask_script("./yolo_detect/mask_picture/mask_picture.py"))
        widgets.btn_handmask.clicked.connect(self.launch_mask_selector)
        self.ui.input_HPicture.setAlignment(Qt.AlignCenter)
        self.ui.input_HPicture.setScaledContents(False)

        #单张图片检测控制输出图片大小
        self.ui.input_detect.setAlignment(Qt.AlignCenter)
        self.ui.input_detect.setScaledContents(False)

        # 批量界面
        # widgets.btn_openModel_2.clicked.connect(self.open_model_path2)
        # widgets.btn_openInput_2.clicked.connect(self.open_folder)
        # widgets.btn_outputs_dir.clicked.connect(self.out_folder)

        # 批量界面按钮绑定
        widgets.btn_openModel_2.clicked.connect(self.open_model_path2)
        widgets.btn_openInput_2.clicked.connect(self.open_image_folder2)
        widgets.btn_outputs_dir.clicked.connect(self.open_output_folder)
        widgets.detect_batch.clicked.connect(self.run_selected_script)  # 新增运行按钮绑定

        #拼接界面按钮绑定
        widgets.btn_image_left.clicked.connect(self.open_concat_left_image)
        widgets.btn_image_right.clicked.connect(self.open_concat_right_image)

        widgets.btn_concat.clicked.connect(self.concat_images)
        widgets.btn_image_save.clicked.connect(self.save_concat_picture)


        # 新增功能按钮
        widgets.save_picture.clicked.connect(self.save_current_picture)

        # 滑条设置
        widgets.horizontalSlider_conf.setMinimum(1)
        widgets.horizontalSlider_conf.setMaximum(100)
        widgets.horizontalSlider_conf.setValue(30)
        widgets.horizontalSlider_conf.setTickInterval(1)
        widgets.horizontalSlider_conf.valueChanged.connect(self.update_detection)
        widgets.horizontalSlider_IoU.setMinimum(1)
        widgets.horizontalSlider_IoU.setMaximum(100)
        widgets.horizontalSlider_IoU.setValue(40)
        widgets.horizontalSlider_IoU.setTickInterval(1)
        widgets.horizontalSlider_IoU.valueChanged.connect(self.update_detection)
        widgets.progressBar.setValue(0)

        # 左右面板展开/关闭按钮绑定
        #widgets.toggleLeftBox.clicked.connect(lambda: UIFunctions.toggleLeftBox(self, True))
        widgets.extraCloseColumnBtn.clicked.connect(lambda: UIFunctions.toggleLeftBox(self, True))
        #widgets.settingsTopBtn.clicked.connect(lambda: UIFunctions.toggleRightBox(self, True))

        #截图功能暂时
        widgets.settingsTopBtn.clicked.connect(lambda: self.capture_current_interface(r"./1.tif"))

        # 初始化主界面
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

        # 初始化变量
        self.image_files = []
        self.current_image_index = 0
        self.image_dir_path = None
        self.detect_code_path = ""

        self.group1 = QButtonGroup(self)
        self.group2 = QButtonGroup(self)

        self.group1.addButton(widgets.radioButton_SingleCell)
        self.group1.addButton(widgets.radioButton_AllPicture)

        self.group2.addButton(widgets.radioButton_10X)
        self.group2.addButton(widgets.radioButton_20X)
        self.group2.addButton(widgets.radioButton_4)

        # 绑定 comboBox 选项改变事件
        self.ui.comboBox_CellKind.currentIndexChanged.connect(self.update_code_path)

        # 初始化界面和路径设置
        self.update_code_path()

        # 初始化label内容
        self.update_conf_label(self.ui.horizontalSlider_conf.value())
        self.update_iou_label(self.ui.horizontalSlider_IoU.value())

        # 显示主窗口
        self.show()

    def capture_current_interface(self, save_path="interface_capture.png"):
        pixmap = self.grab()  # 截取当前窗口内容（不含桌面边框）
        pixmap.save(save_path)
        print(f"[INFO] Screenshot saved to {save_path}")

    def launch_mask_selector(self):
        self.external_window = self.run_mask_selector()

    def run_mask_selector(self):
        import importlib.util
        script_path = "./yolo_detect/mask_picture/mask_picture.py"
        spec = importlib.util.spec_from_file_location("mask_module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "main"):
            return module.main(main_app=self)  # 👈 把主窗口传进去
    def execute_handmask_script(self, relative_path):
        """Execute a Python script from a given relative path"""
        script_path = os.path.abspath(relative_path)

        if not os.path.isfile(script_path):
            QMessageBox.critical(self, "Error", f"Script file not found:\n{script_path}")
            return

        try:
            spec = importlib.util.spec_from_file_location("custom_script", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "main"):
                # Keep the window alive by storing it as an attribute
                self.external_window = module.main()
                print(f"Script executed successfully: {script_path}")
            else:
                QMessageBox.warning(self, "Warning", f"No 'main()' function defined in:\n{script_path}")
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Failed to execute the script:\n{e}")

    def execute_script(self, script_path, model_path, input_folder, output_folder, progress_callback=None):
        """
        动态加载并执行指定的 Python 脚本，并传递必要的参数
        """
        if not os.path.isfile(script_path):
            print(f"Script file does not exist: {script_path}")
            return

        try:
            spec = importlib.util.spec_from_file_location("vessel_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 调用 Vessel.py 的 main 方法
            if hasattr(module, "main"):
                module.main(input_folder, model_path, output_folder, progress_callback)
                print(
                    f"Script executed successfully: {script_path}, arguments passed: input_folder={input_folder}, model_path={model_path}, output_folder={output_folder}")
            else:
                print(f"No 'main' function defined in script {script_path}")
        except Exception as e:
            print(f"Failed to load script: {script_path}\nError message: {e}")

    def get_script_path(self):
        """根据用户选择获取对应脚本路径"""
        base_path = os.path.abspath("./yolo_detect/batch_detectcode")  # 使用绝对路径
        print(f"Base path (absolute path): {base_path}")

        if not os.path.isdir(base_path):
            print(f"Base path does not exist: {base_path}")
            return None

        # 获取 radioButton 选择
        if self.ui.radioButton_SingleCell.isChecked():
            mode_folder = "radioButton_SingleCell"
        elif self.ui.radioButton_AllPicture.isChecked():
            mode_folder = "radioButton_AllPicture"
        else:
            print("Please select SingleCell or AllPicture mode!")
            return None

        # 获取放大倍数
        if self.ui.radioButton_10X.isChecked():
            magnification_folder = "radioButton_10X"
        elif self.ui.radioButton_20X.isChecked():
            magnification_folder = "radioButton_20X"
        elif self.ui.radioButton_4.isChecked():
            magnification_folder = "radioButton_4"
        else:
            print("Please select a magnification level!")
            return None

        # 获取 comboBox 选择
        selected_cell_type = self.ui.comboBox_CellKind_2.currentText()
        script_file = f"{selected_cell_type}.py"

        # 拼接完整路径
        script_path = os.path.join(base_path, mode_folder, magnification_folder, script_file)
        print(f"Constructed script path: {script_path}")

        # 检查脚本路径是否存在
        if not os.path.isfile(script_path):
            print(f"Script file does not exist: {script_path}")
            return None

        return script_path

    def run_selected_script(self):
        """根据用户选择运行对应脚本"""
        script_path = self.get_script_path()  # 确保路径正确
        model_path = self.ui.LineEdit_openModel_2.text()
        input_folder = self.ui.LineEdit_openInput_2.text()
        output_folder = self.ui.LineEdit_outputs_dir.text()

        # 检查路径有效性
        if not script_path or not os.path.isfile(script_path):
            print("Invalid script path!")
            return
        if not model_path or not os.path.isfile(model_path):
            print("Invalid script path!")
            return
        if not input_folder or not os.path.isdir(input_folder):
            print("Invalid input folder path!")
            return
        if not output_folder or not os.path.isdir(output_folder):
            print("Invalid input folder path!")
            return

        # 在执行脚本之前，检查文件是否为空
        if self.is_script_empty(script_path):
            self.show_error_message("10X detection is not supported for this cell type yet")
            return

        # 定义进度回调函数
        def progress_callback(value):
            """进度回调函数，用于更新进度条"""
            self.update_progress_bar(value)

        # 执行脚本并传递回调函数
        self.execute_script(script_path, model_path, input_folder, output_folder, progress_callback)

        # 弹出完成提示
        self.show_completion_message()

        # 进度条归零
        self.ui.progressBar.setValue(0)

    def is_script_empty(self, script_path):
        """检查脚本文件是否为空"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"Script {script_path} is empty")
                    return True
                # 检查脚本中是否包含主要的检测函数（例如 "main"）
                if 'main' not in content:
                    print(f"Script {script_path} does not define a 'main' function")
                    return True
                return False
        except Exception as e:
            print(f"Failed to read script file: {e}")
            return True

    def show_error_message(self, message):
        """显示错误消息弹窗"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Warning")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def update_progress_bar(self, value):
        """更新进度条并刷新UI"""
        self.ui.progressBar.setValue(value)
        QApplication.processEvents()

    def show_completion_message(self):
        """显示任务完成提示"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Task completed!")
        msg_box.setWindowTitle("Notice")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.show()

        # 自动关闭提示框
        QTimer.singleShot(2000, msg_box.close)  # 2秒后自动关闭

    def open_model_path2(self):
        """选择目标权重文件（*.pt）并显示到 LineEdit_openModel_2 中"""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select model weight file", "", "Model file (*.pt)")
        if model_path:
            self.ui.LineEdit_openModel_2.setText(model_path)

    def open_image_folder2(self):
        """选择目标图片文件夹并显示到 LineEdit_openInput_2 中"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select target image folder")
        if folder_path:
            self.ui.LineEdit_openInput_2.setText(folder_path)

    def open_output_folder(self):
        """选择输出文件夹并显示到 LineEdit_outputs_dir 中"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder_path:
            self.ui.LineEdit_outputs_dir.setText(folder_path)

    def open_model_path(self):
        """选择模型权重文件路径并显示在 LineEdit_openModel"""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select model weight file", "", "Model file (*.pt)")
        if model_path:
            self.ui.LineEdit_openModel.setText(model_path)

    def open_model_path2(self):
        """选择模型权重文件路径并显示在 LineEdit_openModel"""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select model weight file", "", "Model file (*.pt)")
        if model_path:
            self.ui.LineEdit_openModel_2.setText(model_path)

    def open_image_folder(self):
        """选择图片文件夹并显示第一张图片"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if folder_path:
            self.image_dir_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path) if
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            self.image_files.sort()
            self.current_image_index = 0

            if self.image_files:
                first_image_path = os.path.join(folder_path, self.image_files[0])
                self.ui.LineEdit_openInput.setText(first_image_path)
                self.ui.input_detect.setPixmap(QPixmap(first_image_path))
            else:
                print("No image files found in this folder")

    def open_folder(self):
        """通过按钮打开文件夹，并返回路径"""
        # 打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder")

        # 检查用户是否选择了文件夹
        if folder_path:
            self.ui.LineEdit_openInput_2.setText(folder_path)
        else:
            print("No folder selected")
            return None

    def out_folder(self):
        """通过按钮打开文件夹，并返回路径"""
        # 打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder")

        # 检查用户是否选择了文件夹
        if folder_path:
            self.ui.LineEdit_outputs_dir.setText(folder_path)
        else:
            print("No folder selected")
            return None

    def update_code_path(self):
        """更新 YOLOv8 检测代码路径"""
        selected_option = self.ui.comboBox_CellKind.currentText()

        detection_code_map = {
            "Vessel": "./yolo_detect/detect_code/Vessel/vessel_count.py",
            "Fibre": "./yolo_detect/detect_code/Fibre/fibre_count.py",
            "Ray-Cross": "./yolo_detect/detect_code/Ray-Cross/ray-c.py",
            "Ray-Chord": "./yolo_detect/detect_code/Ray-Chord/ray-count.py",
            "Ray-Kind": "./yolo_detect/detect_code/Ray-Chord-Kind/ray-kind.py",
        }
        self.detect_code_path = detection_code_map.get(selected_option, "")

        if not os.path.isfile(self.detect_code_path):
            print("Invalid detection code path")
        else:
            print(f"Detection code path is valid: {self.detect_code_path}")

    def update_conf_label(self, value):
        """更新conf滑条值和显示的label内容"""
        self.ui.label_conf.setText(f"conf: {value}")

    def update_iou_label(self, value):
        """更新IoU滑条值和显示的label内容"""
        self.ui.label_IoU.setText(f"IoU: {value}")

    def update_detection(self):
        """通过滑条实时更新检测结果"""
        conf = self.ui.horizontalSlider_conf.value() / 100.0  # 将滑条值转换为0.01-1.0范围
        iou = self.ui.horizontalSlider_IoU.value() / 100.0  # 将滑条值转换为0.01-1.0范围
        self.update_conf_label(self.ui.horizontalSlider_conf.value())
        self.update_iou_label(self.ui.horizontalSlider_IoU.value())
        self.run_detection(conf, iou)

    def run_detection(self, conf=0.25, iou=0.5):
        """运行检测并显示检测结果"""
        model_path = self.ui.LineEdit_openModel.text()
        image_path = self.ui.LineEdit_openInput.text()
        selected_kind = self.ui.comboBox_CellKind.currentText()  # 获取 comboBox 选择的内容

        # 检查路径有效性
        if not model_path or not os.path.isfile(model_path):
            print("Invalid model weight path, please check if the path is correct")
            return
        if not image_path or not os.path.isfile(image_path):
            print("Invalid image path, please check if the path is correct")
            return
        if not self.detect_code_path or not os.path.isfile(self.detect_code_path):
            print("Invalid detection code path, please check if the path is correct")
            return

        # 显示原图到 input_HPicture，并保持缩放居中显示
        if os.path.exists(image_path):
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qimage).scaled(
                    self.ui.input_HPicture.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ui.input_HPicture.setPixmap(pixmap)
        try:
            # 动态加载 YOLOv8 检测代码
            print(f"Loading detection code module: {self.detect_code_path}")
            spec = importlib.util.spec_from_file_location("detect_module", self.detect_code_path)
            detect_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detect_module)

            # 调用检测函数
            if selected_kind == "Ray-Kind":
                output_image, class_counts = detect_module.detect_objects(image_path, model_path, iou=iou, conf=conf)

                # 显示图片到 input_detect
                height, width, channel = output_image.shape
                bytes_per_line = channel * width
                q_image = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    self.ui.input_detect.size(),  # QLabel 当前大小
                    Qt.KeepAspectRatio,  # 保持长宽比
                    Qt.SmoothTransformation  # 平滑缩放效果
                )
                self.ui.input_detect.setPixmap(scaled_pixmap)

                # 提取图片名称
                image_name = os.path.basename(image_path)

                # 格式化并显示每个类别的检测数量
                result_text = (
                    f"Class 1: {class_counts[0]};\n"
                    f"Class 2: {class_counts[1]};\n"
                    f"Class 3: {class_counts[2]};\n"
                    f"Class 4: {class_counts[3]};"
                )

                # 合并图片名称与检测结果
                output_text = f"{image_name}\n{result_text}"
                self.ui.plainTextEdit_2.setPlainText(output_text)


            else:
                # 其他检测类型逻辑，返回检测图片和对象总数
                output_image, object_count = detect_module.detect_objects(image_path, model_path, iou=iou, conf=conf)

                # 显示图片到 input_detect
                height, width, channel = output_image.shape
                bytes_per_line = channel * width
                q_image = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    self.ui.input_detect.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ui.input_detect.setPixmap(scaled_pixmap)

                # 格式化并显示对象总数
                result_text = f"{selected_kind}-Number: {object_count};"

                # 提取图片名称
                image_name = os.path.basename(image_path)

                # 合并内容并设置到 plainTextEdit_2
                output_text = f"{image_name}\n{result_text}"
                self.ui.plainTextEdit_2.setPlainText(output_text)


        except Exception as e:
            print(f"Detection failed: {e}")

    def save_current_picture(self):
        """保存当前 input_detect 中的图片到指定文件夹，包括支持 TIFF/TIF 格式"""
        pixmap = self.ui.input_detect.pixmap()
        if pixmap:
            # 弹出保存对话框，支持 TIFF/TIF 格式
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save image", "", "Image file (*.png)"
            )
            if save_path:
                # 检查文件扩展名
                extension = os.path.splitext(save_path)[-1].lower()
                if extension in ['.tiff', '.tif']:
                    # 将 QPixmap 转换为 QImage
                    q_image = pixmap.toImage()
                    buffer = q_image.bits().asstring(q_image.byteCount())  # 获取像素数据
                    image = Image.frombytes(
                        "RGBA", (q_image.width(), q_image.height()), buffer
                    )
                    image = image.convert("RGB")  # TIFF 通常使用 RGB 模式
                    image.save(save_path, format="TIFF")  # 使用 Pillow 保存为 TIFF 格式
                    print(f"Image saved in TIFF format: {save_path}")
                else:
                    # 使用 QPixmap.save 保存常规格式
                    pixmap.save(save_path)
                    print(f"Image saved as: {save_path}")
        else:
            print("No image available to save!")

    def show_next_image(self):
        """显示下一张图片"""
        if not self.image_files:
            print("No image folder loaded")
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        next_image_path = os.path.join(self.image_dir_path, self.image_files[self.current_image_index])
        pixmap = QPixmap(next_image_path).scaled(
            self.ui.input_detect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.input_detect.setPixmap(pixmap)
        #self.ui.input_detect.setPixmap(QPixmap(next_image_path))
        self.ui.LineEdit_openInput.setText(next_image_path)

    def show_previous_image(self):
        """显示上一张图片"""
        if not self.image_files:
            print("Image folder not loaded")
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        previous_image_path = os.path.join(self.image_dir_path, self.image_files[self.current_image_index])
        pixmap = QPixmap(previous_image_path).scaled(
            self.ui.input_detect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.input_detect.setPixmap(pixmap)

        #self.ui.input_detect.setPixmap(QPixmap(previous_image_path))
        self.ui.LineEdit_openInput.setText(previous_image_path)

    def open_model_path2(self):
        """选择目标权重文件（*.pt）并显示到 LineEdit_openModel_2 中"""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select model weight file", "", "Model file (*.pt)")
        if model_path:
            self.ui.LineEdit_openModel_2.setText(model_path)

    def open_image_folder2(self):
        """选择目标图片文件夹并显示到 LineEdit_openInput_2 中"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select target image folder")
        if folder_path:
            self.ui.LineEdit_openInput_2.setText(folder_path)

    def open_output_folder(self):
        """选择输出文件夹并显示到 LineEdit_outputs_dir 中"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder_path:
            self.ui.LineEdit_outputs_dir.setText(folder_path)

    def buttonClick(self):
        btn = self.sender()
        btnName = btn.objectName()

        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))
        elif btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        elif btnName == "btn_HandMask":
            widgets.stackedWidget.setCurrentWidget(widgets.HandMask)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        elif btnName == "btn_batch":
            widgets.stackedWidget.setCurrentWidget(widgets.batchdetect)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        elif btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        elif btnName == "btn_save":
            print("Save BTN clicked!")
        elif btnName == "btn_detect2":
            widgets.stackedWidget.setCurrentWidget(widgets.detectCross)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        elif btnName == "ImageStitching":
            widgets.stackedWidget.setCurrentWidget(widgets.ImageStitch)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        elif btnName == "btn_next":
            self.show_next_image()
        elif btnName == "btn_Pre":
            self.show_previous_image()

        print(f'Button "{btnName}" pressed!')

    def resizeEvent(self, event):
        UIFunctions.resize_grips(self)

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def open_concat_left_image(self):
        # 打开文件对话框，选择图像
        filePath, _ = QFileDialog.getOpenFileName(self, dir="./", filter="*.jpg;*.png;*.tiff;*.tif;*.jpeg")
        # 如果选择了文件
        if filePath:
            # 更新左图路径显示
            widgets.LineEdit_image_left.setText(filePath)
            # 使用 QPixmap 打开图片
            pixmap = QPixmap(filePath)
            # 获取 image_left QLabel 的大小
            label_width = widgets.image_left.width()
            label_height = widgets.image_left.height()
            # 使用 scaled() 方法按 QLabel 的大小调整图像，保持长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 设置调整后的图像到 QLabel
            widgets.image_left.setPixmap(scaled_pixmap)

    def open_concat_right_image(self):
        filePath, _ = QFileDialog.getOpenFileName(self, dir="./", filter="*.jpg;*.png;*.tiff;*.tif;*.jpeg")
        # 如果选择了文件
        if filePath:
            # 更新左图路径显示
            widgets.LineEdit_image_right.setText(filePath)
            # 使用 QPixmap 打开图片
            pixmap = QPixmap(filePath)
            # 获取 image_left QLabel 的大小
            label_width = widgets.image_right.width()
            label_height = widgets.image_right.height()
            # 使用 scaled() 方法按 QLabel 的大小调整图像，保持长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 设置调整后的图像到 QLabel
            widgets.image_right.setPixmap(scaled_pixmap)


    def concat_images(self):
        # 获取左边和右边图像的路径
        img1_path = widgets.LineEdit_image_left.text()
        img2_path = widgets.LineEdit_image_right.text()

        # 打印路径，检查路径是否正确
        print(f"Left image path: {img1_path}")
        print(f"Right image path: {img2_path}")

        # 替换反斜杠为正斜杠
        img1_path = img1_path.replace("\\", "/")
        img2_path = img2_path.replace("\\", "/")

        # 检查文件是否存在
        if not os.path.exists(img1_path):
            print(f"File 1 does not exist: {img1_path}")
            QMessageBox.critical(self, "Error", f"File 1 does not exist: {img1_path}")
            return
        if not os.path.exists(img2_path):
            print(f"File 2 does not exist: {img2_path}")
            QMessageBox.critical(self, "Error", f"File 2 does not exist: {img2_path}")
            return

        # 定义一个函数来读取中文路径的图像文件，使用 imdecode
        def cv_imread_chinese(path):
            # 使用文件系统编码方式处理中文路径
            # Python3 默认使用 Unicode 字符串，所以直接读取
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                img_array = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
            except Exception as e:
                print(f"Error reading image: {e}")
                return None

        # 使用 OpenCV 直接读取图像
        # 不使用 Pillow，以保持与原始代码一致的图像数据和像素信息
        try:
            img1 = cv_imread_chinese(img1_path)
            img2 = cv_imread_chinese(img2_path)
        except Exception as e:
            print(f"Unable to read image: {e}")
            QMessageBox.critical(self, "Error", f"Unable to read image: {e}")
            return

        # 再次检查图像是否成功读取
        if img1 is None:
            print(f"Unable to read image 1: {img1_path}")
            QMessageBox.critical(self, "Error", f"Unable to read image 1: {img1_path}")
            return
        if img2 is None:
            print(f"Unable to read image 2: {img2_path}")
            QMessageBox.critical(self, "Error", f"Unable to read image 2: {img2_path}")
            return

        print("Image successfully loaded.")

        # 创建 Stitcher 对象进行全景拼接
        try:
            stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        except AttributeError:
            # 兼容旧版本 OpenCV
            stitcher = cv2.createStitcher(False)

        # 执行拼接
        status, pano = stitcher.stitch([img1, img2])

        # 判断拼接是否成功
        if status != cv2.Stitcher_OK:
            print(f"Cannot stitch images, error code = {status}")
            QMessageBox.critical(self, "Error", f"Cannot stitch images, error code = {status}")
            return

        print("Stitching successful.")

        # 存储拼接后的图像到实例变量，以便保存
        self.stitched_image = pano  # 原始拼接结果，保证像素质量不受影响

        # 转换拼接图像为 QPixmap 显示在 QLabel 上
        try:
            pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            h, w, c = pano_rgb.shape
            bytes_per_line = c * w
            qimg = QImage(pano_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            if pixmap.isNull():
                print("Unable to load image into QPixmap")
                QMessageBox.critical(self, "Error", "Unable to load stitched image into display area.")
                return

            # 获取 image_stitch QLabel 的大小
            label_width = widgets.image_stitch.width()
            label_height = widgets.image_stitch.height()

            # 使用 scaled() 方法按 QLabel 的大小调整图像显示，但这不影响 self.stitched_image 的原图分辨率
            scaled_pixmap = pixmap.scaled(
                label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # 设置调整后的图像到 QLabel
            widgets.image_stitch.setPixmap(scaled_pixmap)

            print("Stitched image displayed.")
            QMessageBox.information(self, "Success", "Image stitching successful.")
        except Exception as e:
            print(f"Error displaying stitched image: {e}")
            QMessageBox.critical(self, "Error", f"Error displaying stitched image: {e}")

    def save_concat_picture(self):
        # 确保 stitched_image 已存在
        if not hasattr(self, 'stitched_image') or self.stitched_image is None:
            print("没有图像可以保存")
            QMessageBox.warning(self, "Warning", "No image available to save.")
            return

        # 打开文件保存对话框，选择保存路径和文件格式
        filePath, _ = QFileDialog.getSaveFileName(
            self,
            "保存图像",
            "",
            "PNG 文件 (*.png);;JPEG 文件 (*.jpg *.jpeg);;TIFF 文件 (*.tiff *.tif)"
        )

        # 如果选择了路径
        if filePath:
            try:
                # 确保目录存在
                output_dir = os.path.dirname(filePath)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 获取文件扩展名
                ext = os.path.splitext(filePath)[1].lower()

                # 设置编码参数
                if ext in ['.jpg', '.jpeg']:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                elif ext in ['.png']:
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
                elif ext in ['.tiff', '.tif']:
                    encode_param = []  # 默认参数
                else:
                    QMessageBox.critical(self, "错误", f"不支持的文件格式: {ext}")
                    return

                # 使用 imencode 进行编码
                if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                    result, encoded_img = cv2.imencode(ext, self.stitched_image, encode_param)
                    if not result:
                        print("Failed to encode image")
                        QMessageBox.critical(self, "Error", "Failed to encode image.")
                        return

                    # 将编码后的图像写入文件
                    with open(filePath, 'wb') as f:
                        f.write(encoded_img.tobytes())

                    print(f"Image saved to: {filePath}")
                    QMessageBox.information(self, "Success", f"Image saved to: {filePath}")
                else:
                    QMessageBox.critical(self, "Error", f"Unsupported file format: {ext}")
            except Exception as e:
                print(f"Error saving image: {e}")
                QMessageBox.critical(self, "Error", f"An error occurred while saving the image: {e}")


if __name__ == "__main__":
    # ✅ PyInstaller/Windows 必需：避免子进程把整个 exe 再拉起一次
    import multiprocessing as mp
    mp.freeze_support()

    # ✅ 只创建一个 QApplication（源码/打包都安全）
    app = QApplication.instance() or QApplication(sys.argv)

    # （可选）如果目标机没有你的 ico，可以用 APP_ROOT 拼绝对路径更稳
    # from pathlib import Path
    # icon_path = (APP_ROOT / "40x40_icon_final.ico")
    # if icon_path.exists():
    #     app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    sys.exit(app.exec())

