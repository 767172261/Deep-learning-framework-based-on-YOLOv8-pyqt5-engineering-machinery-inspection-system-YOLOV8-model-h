**深度学习框架基于YOLOv8➕pyqt5工程机械检测系统，**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f33ba498a4c243b796b3bcccb3105e10.png)

2655张工程机械数据集
包括['挖掘机', '自卸卡车', '轮式装载机']，3类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ed73349c2bf04cb2877e79233375b421.png)


也可自行替换模型，使用该界面做其他检测

以下是完整的 **基于 YOLOv8 + PyQt5 的工程机械检测系统**，支持：

- 3 类目标检测：`挖掘机`、`自卸卡车`、`轮式装载机`
- 图像 / 视频 / 摄像头实时检测
- 可视化检测框、类别（中文）、置信度、坐标
- 支持替换任意 YOLOv8 模型（如 `best.pt`）
- 保存检测结果为文本报告
- 界面简洁、功能完整、代码可直接运行

---

## ✅ 一、环境依赖

```bash
pip install ultralytics pyqt5 opencv-python numpy
```

> - `ultralytics`：官方 YOLOv8 库  
> - `PyQt5`：图形界面  
> - `OpenCV`：图像/视频处理

---

## ✅ 二、数据集说明（供训练参考）

| 属性 | 内容 |
|------|------|
| 数据集名称 | Construction Machinery Detection Dataset |
| 图像总数 | 2655 张 |
| 类别数量 | 3 类 |
| 类别名称 | `挖掘机`、`自卸卡车`、`轮式装载机` |
| 标注格式 | YOLO（`.txt`） |
| 推荐划分 | train: ~2100, val: ~400, test: ~155 |

> 📂 示例结构：
```
construction_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## ✅ 三、完整系统代码：`construction_detection.py`

```python
# construction_detection.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QTextEdit, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from ultralytics import YOLO


class ConstructionMachineryDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv8的工程机械检测系统")
        self.setGeometry(100, 100, 1200, 700)

        # 加载模型（可替换为你的 best.pt）
        self.model = None
        self.load_model()

        # 初始化 UI
        self.init_ui()

        # 摄像头/视频控制
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_model(self):
        """加载 YOLOv8 模型"""
        try:
            # 替换为你训练好的模型路径，例如 'runs/detect/train/weights/best.pt'
            self.model = YOLO('yolov8n.pt')  # 默认使用 yolov8n，可改为 best.pt
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 标题
        title = QLabel("<h2>工程机械智能检测系统</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 主体区域
        main_layout = QHBoxLayout()

        # 左侧：图像显示区
        self.image_label = QLabel("点击“打开图片”上传图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        self.image_label.setMinimumSize(600, 400)
        main_layout.addWidget(self.image_label)

        # 右侧：控制面板
        right_panel = QVBoxLayout()

        # 参数设置
        right_panel.addWidget(QLabel("检测参数"))
        self.conf_spin = QSpinBox()
        self.conf_spin.setValue(25)
        self.conf_spin.setSuffix("%")
        self.conf_spin.setMaximum(100)
        right_panel.addWidget(QLabel("置信度阈值"))
        right_panel.addWidget(self.conf_spin)

        self.iou_spin = QSpinBox()
        self.iou_spin.setValue(45)
        self.iou_spin.setSuffix("%")
        self.iou_spin.setMaximum(100)
        right_panel.addWidget(QLabel("IoU 阈值"))
        right_panel.addWidget(self.iou_spin)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_image = QPushButton("打开图片")
        self.btn_video = QPushButton("打开视频")
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_save = QPushButton("保存结果")
        self.btn_exit = QPushButton("退出")

        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_exit.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_image)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_exit)
        right_panel.addLayout(btn_layout)

        # 检测结果显示
        right_panel.addWidget(QLabel("检测信息"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        right_panel.addWidget(self.result_text)

        main_layout.addLayout(right_panel)
        layout.addLayout(main_layout)

        # 结果表格（下方）
        layout.addWidget(QLabel("检测结果明细"))
        self.table_text = QTextEdit()
        self.table_text.setReadOnly(True)
        layout.addWidget(self.table_text)

        central_widget.setLayout(layout)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.jpg *.png *.jpeg)")
        if file_name:
            self.detect_and_show(file_name, is_video=False)

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.timer.start(30)
            self.result_text.setText("正在播放视频...")

    def open_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.result_text.setText("正在使用摄像头...")

    def detect_and_show(self, source, is_video=False):
        if self.model is None:
            self.result_text.setText("❌ 模型未加载！")
            return

        img = cv2.imread(source) if not is_video else source
        if img is None:
            self.result_text.setText("❌ 图像读取失败！")
            return

        conf = self.conf_spin.value() / 100.0
        iou = self.iou_spin.value() / 100.0

        results = self.model(img, conf=conf, iou=iou)
        annotated_img = results[0].plot()

        # 显示图像
        self.display_image(annotated_img)

        # 显示结果
        self.show_results(results[0], source)

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.result_text.setText("❌ 视频/摄像头结束！")
            return
        self.detect_and_show(frame, is_video=True)

    def display_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))

    def show_results(self, result, source_path):
        boxes = result.boxes.cpu().numpy()
        class_names = ['挖掘机', '自卸卡车', '轮式装载机']  # 中文映射（按训练时类别顺序）

        if len(boxes) == 0:
            self.result_text.setText("🔍 未检测到工程机械。")
            self.table_text.setText("")
            return

        # 构建文本
        text = f"🎯 检测完成！共 {len(boxes)} 个目标\n"
        table_lines = ["序号\t类别\t置信度\t坐标 [x1,y1,x2,y2]"]

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].astype(int)
            label = class_names[cls_id] if cls_id < len(class_names) else f"未知类({cls_id})"
            conf_str = f"{conf*100:.2f}%"
            coord_str = f"[{xyxy[0]},{xyxy[1]},{xyxy[2]},{xyxy[3]}]"
            text += f"• {label} ({conf_str})\n"
            table_lines.append(f"{i+1}\t{label}\t{conf_str}\t{coord_str}")

        self.result_text.setText(text)
        self.table_text.setText("\n".join(table_lines))

    def save_results(self):
        content = self.result_text.toPlainText() + "\n\n" + self.table_text.toPlainText()
        if not content.strip():
            self.result_text.setText("❌ 无结果可保存！")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.result_text.setText("✅ 结果已保存！")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ConstructionMachineryDetectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
```

---

## ✅ 四、如何使用？

### 1. 替换为你自己的模型
将以下行：
```python
self.model = YOLO('yolov8n.pt')
```
改为：
```python
self.model = YOLO('path/to/your/best.pt')
```

> ⚠️ 注意：`best.pt` 必须是在 **相同类别顺序** 下训练的（即：0=挖掘机, 1=自卸卡车, 2=轮式装载机）

### 2. 运行程序
```bash
python construction_detection.py
```

---

## ✅ 五、训练建议（如需训练）

### `data.yaml` 配置文件：
```yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 3
names: ['excavator', 'dump_truck', 'wheel_loader']
```

> 训练时使用英文名，推理时在 GUI 中映射为中文。

### 训练命令：
```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16
```

---

## ✅ 六、扩展能力

- ✅ **通用性**：只需修改 `class_names` 和模型路径，即可用于其他检测任务（如车辆、动物、缺陷等）
- ✅ **部署友好**：可打包为 `.exe`（使用 PyInstaller）
- ✅ **多语言支持**：轻松添加英文/其他语言标签

---

> 💡 **提示**：若你的模型输出是英文类别（如 `'excavator'`），可在 `show_results()` 中加入映射字典：
```python
en_to_cn = {
    'excavator': '挖掘机',
    'dump_truck': '自卸卡车',
    'wheel_loader': '轮式装载机'
}
label = en_to_cn.get(predicted_name, predicted_name)
```

---


