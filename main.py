from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QSlider,
    QHBoxLayout,
    QGroupBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon
import sys
import os
from glob import glob
from sl0thifier.logger import logger


class DropLabel(QLabel):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and path.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):
                paths.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            paths.append(os.path.join(root, f))
        self.files_dropped.emit(paths)


class Worker(QObject):
    finished = Signal(str)
    progress = Signal(int)

    def __init__(
        self,
        img_path,
        output_path,
        width,
        height,
        remove_bg,
        bg_color,
        model_name,
        clip_limit,
        tile_size,
    ):
        super().__init__()
        self.img_path = img_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.remove_bg = remove_bg
        self.bg_color = bg_color
        self.model_name = model_name
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def run(self):
        from sl0thify import KingSl0th
        from PIL import Image
        import traceback

        try:
            logger.info("[Worker] Processing: %s", self.img_path)
            image = Image.open(self.img_path)
            image.load()

            king = KingSl0th()

            result_image = king.sl0thify(
                img=image,
                model_name=self.model_name,
                clip_limit=self.clip_limit,
                tile_size=self.tile_size,
                output_width=self.width,
                output_height=self.height,
                remove_bg=self.remove_bg,
                bg_color=self.bg_color,
            )

            name = os.path.splitext(os.path.basename(self.img_path))[0]
            save_path = os.path.join(self.output_path, f"{name}_sl0thified.png")
            result_image.save(save_path)
            logger.info("✅ Output saved to: %s", save_path)

        except Exception as e:
            logger.error(
                "[Worker][ERROR] Failed to sl0thify '%s': %s", self.img_path, e
            )
            traceback.print_exc()

        finally:
            self.progress.emit(100)
            self.finished.emit(self.img_path)


class Sl0thifierGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("sl0thifier 🦥")
        icon_path = os.path.join(".", "sl0thifier", "assets", "sl0thm4n.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setMinimumSize(400, 300)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.selected_width = 512
        self.selected_height = 512
        self.threads = []
        self.workers = []

        self.label = DropLabel()
        self.label.setText("Drop images or folders here")
        self.label.setStyleSheet(
            "font-size: 16px; border: 2px dashed gray; padding: 40px;"
        )
        self.label.setMinimumHeight(100)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.files_dropped.connect(self.handle_dropped_files)
        self.layout.addWidget(self.label)

        self.remove_bg_checkbox = QCheckBox("Remove Background")
        self.remove_bg_checkbox.stateChanged.connect(self.toggle_bg_color_select)
        self.layout.addWidget(self.remove_bg_checkbox)

        self.bg_color_label = QLabel("New Background Color:")
        self.bg_color_select = QComboBox()
        self.bg_color_select.addItems(["None", "White", "Black", "Green"])
        self.bg_color_select.setEnabled(False)
        self.layout.addWidget(self.bg_color_label)
        self.layout.addWidget(self.bg_color_select)

        # Group: Enhancement Settings
        self.enhance_group = QGroupBox("Enhancement Settings")
        self.enhance_layout = QVBoxLayout()

        self.model_select = QComboBox()
        model_names = self.load_models()
        self.model_select.addItems(model_names)
        self.enhance_layout.addWidget(QLabel("Upscaler Model:"))
        self.enhance_layout.addWidget(self.model_select)

        self.clip_slider_label = QLabel("CLAHE Clip Limit: 1.0")
        self.clip_slider = QSlider(Qt.Horizontal)
        self.clip_slider.setMinimum(1)
        self.clip_slider.setMaximum(40)
        self.clip_slider.setValue(10)
        self.clip_slider.valueChanged.connect(self.update_clip_limit_label)
        self.enhance_layout.addWidget(self.clip_slider_label)
        self.enhance_layout.addWidget(self.clip_slider)

        self.tile_slider_label = QLabel("CLAHE Tile Size: 4")
        self.tile_slider = QSlider(Qt.Horizontal)
        self.tile_slider.setMinimum(2)
        self.tile_slider.setMaximum(16)
        self.tile_slider.setValue(4)
        self.tile_slider.valueChanged.connect(self.update_tile_size_label)
        self.enhance_layout.addWidget(self.tile_slider_label)
        self.enhance_layout.addWidget(self.tile_slider)

        self.enhance_group.setLayout(self.enhance_layout)
        self.layout.addWidget(self.enhance_group)

        self.size_select = QComboBox()
        self.size_select.addItems(["512 x 512", "1024 x 1024"])
        self.size_select.currentIndexChanged.connect(self.update_output_size)
        self.layout.addWidget(QLabel("Output Size:"))
        self.layout.addWidget(self.size_select)

        self.choose_button = QPushButton("Choose Output Folder")
        self.choose_button.clicked.connect(self.select_output_dir)
        self.layout.addWidget(self.choose_button)

        self.start_button = QPushButton("Sl0thify Now!")
        self.start_button.setStyleSheet(
            "font-size: 16px; padding: 12px; background-color: #88cc88; font-weight: bold;"
        )
        self.start_button.clicked.connect(self.sl0thify_images)
        self.layout.addWidget(self.start_button)

    def update_clip_limit_label(self, value):
        real_value = value / 10.0
        self.clip_slider_label.setText(f"CLAHE Clip Limit: {real_value:.1f}")

    def update_tile_size_label(self, value):
        self.tile_slider_label.setText(f"CLAHE Tile Size: {value}")

    def load_models(self):
        model_dir = os.path.join(".", "realesrgan", "models")
        if not os.path.isdir(model_dir):
            return []

        bin_files = glob(os.path.join(model_dir, "*.bin"))
        param_files = glob(os.path.join(model_dir, "*.param"))

        bin_names = {os.path.splitext(os.path.basename(f))[0] for f in bin_files}
        param_names = {os.path.splitext(os.path.basename(f))[0] for f in param_files}

        model_names = sorted(bin_names & param_names)
        logger.info("[Model Loader] Found models: %s", model_names)
        return model_names

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path = path
            self.label.setText(f"Output folder selected:\n{path}")

    def toggle_bg_color_select(self):
        self.bg_color_select.setEnabled(self.remove_bg_checkbox.isChecked())

    def handle_dropped_files(self, paths):
        self.selected_paths = paths
        self.label.setText(f"{len(self.selected_paths)} file(s) ready to sl0thify")

    def update_output_size(self):
        size = self.size_select.currentText()
        width, height = size.split(" x ")
        self.selected_width = int(width)
        self.selected_height = int(height)

    def start_thread(self, img_path):
        model_name = self.model_select.currentText()
        clip_limit = self.clip_slider.value() / 10.0
        tile_size = self.tile_slider.value()

        thread = QThread()
        worker = Worker(
            img_path=img_path,
            output_path=self.output_path,
            width=self.selected_width,
            height=self.selected_height,
            remove_bg=self.remove_bg_checkbox.isChecked(),
            bg_color=self.bg_color_select.currentText(),
            model_name=model_name,
            clip_limit=clip_limit,
            tile_size=tile_size,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.threads.append(thread)
        self.workers.append(worker)

        thread.start()

    def on_done(self, img_path):
        logger.info("[GUI] Sl0thified: %s", img_path)
        self.label.setText(f"Sl0thified: {os.path.basename(img_path)}")

    def sl0thify_images(self):
        if not hasattr(self, "selected_paths") or not self.selected_paths:
            self.label.setText("No images dropped.")
            return

        if not hasattr(self, "output_path") or not self.output_path:
            self.label.setText("No valid output folder selected.")
            return

        for img_path in self.selected_paths:
            self.start_thread(img_path)

    def closeEvent(self, event):
        logger.info("[GUI] Closing: Waiting for threads to finish...")
        for thread in self.threads:
            if thread.isRunning():
                thread.quit()
                thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Sl0thifierGUI()
    window.show()
    sys.exit(app.exec())
