import sys
import cv2
import numpy as np
import random # Needed for salt and pepper noise
import traceback # Added for debugging
# Importaciones PyQt5: Aseguramos que están QDialog, QComboBox y otros que usamos
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,
                             QWidget, QFileDialog, QMessageBox, QHBoxLayout,
                             QSlider, QCheckBox, QGroupBox, QSizePolicy,
                             QSplitter, QLineEdit, QFormLayout, QRadioButton,
                             QDialog, QComboBox, QGridLayout) # Añadidos QDialog, QComboBox, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QFont, QDoubleValidator
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QThread, QMutex

# Importaciones para Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure # Necesario para crear la figura del gráfico
import matplotlib.pyplot as plt # Necesario para plotear

# Desactivar barras de herramientas por defecto de Matplotlib
plt.rcParams["toolbar"] = "None"

# Importar DeepLab, Mask R-CNN y Torch
try:
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision import transforms
    from PIL import Image
    DEEPLAB_AVAILABLE = True
    MASKRCNN_AVAILABLE = True
except ImportError:
    DEEPLAB_AVAILABLE = False
    MASKRCNN_AVAILABLE = False
    print("Advertencia: Las librerías de DeepLab, Mask R-CNN o Torch no están instaladas.")
    print("La funcionalidad de segmentación con DeepLab o Mask R-CNN no estará disponible.")

# Intenta importar scipy, necesario para el filtro de moda (Aunque no se usa en APP1, se mantiene por si acaso)
try:
    from scipy import ndimage, stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # El mensaje de advertencia ahora se mostrará al intentar usar el filtro de Moda
    # print("Advertencia: SciPy no está instalado. El filtro de Moda no funcionará.")
    # print("Instálalo con: pip install scipy")


# Importar YOLO, con manejo de error si la librería no está instalada
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Advertencia: La librería 'ultralytics' no está instalada.")
    print("La funcionalidad de segmentación YOLOv8 no estará disponible.")
    # print("Instala las librerías necesarias con: pip install ultralytics torch torchvision opencv-python PyQt5") # Moved to UI message

# Importar Detectron2 y Torch, con manejo de error si las librerías no están instaladas
try:
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.model_zoo import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
    DETECTRON_AVAILABLE = True
except ImportError:
    DETECTRON_AVAILABLE = False
    print("Advertencia: Las librerías de Detectron2 o Torch no están instaladas.")
    print("La funcionalidad de segmentación con Detectron2 no estará disponible.")
    # print("Instala las librerías necesarias. Puedes seguir las instrucciones de instalación de Detectron2:") # Moved to UI message
    # print("https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md")
    # Asegúrate de instalar PyTorch compatible con tu configuración (CUDA si tienes GPU): https://pytorch.org/get-started/locally/


# --- Clase para la Ventana de Histograma (AVANZADA) ---
class HistogramWindow(QDialog): # Heredamos de QDialog para que sea una ventana separada
    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram Window") # Título por defecto
        self.setGeometry(200, 200, 750, 550) # Ajustar tamaño para los controles

        self.image_data = image_data
        # Crear la figura y el canvas de Matplotlib
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        # --- Layout de Controles del Histograma ---
        controls_layout = QVBoxLayout()

        # Checkboxes para controlar la visibilidad de canales/luminancia
        histogram_checkbox_layout = QHBoxLayout()
        self.cb_red = QCheckBox("Red")
        self.cb_red.setChecked(True)
        self.cb_green = QCheckBox("Green")
        self.cb_green.setChecked(True)
        self.cb_blue = QCheckBox("Blue")
        self.cb_blue.setChecked(True)
        self.cb_luminance = QCheckBox("Luminance")
        self.cb_luminance.setChecked(True)

        histogram_checkbox_layout.addWidget(self.cb_red)
        histogram_checkbox_layout.addWidget(self.cb_green)
        histogram_checkbox_layout.addWidget(self.cb_blue)
        histogram_checkbox_layout.addWidget(self.cb_luminance)
        histogram_checkbox_layout.addStretch()
        controls_layout.addLayout(histogram_checkbox_layout)

        # Controles Adicionales (Tipo de Plot, Valores, Grid, Etiquetas Ejes, Límite Y, Guardar)
        additional_controls_layout = QHBoxLayout()

        # Checkbox para alternar entre Líneas y Barras
        self.cb_show_bars = QCheckBox("Show as Bars")
        self.cb_show_bars.setChecked(False) # Por defecto Líneas

        # Checkbox para mostrar valores numéricos (tiene sentido en barras)
        self.cb_show_values = QCheckBox("Show Values")
        self.cb_show_values.setChecked(False) # Por defecto no mostrar valores
        # El estado habilitado se establecerá en update_plot basado en cb_show_bars

        # Checkboxes para Grid y Etiquetas de Ejes
        self.cb_grid = QCheckBox("Show Grid")
        self.cb_grid.setChecked(True) # Por defecto mostrar grid
        self.cb_axis_labels = QCheckBox("Show Axis Labels")
        self.cb_axis_labels.setChecked(True) # Por defecto mostrar etiquetas

        additional_controls_layout.addWidget(self.cb_show_bars)
        additional_controls_layout.addWidget(self.cb_show_values)
        additional_controls_layout.addWidget(self.cb_grid)
        additional_controls_layout.addWidget(self.cb_axis_labels)

        # ComboBox para el Límite del Eje Y con valores escalonados
        self.y_limit_label = QLabel("Y Max:")
        self.y_limit_combobox = QComboBox()
        # Poblamos el ComboBox con los valores especificados
        y_values = []
        y_values.extend(range(100, 1001, 100)) # 100, 200, ..., 1000 (pasos de 100)
        y_values.extend(range(2000, 10001, 1000)) # 2000, 3000, ..., 10000 (pasos de 1000)
        y_values.extend(range(20000, 100001, 10000)) # 20000, 30000, ..., 100000 (pasos de 10000)

        # Si el número total de píxeles es muy grande, añadimos valores más altos al ComboBox
        if image_data is not None and len(image_data.shape) > 0:
             total_pixels = image_data.shape[0] * image_data.shape[1]
             if total_pixels > 100000:
                 step = 50000 if total_pixels < 500000 else 100000
                 for val in range(110000, total_pixels + step, step):
                     if val > y_values[-1]: # Asegurarse de no duplicar y añadir en orden ascendente
                         y_values.append(val)

        # Asegurarse de que los valores sean únicos y estén ordenados
        y_values = sorted(list(set(y_values)))


        # Añadir los valores al ComboBox como cadenas
        self.y_limit_combobox.addItems([str(val) for val in y_values])

        # Botón Guardar
        self.save_button = QPushButton("Guardar Histograma")
        self.save_button.clicked.connect(self.save_histogram_plot)

        additional_controls_layout.addWidget(self.y_limit_label)
        additional_controls_layout.addWidget(self.y_limit_combobox)
        additional_controls_layout.addWidget(self.save_button)
        additional_controls_layout.addStretch()

        controls_layout.addLayout(additional_controls_layout)
        # --- Fin Layout de Controles del Histograma ---

        # Conectar señales a update_plot
        self.cb_red.stateChanged.connect(self.update_plot)
        self.cb_green.stateChanged.connect(self.update_plot)
        self.cb_blue.stateChanged.connect(self.update_plot)
        self.cb_luminance.stateChanged.connect(self.update_plot)
        self.cb_show_bars.stateChanged.connect(self.update_plot)
        self.cb_show_values.stateChanged.connect(self.update_plot)
        self.cb_grid.stateChanged.connect(self.update_plot)
        self.cb_axis_labels.stateChanged.connect(self.update_plot)
        self.y_limit_combobox.currentTextChanged.connect(self.update_plot) # Conectar ComboBox

        # Layout principal de la Ventana del Histograma
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(controls_layout)

        # Calcular y mostrar al inicio
        self.calculate_histograms()
        self.set_initial_y_limit() # Ajustar el valor inicial del ComboBox
        self.update_plot()

    # Calcular histogramas
    def calculate_histograms(self):
        self.hist_blue = self.hist_green = self.hist_red = self.hist_luminance = None
        is_color_image = False

        if self.image_data is not None and len(self.image_data.shape) > 0:
            if len(self.image_data.shape) >= 3: # Imagen a color (3 o 4 canales)
                is_color_image = True
                # Para calcular histogramas de color, necesitamos BGR (sin alfa)
                if self.image_data.shape[2] == 4:
                    img_bgr = cv2.cvtColor(self.image_data, cv2.COLOR_BGRA2BGR)
                else:
                    img_bgr = self.image_data

                self.hist_blue = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])
                self.hist_green = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])
                self.hist_red = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])

                # Histograma de luminancia (escala de grises)
                gray_image = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
                self.hist_luminance = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

            elif len(self.image_data.shape) == 2: # Imagen en escala de grises (1 canal)
                self.hist_luminance = cv2.calcHist([self.image_data], [0], None, [256], [0, 256])

        # Habilitar/deshabilitar checkboxes de color
        self.cb_red.setEnabled(is_color_image)
        self.cb_green.setEnabled(is_color_image)
        self.cb_blue.setEnabled(is_color_image)
        self.cb_luminance.setEnabled(self.hist_luminance is not None)


    # Establecer el límite Y inicial automáticamente (ajustada para ComboBox)
    def set_initial_y_limit(self):
        max_count = 0
        # Encontrar el máximo entre todos los histogramas calculados
        if self.hist_red is not None:
             max_count = max(max_count, np.max(self.hist_red))
        if self.hist_green is not None:
             max_count = max(max_count, np.max(self.hist_green))
        if self.hist_blue is not None:
             max_count = max(max_count, np.max(self.hist_blue))
        if self.hist_luminance is not None:
             max_count = max(max_count, np.max(self.hist_luminance))

        if max_count > 0:
            # Encontrar el valor en la lista del ComboBox que sea igual o inmediatamente superior al max_count
            initial_limit_target = int(max_count * 1.05) # Comenzamos con un pequeño margen
            selected_index = 0 # Índice por defecto

            for i in range(self.y_limit_combobox.count()):
                item_value = int(self.y_limit_combobox.itemText(i))
                if item_value >= initial_limit_target:
                    selected_index = i
                    break
                # Si llegamos al final de la lista y ningún valor es mayor o igual, seleccionamos el último
                if i == self.y_limit_combobox.count() - 1:
                    selected_index = i

            self.y_limit_combobox.setCurrentIndex(selected_index)

        else:
            # Si no hay datos de histograma, seleccionar un valor bajo por defecto (e.g., 1000) si está en la lista
            default_index = self.y_limit_combobox.findText("1000")
            if default_index != -1:
                self.y_limit_combobox.setCurrentIndex(default_index)
            else:
                 # Si 1000 no está en la lista, seleccionar el primer valor disponible
                 if self.y_limit_combobox.count() > 0:
                     self.y_limit_combobox.setCurrentIndex(0)


    # Actualizar el plot basado en los controles y datos
    def update_plot(self):
        self.axes.clear() # Limpiar el plot anterior

        show_bars = self.cb_show_bars.isChecked()
        show_values = self.cb_show_values.isChecked()
        show_grid = self.cb_grid.isChecked()
        show_axis_labels = self.cb_axis_labels.isChecked()

        # Habilitar/deshabilitar "Show Values" basado en "Show as Bars"
        self.cb_show_values.setEnabled(show_bars)

        bin_edges = np.arange(256).reshape(-1, 1) # Posiciones del eje X (0 a 255)

        # Plotear histogramas solo si existen y los checkboxes están marcados
        if self.hist_blue is not None and self.cb_blue.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_blue.flatten(), color='blue', alpha=0.7, width=1, label='Blue')
                if show_values:
                    # Mostrar valores solo si están marcados y hay barras
                    for i, count in enumerate(self.hist_blue.flatten()):
                        if count > 0: # Solo mostrar valores > 0
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='blue')
            else:
                self.axes.plot(self.hist_blue, color='blue', label='Blue')

        if self.hist_green is not None and self.cb_green.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_green.flatten(), color='green', alpha=0.7, width=1, label='Green')
                if show_values:
                     for i, count in enumerate(self.hist_green.flatten()):
                         if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='green')
            else:
                self.axes.plot(self.hist_green, color='green', label='Green')

        if self.hist_red is not None and self.cb_red.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_red.flatten(), color='red', alpha=0.7, width=1, label='Red')
                if show_values:
                     for i, count in enumerate(self.hist_red.flatten()):
                         if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='red')
            else:
                self.axes.plot(self.hist_red, color='red', label='Red')

        if self.hist_luminance is not None and self.cb_luminance.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_luminance.flatten(), color='gray', alpha=0.7, width=1, label='Luminance')
                if show_values:
                     for i, count in enumerate(self.hist_luminance.flatten()):
                         if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='gray')
            else:
                self.axes.plot(self.hist_luminance, color='gray', label='Luminance')

        # Configurar Grid y Etiquetas de Ejes
        self.axes.grid(show_grid) # Mostrar grid si está marcado

        if show_axis_labels:
            self.axes.set_title("Histograma de Canales y Luminancia")
            self.axes.set_xlabel("Nivel de Intensidad")
            self.axes.set_ylabel("Cantidad de Pixeles")
            self.axes.tick_params(axis='x', labelbottom=True)
            self.axes.tick_params(axis='y', labelleft=True)
            # Mostrar leyenda si hay elementos ploteados con etiquetas
            handles, labels = self.axes.get_legend_handles_labels()
            if handles:
                self.axes.legend()
        else:
            # Ocultar título, etiquetas y leyenda si no están marcados
            self.axes.set_title("")
            self.axes.set_xlabel("")
            self.axes.set_ylabel("")
            self.axes.tick_params(axis='x', labelbottom=False)
            self.axes.tick_params(axis='y', labelleft=False)
            if self.axes.get_legend():
                self.axes.get_legend().remove()


        # Establecer el límite del eje Y desde el ComboBox
        y_limit_str = self.y_limit_combobox.currentText()
        if y_limit_str: # Asegurarse de que haya texto seleccionado
            y_limit = int(y_limit_str)
            self.axes.set_ylim(0, y_limit) # Establecer límite Y
        self.axes.set_xlim(-1, 256) # Establecer límite X para ver todos los bins

        self.canvas.draw() # Redibujar el canvas

    # Función para guardar el plot del histograma
    def save_histogram_plot(self):
        # Asegurarse de que haya datos de histograma para guardar
        if self.hist_luminance is not None or self.hist_red is not None:
            file_dialog = QFileDialog()
            filepath, _ = file_dialog.getSaveFileName(self, "Guardar Histograma", "", "Archivos de Imagen (*.png *.jpg *.jpeg)")

            if filepath:
                try:
                    # Guardar la figura de Matplotlib
                    self.figure.savefig(filepath)
                    print(f"Histograma guardado en: {filepath}")
                except Exception as e:
                    print(f"Error al guardar el histograma: {e}")

# --- Fin Clase para la Ventana de Histograma (AVANZADA) ---


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self._histogram_window = None
        # Variables críticas de estado (deben existir antes de cualquier método de UI)
        self.image = None
        self.yolo_model = None
        self.yolo_results = {}
        self.yolo_counts = {}
        self.detectron_model = None
        self.detectron_results = {}
        self.detectron_counts = {}
        self.deeplab_model = None
        self.deeplab_results = {}
        self.deeplab_counts = {}
        self.maskrcnn_model = None
        self.maskrcnn_results = {}
        self.maskrcnn_counts = {}
        self._current_display_image = None
        # Clases objetivo COCO para segmentación (Detectron2 y YOLO)
        self.target_classes = [0, 2, 3, 9]  # person=0, car=2, motorcycle=3, traffic light=9
        # Labels principales
        self.processed_label = QLabel(self)
        self.processed_label.setText(
            "<div style='font-size:22pt; color:#2196F3; font-weight:bold; text-shadow: 1px 1px 8px #90caf9;'>🪐 Bienvenido a la Plataforma de Segmentación Futurista 🪐<br><span style='font-size:13pt;'>Carga una imagen y explora DeepLab, Mask R-CNN y más</span></div>"
        )
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setScaledContents(True)
        self.processed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_label.setProperty("cssClass", "image-display")
        self.processed_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e3f2fd, stop:1 #bbdefb); border: 2px solid #2196F3; border-radius: 16px;")

        self.processed_info_label = QLabel("⚙️ Plataforma de Segmentación IA - Listo para innovar")
        self.processed_info_label.setAlignment(Qt.AlignCenter)
        self.processed_info_label.setStyleSheet("font-weight: bold; color: #1976d2; font-size: 12pt;")

        self.load_image_button = QPushButton("🚀 Cargar Imagen Futurista")
        self.load_image_button.setToolTip("Carga una imagen para segmentar con IA avanzada")
        self.load_image_button.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #90caf9, stop:1 #42a5f5); color: #212121; font-weight:bold; font-size:11pt; border-radius:8px;")
        self.load_image_button.clicked.connect(self.load_image)

        # Thumbnail for the loaded image (Left Panel)
        self.image1_thumbnail_label = QLabel(self)
        self.image1_thumbnail_label.setFixedSize(150, 150)
        self.image1_thumbnail_label.setStyleSheet("border: 2px solid #42a5f5; background: qradialgradient(cx:0.5, cy:0.5, radius: 0.9, fx:0.5, fy:0.5, stop:0 #e3f2fd, stop:1 #90caf9); border-radius: 12px; font-size: 48pt; color: #1976d2;")
        self.image1_thumbnail_label.setAlignment(Qt.AlignCenter)
        self.image1_thumbnail_label.setText("🛰️")
        self.image1_thumbnail_label.setScaledContents(True)

        # --- Inicialización de widgets y grupos de controles para el panel derecho ---
        # --- Selección de método de segmentación ---
        self.method_selection_group = QGroupBox("Método de Segmentación")
        self.radio_yolo = QRadioButton("YOLOv8")
        self.radio_detectron = QRadioButton("Detectron2")
        self.radio_deeplab = QRadioButton("DeepLab")
        self.radio_mask2former = QRadioButton("Mask2Former (Panoptic)")
        self.radio_maskrcnn = QRadioButton("Mask R-CNN (Panoptic)")
        self.radio_segformer = QRadioButton("SegFormer (Panoptic)")
        method_vbox = QVBoxLayout()
        method_vbox.addWidget(self.radio_yolo)
        method_vbox.addWidget(self.radio_detectron)
        method_vbox.addWidget(self.radio_deeplab)
        method_vbox.addWidget(self.radio_mask2former)
        method_vbox.addWidget(self.radio_maskrcnn)
        method_vbox.addWidget(self.radio_segformer)
        self.method_selection_group.setLayout(method_vbox)
        self.radio_yolo.setChecked(True)

        # --- Botón para ejecutar segmentación ---
        self.run_segmentation_button = QPushButton("🚀 Ejecutar Segmentación")
        self.run_segmentation_button.setToolTip("Ejecuta la segmentación con el método seleccionado.")

        # --- Grupo de conteo de objetos ---
        self.count_group = QGroupBox("Conteo de Objetos")
        count_layout = QVBoxLayout()
        self.person_count_label = QLabel("👨‍👩‍👧‍👦 Personas: 0")
        self.vehicle_count_label = QLabel("🚗 Vehículos: 0")
        self.traffic_count_label = QLabel("🚦 Semáforos: 0")
        count_layout.addWidget(self.person_count_label)
        count_layout.addWidget(self.vehicle_count_label)
        count_layout.addWidget(self.traffic_count_label)
        self.count_group.setLayout(count_layout)

        # --- Grupo de selección de salida ---
        self.output_group = QGroupBox("Visualización de Resultados")
        self.output_layout = QVBoxLayout()
        self.output_group.setLayout(self.output_layout)
        self.output_group.setEnabled(False)

        # --- Botón y grupo de umbralización de segmentos ---
        self.segment_threshold_button = QPushButton("Ajustar Umbral de Segmentos")
        self.segment_thresh_controls_group = QGroupBox("Controles de Umbralización")
        self.segment_thresh_controls_layout = QVBoxLayout()
        self.segment_thresh_controls_group.setLayout(self.segment_thresh_controls_layout)
        self.segment_thresh_controls_group.setVisible(False)

        # Conexiones de botones y radio buttons principales
        self.run_segmentation_button.clicked.connect(self.run_selected_segmentation)
        self.radio_yolo.toggled.connect(self.on_segmentation_method_changed)
        self.radio_detectron.toggled.connect(self.on_segmentation_method_changed)
        self.radio_deeplab.toggled.connect(self.on_segmentation_method_changed)
        self.radio_maskrcnn.toggled.connect(self.on_segmentation_method_changed)
        # Agrega aquí los otros métodos si usas Mask2Former o SegFormer

        # --- Botón y grupo de histograma ---
        self.histogram_button = QPushButton("📊 Ver Histograma Avanzado")
        self.histogram_controls_group = QGroupBox("Controles de Histograma")
        self.histogram_controls_layout = QVBoxLayout()
        self.histogram_controls_group.setLayout(self.histogram_controls_layout)
        self.histogram_controls_group.setVisible(False)

        # --- Botón para guardar la imagen del panel central ---
        self.save_image_button = QPushButton("💾 Guardar Imagen")
        self.save_image_button.setToolTip("Descarga la imagen actualmente mostrada en el panel central.")
        self.save_image_button.setEnabled(False)
        self.save_image_button.clicked.connect(self.save_current_image)

        self.init_ui()

    def init_ui(self):
        # --- Panel Izquierdo: Miniatura y botón de carga ---
        left_panel_layout = QVBoxLayout()
        self.image1_thumbnail_label.setFixedSize(150, 150)
        self.image1_thumbnail_label.setStyleSheet("border: 2px solid #42a5f5; background: qradialgradient(cx:0.5, cy:0.5, radius: 0.9, fx:0.5, fy:0.5, stop:0 #e3f2fd, stop:1 #90caf9); border-radius: 12px; font-size: 48pt; color: #1976d2;")
        self.image1_thumbnail_label.setAlignment(Qt.AlignCenter)
        self.image1_thumbnail_label.setText("🛰️")
        left_panel_layout.addWidget(self.image1_thumbnail_label)
        left_panel_layout.addSpacing(10)
        left_panel_layout.addWidget(self.load_image_button)
        left_panel_layout.addSpacing(20)
        left_panel_layout.addWidget(self.save_image_button)  # Agregar el botón de guardar imagen
        left_panel_layout.addSpacing(10)
        left_panel_layout.addWidget(self.count_group)
        left_panel_layout.addStretch(1)

        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)
        left_panel_widget.setMinimumWidth(200)
        left_panel_widget.setStyleSheet("background-color: #e3f2fd;")

        # --- Panel Central: Imagen procesada y texto info ---
        center_panel_layout = QVBoxLayout()
        self.processed_label.setMinimumSize(400, 300)
        center_panel_layout.addWidget(self.processed_label, 8)
        center_panel_layout.addSpacing(8)
        center_panel_layout.addWidget(self.processed_info_label, alignment=Qt.AlignCenter)
        center_panel_layout.addStretch(1)

        center_panel_widget = QWidget()
        center_panel_widget.setLayout(center_panel_layout)
        center_panel_widget.setMinimumWidth(500)
        center_panel_widget.setStyleSheet("background-color: #bbdefb;")

        # --- Panel Derecho: Controles ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(QLabel("<h2>Controles</h2>", alignment=Qt.AlignCenter))
        right_panel_layout.addWidget(self.method_selection_group)
        right_panel_layout.addWidget(self.run_segmentation_button)
        right_panel_layout.addWidget(self.count_group)
        right_panel_layout.addWidget(self.output_group)
        right_panel_layout.addWidget(self.segment_threshold_button)
        right_panel_layout.addWidget(self.segment_thresh_controls_group)
        right_panel_layout.addWidget(self.histogram_button)
        right_panel_layout.addWidget(self.histogram_controls_group)
        right_panel_layout.addStretch(1)

        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel_layout)
        right_panel_widget.setMinimumWidth(300)
        right_panel_widget.setStyleSheet("background-color: #e3f2fd;")

        # --- QSplitter para los 3 paneles ---
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(left_panel_widget)
        self.main_splitter.addWidget(center_panel_widget)
        self.main_splitter.addWidget(right_panel_widget)
        self.main_splitter.setSizes([250, 600, 300])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        window_layout = QVBoxLayout()
        window_layout.addWidget(self.main_splitter)
        self.setLayout(window_layout)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("background-color: #e3f2fd;")


    # --- Carga modelo Mask R-CNN ---
    def load_maskrcnn_model(self):
        if not MASKRCNN_AVAILABLE:
            print("Error: La librería 'maskrcnn' no está disponible.")
            return False
        if self.maskrcnn_model is None:
            try:
                print("Cargando modelo Mask R-CNN. Esto puede tardar un poco...")
                QApplication.processEvents() # Keep UI responsive during loading
                self.maskrcnn_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                self.maskrcnn_categories = self.maskrcnn_weights.meta["categories"]
                self.maskrcnn_preprocess = self.maskrcnn_weights.transforms()
                self.maskrcnn_model = maskrcnn_resnet50_fpn(weights=self.maskrcnn_weights, progress=True)
                self.maskrcnn_model.eval()
                self.maskrcnn_model.to("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Modelo Mask R-CNN cargado con {len(self.maskrcnn_categories)} clases (pesos {self.maskrcnn_weights}).")
                return True
            except Exception as e:
                self.maskrcnn_model = None
                print(f"Error al cargar el modelo Mask R-CNN: {e}")
                traceback.print_exc()
                return False
        return True # Model is already loaded

    # --- Procesa segmentación Mask R-CNN ---
    def process_maskrcnn_segmentation(self):
        try:
            print("Ejecutando segmentación Mask R-CNN. Esto puede tardar un poco...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self.reset_segmentation_display()

            img_pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            input_tensor = self.maskrcnn_preprocess(img_pil).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                prediction = self.maskrcnn_model(input_tensor)[0]

            detection_threshold = 0.7
            keep = (prediction['scores'] > detection_threshold) & (prediction['labels'] != 0)
            boxes = prediction['boxes'][keep].cpu().numpy()
            labels = prediction['labels'][keep].cpu().numpy()
            scores = prediction['scores'][keep].cpu().numpy()
            masks = (prediction['masks'][keep] > 0.5).squeeze(1).cpu().numpy()

            print(f"Se detectaron y segmentaron {len(boxes)} objetos con confianza >= {detection_threshold}")

            # Colores aleatorios para cada instancia
            instance_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(masks))]
            img_with_results_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).copy()
            only_masks = np.zeros_like(img_with_results_np)

            for i in range(len(boxes)):
                box = boxes[i]
                label = labels[i]
                score = scores[i]
                mask = masks[i]
                color = instance_colors[i]
                # Mezclar máscara con transparencia
                original_pixels_in_mask = img_with_results_np[mask]
                colored_pixels_to_overlay = np.full_like(original_pixels_in_mask, color, dtype=np.uint8)
                blended_pixels = (colored_pixels_to_overlay * 0.5 + original_pixels_in_mask * 0.5).astype(np.uint8)
                img_with_results_np[mask] = blended_pixels
                only_masks[mask] = color
                # Dibujar caja
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(img_with_results_np, (x1, y1), (x2, y2), color, 2)
                # Etiqueta
                class_name = self.maskrcnn_categories[label]
                label_text = f"{class_name}: {score:.2f}"
                text_origin = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img_with_results_np, (x1, text_origin[1] - text_height - baseline), (x1 + text_width, text_origin[1] + baseline), color, -1)
                text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
                cv2.putText(img_with_results_np, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            # Asegura que siempre exista 'annotated' aunque no haya detecciones
            if img_with_results_np is not None:
                self.maskrcnn_results['annotated'] = img_with_results_np.copy()
            else:
                self.maskrcnn_results['annotated'] = self.image.copy() if self.image is not None else np.zeros((256,256,3), np.uint8)
            self.maskrcnn_results['only_masks'] = only_masks if 'only_masks' in locals() else np.zeros_like(self.maskrcnn_results['annotated'])
            self.populate_output_buttons('maskrcnn')
            self._current_output_key = 'annotated'
            self._current_segmentation_method = 'maskrcnn'
            self.display_segmentation_output('annotated', 'maskrcnn')
            self.output_group.setEnabled(True)
            self.histogram_button.setEnabled(True)
            self.segment_threshold_button.setEnabled(False)
            print("Segmentación Mask R-CNN completada.")
        except Exception as e:
            print(f"Error durante la segmentación Mask R-CNN: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error Mask R-CNN", f"Ocurrió un error durante la segmentación Mask R-CNN: {e}")
            self.reset_segmentation_display()
        finally:
            QApplication.restoreOverrideCursor()

        super().__init__()
        self.setWindowTitle("YOLOv8 & Detectron2 & DeepLab Segmentation App - PyQt5") # Updated window title
        self.image = None # Stores the original loaded image (OpenCV format)

        # YOLOv8 related variables
        self.yolo_model = None
        self.yolo_results = {} # Stores different YOLOv8 output visualizations (OpenCV format)
        self.yolo_counts = {} # Stores YOLOv8 object counts

        # Detectron2 related variables
        self.detectron_model = None
        self.detectron_results = {} # Stores different Detectron2 output visualizations (OpenCV format)
        self.detectron_counts = {} # Stores Detectron2 object counts
        self.detectron_metadata = None # Stores Detectron2 metadata for visualization

        # DeepLab related variables
        self.deeplab_model = None
        self.deeplab_results = {} # Stores different DeepLab output visualizations (OpenCV format)
        self.deeplab_counts = {} # Stores DeepLab object counts
        self.deeplab_weights = None # Stores DeepLab weights
        self.deeplab_preprocess = None # Stores DeepLab preprocessing transforms

        # Mask R-CNN related variables
        self.maskrcnn_model = None
        self.maskrcnn_results = {} # Stores different Mask R-CNN output visualizations (OpenCV format)
        self.maskrcnn_counts = {} # Stores Mask R-CNN object counts
        self.maskrcnn_weights = None # Stores Mask R-CNN weights
        self.maskrcnn_preprocess = None # Stores Mask R-CNN preprocessing transforms
        self.maskrcnn_categories = None # Stores Mask R-CNN categories

        # Target classes from COCO dataset and their names for display (Used by Detectron2, can be adapted for YOLO if needed)
        # person=0, car=2, motorcycle=3, traffic light=10
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            10: 'traffic light'
        }
        # Map internal keys to display names for masks (Standardized keys for both models)
        self.segment_mask_display_names = {
            'mask_person_gray': "Personas",
            'mask_vehicle_gray': "Vehículos",
            'mask_traffic_gray': "Semáforos"
        }

        # Keep track of the currently displayed output key and the method that produced it
        self._current_segmentation_method = None # 'yolo' or 'detectron' or 'deeplab' or 'maskrcnn'
        self._current_output_key = None # e.g., 'annotated', 'mask_person_gray'

        # Keep track of the histogram window instance
        self._histogram_window = None

        # Variable to track which segment mask is currently being thresholded
        self._current_segment_threshold_key = None

        # Almacenar la imagen actualmente mostrada en el panel central (para histograma)
        self._current_display_image = None

        self.init_ui()
        self.apply_styles()

        # Load models on startup (can be moved to a separate button click if preferred)
        print("=== INICIO carga modelo YOLOv8 ===")
        if self.load_yolo_model():
            print("✅ Modelo YOLOv8 cargado correctamente.")
        else:
            print("❌ Error al cargar el modelo YOLOv8.")
            QMessageBox.critical(self, "Error de Modelo", "No se pudo cargar el modelo YOLOv8. Verifica la instalación de ultralytics y los pesos.")
        print("=== FIN carga modelo YOLOv8 ===")

        print("=== INICIO carga modelo Detectron2 ===")
        if self.load_detectron_model():
            print("✅ Modelo Detectron2 cargado correctamente.")
        else:
            print("❌ Error al cargar el modelo Detectron2.")
            QMessageBox.critical(self, "Error de Modelo", "No se pudo cargar el modelo Detectron2. Verifica la instalación de detectron2 y los pesos.")
        print("=== FIN carga modelo Detectron2 ===")

        self.image1_thumbnail_label.setScaledContents(True) # Ensure thumbnail scales


        # --- Segmentation Method Selection (Right Panel) ---
        self.method_selection_group = QGroupBox("Método de Segmentación")
        self.radio_yolo = QRadioButton("YOLOv8")
        self.radio_detectron = QRadioButton("Detectron2")
        self.radio_deeplab = QRadioButton("DeepLab")
        self.radio_maskrcnn = QRadioButton("Mask R-CNN")

        method_vbox = QVBoxLayout()
        method_vbox.addWidget(self.radio_yolo)
        method_vbox.addWidget(self.radio_detectron)
        method_vbox.addWidget(self.radio_deeplab)
        method_vbox.addWidget(self.radio_maskrcnn)
        self.method_selection_group.setLayout(method_vbox)

        # Set default selection and trigger the change handler
        self.radio_yolo.setChecked(True)


        # --- Single Run Segmentation Button ---
        self.run_segmentation_button = QPushButton("🚀 Ejecutar Segmentación")
        self.run_segmentation_button.setToolTip("Ejecuta la segmentación con el método seleccionado.")


        # --- Object Counts Group ---
        self.count_group = QGroupBox("Conteo de Objetos Detectados")
        self.person_count_label = QLabel("👨‍👩‍👧‍👦 Personas: 0")
        self.vehicle_count_label = QLabel("🚗 Vehículos: 0")
        self.traffic_count_label = QLabel("🚦 Semáforos: 0")
        count_vbox = QVBoxLayout()
        count_vbox.addWidget(self.person_count_label)
        count_vbox.addWidget(self.vehicle_count_label)
        count_vbox.addWidget(self.traffic_count_label)
        self.count_group.setLayout(count_vbox)


        # --- Output Display Group (Dynamically populated based on method) ---
        self.output_group = QGroupBox("Mostrar Resultado")
        self.output_layout = QVBoxLayout() # Use a layout that will be cleared and repopulated
        self.output_group.setLayout(self.output_layout)


        # --- Individual Segment Thresholding Controls ---
        self.segment_threshold_button = QPushButton("🔳 Umbralización de Segmentos")
        self.segment_threshold_button.setToolTip("Muestra opciones para umbralizar máscaras de segmentos individuales.")
        self.segment_threshold_button.setEnabled(False) # Disabled until results are available

        self.segment_thresh_controls_group = QGroupBox("Opciones Umbralización de Segmentos")
        self.segment_thresh_slider_label = QLabel("Umbral: 127") # Label to show slider value
        self.segment_thresh_slider = QSlider(Qt.Horizontal)
        self.segment_thresh_slider.setRange(0, 255)
        self.segment_thresh_slider.setValue(127)
        self.segment_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.segment_thresh_slider.setTickInterval(10)
        self.segment_thresh_inv_checkbox = QCheckBox("Umbralización Invertida")

        # Layout for segment thresholding buttons
        # Use a QGridLayout for better control over button arrangement
        segment_buttons_grid = QGridLayout()
        self.thresh_person_button = QPushButton("👤 Personas") # Simplified button text
        self.thresh_vehicle_button = QPushButton("🚗 Vehículos") # Simplified button text
        self.thresh_traffic_button = QPushButton("🚦 Semáforos") # Simplified button text

        # Add buttons to the grid layout
        segment_buttons_grid.addWidget(self.thresh_person_button, 0, 0) # Row 0, Col 0
        segment_buttons_grid.addWidget(self.thresh_vehicle_button, 0, 1) # Row 0, Col 1
        segment_buttons_grid.addWidget(self.thresh_traffic_button, 0, 2) # Row 0, Col 2
        # Add stretch to columns to distribute space
        segment_buttons_grid.setColumnStretch(0, 1)
        segment_buttons_grid.setColumnStretch(1, 1)
        segment_buttons_grid.setColumnStretch(2, 1)


        segment_thresh_vbox = QVBoxLayout()
        segment_thresh_vbox.addWidget(self.segment_thresh_slider_label)
        segment_thresh_vbox.addWidget(self.segment_thresh_slider)
        segment_thresh_vbox.addWidget(self.segment_thresh_inv_checkbox)
        segment_thresh_vbox.addSpacing(10) # Space between checkbox and category buttons
        segment_thresh_vbox.addLayout(segment_buttons_grid) # Add the grid layout of buttons

        self.segment_thresh_controls_group.setLayout(segment_thresh_vbox)
        self.segment_thresh_controls_group.setVisible(False) # Hidden initially


        # --- Histogram Controls ---
        self.histogram_button = QPushButton("📊 Mostrar Histograma")
        self.histogram_button.setToolTip("Muestra las opciones para generar histogramas.")
        self.histogram_button.setEnabled(False) # Disabled until an image is loaded

        self.histogram_controls_group = QGroupBox("Generar Histograma")
        self.hist_original_button = QPushButton("Imagen Original")
        self.hist_original_button.setToolTip("Generar histograma para la imagen original.")
        self.hist_processed_button = QPushButton("Imagen Procesada")
        self.hist_processed_button.setToolTip("Generar histograma para la imagen procesada actual.")

        hist_vbox = QVBoxLayout()
        hist_vbox.addWidget(self.hist_original_button)
        hist_vbox.addWidget(self.hist_processed_button)
        self.histogram_controls_group.setLayout(hist_vbox)
        self.histogram_controls_group.setVisible(False)


        # --- Layout for the processed image display ---
        processed_display_layout = QVBoxLayout()
        processed_display_layout.addWidget(self.processed_info_label)

        # Add the processed_label and give it a stretch factor so it takes available vertical space
        # --- Connect signals ---

        self.load_image_button.clicked.connect(self.load_image)

    def load_image(self):
        """Carga una imagen y actualiza el thumbnail y el panel central."""
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Selecciona una imagen", "", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)")
        if filepath:
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.image = img
                # Actualiza thumbnail
                thumb_pixmap = self.cv_to_qpixmap(img)
                self.image1_thumbnail_label.setPixmap(thumb_pixmap.scaled(self.image1_thumbnail_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.image1_thumbnail_label.setText("")
                # Muestra imagen cargada en el panel central
                self.display_processed_image(img, "Imagen cargada - Listo para segmentar 🚀")
                self.histogram_button.setEnabled(True)
                self.output_group.setEnabled(False)
                self.reset_segmentation_display()
            else:
                QMessageBox.warning(self, "Error", "No se pudo cargar la imagen seleccionada.")

        # Connect radio buttons to the method changed handler
        self.radio_yolo.toggled.connect(self.on_segmentation_method_changed)
        self.radio_detectron.toggled.connect(self.on_segmentation_method_changed)
        self.radio_deeplab.toggled.connect(self.on_segmentation_method_changed)
        self.radio_maskrcnn.toggled.connect(self.on_segmentation_method_changed)

        # Connect the single run button
        self.run_segmentation_button.clicked.connect(self.run_selected_segmentation)


        # Connections for Histogram controls
        self.histogram_button.clicked.connect(self.show_histogram_controls)
        self.hist_original_button.clicked.connect(lambda: self.generate_and_display_histogram('original'))
        self.hist_processed_button.clicked.connect(lambda: self.generate_and_display_histogram('processed'))

        # --- Individual Segment Thresholding Connections ---
        self.segment_threshold_button.clicked.connect(self.show_segment_threshold_controls)
        self.segment_thresh_slider.valueChanged.connect(self.update_segment_threshold_label)
        # Re-apply threshold when slider or checkbox changes, but only if a segment is currently selected
        self.segment_thresh_slider.valueChanged.connect(lambda: self.apply_segment_threshold() if self._current_segment_threshold_key is not None else None)
        self.segment_thresh_inv_checkbox.stateChanged.connect(lambda: self.apply_segment_threshold() if self._current_segment_threshold_key is not None else None)

        # Buttons to select which segment mask to threshold - they also trigger the initial threshold application
        # Using standardized keys ('mask_..._gray')
        self.thresh_person_button.clicked.connect(lambda: self.apply_segment_threshold('mask_person_gray'))
        self.thresh_vehicle_button.clicked.connect(lambda: self.apply_segment_threshold('mask_vehicle_gray'))
        self.thresh_traffic_button.clicked.connect(lambda: self.apply_segment_threshold('mask_traffic_gray'))
        # --- End Individual Segment Thresholding Connections ---

        # Initial state update based on default radio button selection
        self.on_segmentation_method_changed()


    # --- Apply QSS Styles ---
    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 10pt;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e3f2fd, stop:1 #bbdefb);
            }
            h2 {
                font-size: 16pt;
                color: #1565c0;
                margin-bottom: 5px;
                font-weight: bold;
                letter-spacing: 1px;
                text-shadow: 1px 1px 8px #90caf9;
            }
            QPushButton {
                padding: 10px 20px;
                border: 1.5px solid #42a5f5;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #90caf9, stop:1 #42a5f5);
                color: #212121;
                font-weight: bold;
                font-size: 11pt;
                margin: 4px;
                min-width: 120px;
                box-shadow: 0 2px 8px #90caf9;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #64b5f6, stop:1 #1976d2);
                color: #fff;
                border-color: #1976d2;
            }
            QPushButton:pressed {
                background: #1976d2;
                color: #fff;
            }
            QPushButton:disabled {
                background: #e3f2fd;
                color: #a0a0a0;
                border-color: #bbdefb;
            }

            QGroupBox {
                margin-top: 1.5em;
                border: 2px solid #42a5f5;
                border-radius: 10px;
                padding: 16px;
                padding-top: 24px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e3f2fd, stop:1 #bbdefb);
                min-width: 320px;
                box-shadow: 0 2px 8px #90caf9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                left: 10px;
                color: #1565c0;
                font-weight: bold;
                font-size: 13pt;
            }

            QLabel {
                font-size: 11pt;
            }

            QLabel[cssClass="image-display"] {
               border: 2px solid #2196F3;
               background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e3f2fd, stop:1 #bbdefb);
               min-width: 200px;
               min-height: 200px;
               border-radius: 16px;
               font-size: 18pt;
               color: #1976d2;
            }

            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                height: 8px;
                background: #ddd;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1976d2;
                border: 1px solid #1976d2;
                width: 22px;
                height: 22px;
                margin: -7px 0;
                border-radius: 11px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            #processed_info_label {
                 font-size: 13pt;
                 color: #1976d2;
                 font-weight: bold;
            }

            QSplitter::handle {
                background-color: #90caf9;
            }
            QSplitter::handle:hover {
                background-color: #42a5f5;
            }
            QSplitter::handle:horizontal {
                width: 7px;
            }
        """);


    # --- Helper function to convert OpenCV image to QPixmap ---
    # Using the robust version from the previous step
    def cv_to_qpixmap(self, cv_image):
        if cv_image is None:
            return QPixmap()

        cv_image = np.ascontiguousarray(cv_image) # Ensure contiguity

        if cv_image.ndim == 2: # Grayscale (1 channel)
            height, width = cv_image.shape
            # QImage.Format_Grayscale8 is the correct format for 8-bit grayscale images
            bytes_per_line = width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif cv_image.ndim == 3 and cv_image.shape[2] == 3: # Color BGR (3 channels)
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for QImage
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif cv_image.ndim == 3 and cv_image.shape[2] == 4: # Color BGRA (with alpha, 4 channels)
             height, width, channel = cv_image.shape
             bytes_per_line = 4 * width
             rgba_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA) # Convert BGRA to RGBA
             q_image = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

        else:
            print(f"Formato de imagen no soportado directamente: {cv_image.shape}, Dtype: {cv_image.dtype}")
            # Basic handling attempt if possible (using robust logic from INTERFAZ.py)
            try:
                if cv_image.max() <= 255 and cv_image.min() >=0 and cv_image.dtype==np.uint8:
                    # Could be a weird format but uint8, try displaying as BGR if it has 3 channels
                    if cv_image.ndim == 3 and cv_image.shape[2] == 3:
                         print("Attempting to display as BGR...")
                         return self.cv_to_qpixmap(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                    # Or as grayscale if it has 1 implicit channel (ndim can be 3 with shape[2]==1 sometimes?)
                    elif cv_image.ndim == 2 or (cv_image.ndim == 3 and cv_image.shape[2] == 1):
                         print("Attempting to display as grayscale...")
                         # If it has 3 dim but 1 channel, reduce to 2 dim for QImage.Format_Grayscale8
                         if cv_image.ndim == 3 and cv_image.shape[2] == 1:
                             cv_image = cv_image[:, :, 0] # Take the only channel
                         height, width = cv_image.shape
                         bytes_per_line = width
                         q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    else:
                        raise ValueError("Dimensions not handleable.")


                else:
                     # If not uint8, try normalizing if it's float
                     if cv_image.dtype in [np.float32, np.float64]:
                         print("Normalizing float image to uint8...")
                         img_norm = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                         # Retry conversion with normalized image
                         return self.cv_to_qpixmap(img_norm)
                     else:
                         raise ValueError(f"Data type not handleable: {cv_image.dtype}")


            except Exception as e:
                 QMessageBox.warning(self, "Format Error", f"Unsupported image format: {cv_image.shape}, Dtype: {cv_image.dtype}. Error: {e}")
                 return QPixmap()


        # Check if the QImage is valid before creating QPixmap
        if q_image.isNull():
             print("QImage is null after conversion")
             return QPixmap()

        return QPixmap.fromImage(q_image)


    # --- Method to display an image in the processed label (Central Panel) ---
    def display_processed_image(self, img, info_text="Resultado de Segmentación"):
        if img is None:
            self.processed_label.setText(
                "<div style='font-size:22pt; color:#2196F3; font-weight:bold; text-shadow: 1px 1px 8px #90caf9;'>🪐 Bienvenido a la Plataforma de Segmentación Futurista 🪐<br><span style='font-size:13pt;'>Carga una imagen y explora DeepLab, Mask R-CNN y más</span></div>"
            )
            self.processed_info_label.setText("⚙️ Plataforma de Segmentación IA - Listo para innovar")
            self._current_display_image = None
            self.processed_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e3f2fd, stop:1 #bbdefb); border: 2px solid #2196F3; border-radius: 16px;")
            self.save_image_button.setEnabled(False)
            return
        if img is not None:
            self._current_display_image = img # Almacenar la imagen mostrada para el histograma
            pixmap = self.cv_to_qpixmap(img)
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.processed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_label.setPixmap(scaled_pixmap)
            self.processed_label.setText("") # Clear placeholder text
            self.processed_info_label.setText(f"⚙️ {info_text}") # Update info label

            # Habilitar el botón de guardar imagen solo si hay imagen
            self.save_image_button.setEnabled(True)

            # Actualizar la ventana del histograma si está abierta
            if self._histogram_window is not None and self._histogram_window.isVisible():
                try:
                    self._histogram_window.image_data = img.copy() # Pasar una copia
                    self._histogram_window.calculate_histograms()
                    self._histogram_window.update_plot()
                except RuntimeError:
                    print("RuntimeError: Intentando acceder a ventana de histograma eliminada. Limpiando referencia.")
                    self._histogram_window = None
        else:
            self._current_display_image = None
            self.processed_label.setPixmap(QPixmap())
            self.processed_label.setText("No se pudo procesar/mostrar el resultado")
            self.processed_info_label.setText("❌ Error de Visualización")
            self.save_image_button.setEnabled(False)


    # --- Hide/Show Conditional Controls ---
    def hide_all_conditional_controls(self):
        """Hides all specific control groups."""
        self.histogram_controls_group.setVisible(False)
        self.segment_thresh_controls_group.setVisible(False) # Hide the new group
        self.output_group.setEnabled(False) # Disable output group initially


    # --- Reset Segmentation Display ---
    def reset_segmentation_display(self):
        """Resets the display and control states for segmentation results."""
        self.yolo_results = {}
        self.yolo_counts = {}
        self.detectron_results = {}
        self.detectron_counts = {}
        self.deeplab_results = {}
        self.deeplab_counts = {}
        self.maskrcnn_results = {}
        self.maskrcnn_counts = {}

        # Reset count labels
        self.person_count_label.setText("👨‍👩‍👧‍👦 Personas: 0")
        self.vehicle_count_label.setText("🚗 Vehículos: 0")
        self.traffic_count_label.setText("🚦 Semáforos: 0")

        # Clear and disable output and thresholding controls
        self.clear_output_buttons()
        self.output_group.setEnabled(False)
        self.segment_threshold_button.setEnabled(False)
        self.segment_thresh_controls_group.setVisible(False)
        self._current_segment_threshold_key = None # Reset current segment key

        # Clear the main display label and references
        self._current_display_image = None
        self.processed_label.setPixmap(QPixmap())
        self.processed_label.setText("✨ Resultado de Segmentación Aquí ✨\n(Carga una imagen y ejecuta la segmentación)")
        self.processed_info_label.setText("⚙️ Resultado de Segmentación")
        self._current_segmentation_method = None
        self._current_output_key = None

        # Hide conditional control groups
        self.hide_all_conditional_controls()

        # Disable histogram button if no image is loaded
        if self.image is None:
             self.histogram_button.setEnabled(False)

        # Update run button state based on current method selection and image availability
        self.on_segmentation_method_changed()


    # --- Image Loading Method ---
    def load_image(self):
        """Loads an image file and displays its thumbnail."""
        path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Imagen (*.png *.jpg *.jpeg *.bmp *.tiff);;Todos los archivos (*)")
        if path:
            # Read the image using OpenCV
            img = cv2.imread(path)
            if img is None:
                print(f"Error: No se pudo cargar la imagen desde '{path}'.")
                QMessageBox.critical(self, "Error", f"No se pudo cargar la imagen desde '{path}'.")
                self.image = None
                # Clear thumbnail and reset display
                self.image1_thumbnail_label.setPixmap(QPixmap())
                self.image1_thumbnail_label.setText("📁")
                self.reset_segmentation_display() # Use the unified reset
            else:
                self.image = img.copy() # Store a copy of the original image
                try:
                    # Create and display thumbnail
                    # Use the fixed size set in init_ui for the thumbnail label
                    thumb_size = self.image1_thumbnail_label.size()
                    # Resize the image to the fixed size for the thumbnail
                    thumb = cv2.resize(img, (thumb_size.width(), thumb_size.height()), interpolation=cv2.INTER_AREA)
                    thumb_pixmap = self.cv_to_qpixmap(thumb)
                    # Set the pixmap on the label, it will be scaled because setScaledContents(True) is set
                    self.image1_thumbnail_label.setPixmap(thumb_pixmap)
                    self.image1_thumbnail_label.setText("")
                except Exception as e:
                     print(f"Error al mostrar miniatura de imagen: {e}")
                     traceback.print_exc() # Print full traceback for debugging
                     self.image1_thumbnail_label.setPixmap(QPixmap())
                     self.image1_thumbnail_label.setText("❌")

                # Reset display and enable controls after loading image
                self.reset_segmentation_display() # Use the unified reset
                self.histogram_button.setEnabled(True) # Enable histogram for original image
                # The run button state is updated by reset_segmentation_display calling on_segmentation_method_changed

                print(f"Imagen cargada correctamente desde '{path}'. Lista para segmentación.")


    # --- Load YOLO Model ---
    def load_yolo_model(self):
        """Loads the YOLOv8 model."""
        if not YOLO_AVAILABLE:
             print("Error: La librería 'ultralytics' no está disponible.")
             # QMessageBox.warning(self, "Librería Faltante", "La librería 'ultralytics' no está instalada.") # Show message when trying to run
             return False
        if self.yolo_model is None:
            try:
                print("Cargando modelo YOLOv8x-seg. Esto puede tardar un poco...")
                QApplication.processEvents() # Keep UI responsive during loading
                self.yolo_model = YOLO('yolov8x-seg.pt')
                print("Modelo YOLOv8 cargado correctamente.")
                return True
            except Exception as e:
                self.yolo_model = None
                print(f"Error al cargar el modelo YOLO: {e}")
                traceback.print_exc()
                # QMessageBox.critical(self, "Error del Modelo", f"No se pudo cargar el modelo YOLOv8x-seg. Asegúrate de tenerlo descargado ('yolov8x-seg.pt' en la misma carpeta o en el cache de ultralytics) o de tener conexión a internet para descargarlo. Error: {e}") # Show message when trying to run
                return False
        return True # Model is already loaded

    # --- Load Detectron2 Model ---
    def load_detectron_model(self):
        """Loads the Detectron2 model."""
        if not DETECTRON_AVAILABLE:
             print("Error: Detectron2 o Torch no están disponibles.")
             # QMessageBox.warning(self, "Librería Faltante", "Las librerías de Detectron2 o Torch no están instaladas.") # Show message when trying to run
             return False
        if self.detectron_model is None:
            try:
                print("Cargando modelo Detectron2 (mask_rcnn_R_101_FPN_3x). Esto puede tardar un poco...")
                QApplication.processEvents() # Keep UI responsive during loading

                # === CONFIGURE Detectron2 MODEL ===
                cfg = get_cfg()
                # Merge default config file
                cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
                # Set confidence threshold for predictions
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                # Set device to use (GPU if available, otherwise CPU)
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                # Load model weights
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

                # Create the predictor
                self.detectron_model = DefaultPredictor(cfg)
                self.detectron_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) # Get metadata for visualization

                print("Modelo Detectron2 cargado correctamente.")
                return True
            except Exception as e:
                self.detectron_model = None
                self.detectron_metadata = None
                print(f"Error al cargar el modelo Detectron2: {e}")
                traceback.print_exc() # Print full traceback for debugging
                # QMessageBox.critical(self, "Error del Modelo", f"No se pudo cargar el modelo Detectron2. Asegúrate de tenerlo descargado o de tener conexión a internet para descargarlo. Error: {e}") # Show message when trying to run
                return False
        return True # Model is already loaded

    # --- Load DeepLab Model ---
    def load_deeplab_model(self):
        """Loads the DeepLab model."""
        if not DEEPLAB_AVAILABLE:
             print("Error: DeepLab no está disponible.")
             return False
        if self.deeplab_model is None:
            try:
                print("Cargando modelo DeepLab (deeplabv3_resnet101). Esto puede tardar un poco...")
                QApplication.processEvents() # Keep UI responsive during loading

                # === CONFIGURE DeepLab MODEL ===
                self.deeplab_weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                self.deeplab_preprocess = self.deeplab_weights.transforms()

                self.deeplab_model = deeplabv3_resnet101(weights=self.deeplab_weights, progress=True)
                self.deeplab_model.eval()
                self.deeplab_model.to("cuda" if torch.cuda.is_available() else "cpu")

                # Definir el mapa de colores estándar de Pascal VOC (21 clases) para DeepLab
                self.PASCAL_VOC_COLORMAP = [
                    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128] # Índice 20
                ]

                # === CONFIGURE Faster R-CNN MODEL ===
                print("Cargando modelo Faster R-CNN para detección robusta...")
                self.detection_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.detection_model = fasterrcnn_resnet50_fpn(weights=self.detection_weights, progress=True)
                self.detection_model.eval()
                self.detection_model.to("cuda" if torch.cuda.is_available() else "cpu")
                self.detection_categories = self.detection_weights.meta["categories"]
                self.detection_preprocess = self.detection_weights.transforms()

                # Generar colores aleatorios para las clases de detección
                self.DET_COLORS = {}
                for i in range(len(self.detection_categories)):
                    self.DET_COLORS[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                print("Modelo DeepLab y Faster R-CNN cargados correctamente.")
                return True
            except Exception as e:
                self.deeplab_model = None
                print(f"Error al cargar el modelo DeepLab: {e}")
                traceback.print_exc()
                return False
        return True

    def decode_semseg_mask(self, image_mask, colormap):
        """Convierte una máscara de segmentación semántica a una imagen coloreada."""
        height, width = image_mask.shape
        color_seg_image = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id in np.unique(image_mask):
            if class_id < len(colormap):
                mask_pixels = image_mask == class_id
                color = colormap[class_id]
                color_seg_image[mask_pixels, 0] = color[0] # R
                color_seg_image[mask_pixels, 1] = color[1] # G
                color_seg_image[mask_pixels, 2] = color[2] # B
        return color_seg_image

    # --- Load Mask R-CNN Model ---
    def load_maskrcnn_model(self):
        """Loads the Mask R-CNN model."""
        if not MASKRCNN_AVAILABLE:
            print("Error: Mask R-CNN no está disponible.")
            return False
        if self.maskrcnn_model is None:
            try:
                self.maskrcnn_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                self.maskrcnn_categories = self.maskrcnn_weights.meta["categories"]
                self.maskrcnn_preprocess = self.maskrcnn_weights.transforms()
                self.maskrcnn_model = maskrcnn_resnet50_fpn(weights=self.maskrcnn_weights, progress=True)
                self.maskrcnn_model.eval()
                self.maskrcnn_model.to("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Modelo Mask R-CNN cargado con {len(self.maskrcnn_categories)} clases (pesos {self.maskrcnn_weights}).")
                return True
            except Exception as e:
                self.maskrcnn_model = None
                print(f"Error al cargar el modelo Mask R-CNN: {e}")
                traceback.print_exc()
                return False
        return True

    # --- Handle Segmentation Method Changed ---
    def on_segmentation_method_changed(self):
        """Updates UI based on the selected segmentation method."""
        if self.radio_yolo.isChecked():
            self._current_segmentation_method = 'yolo'
            self.run_segmentation_button.setText("🚀 Ejecutar Segmentación YOLOv8")
            # Enable run button only if YOLO is available and an image is loaded
            self.run_segmentation_button.setEnabled(YOLO_AVAILABLE and self.image is not None)
            if not YOLO_AVAILABLE:
                 self.run_segmentation_button.setToolTip("Instala ultralytics y torch para habilitar la segmentación YOLO.")
            elif self.image is None:
                 self.run_segmentation_button.setToolTip("Carga una imagen para ejecutar la segmentación.")
            else:
                 self.run_segmentation_button.setToolTip("Ejecuta el modelo YOLOv8 para detectar y segmentar objetos.")

        elif self.radio_detectron.isChecked():
            self._current_segmentation_method = 'detectron'
            self.run_segmentation_button.setText("🚀 Ejecutar Segmentación Detectron2")
            # Enable run button only if Detectron2 is available and an image is loaded
            self.run_segmentation_button.setEnabled(DETECTRON_AVAILABLE and self.image is not None)
            if not DETECTRON_AVAILABLE:
                 self.run_segmentation_button.setToolTip("Instala Detectron2 y Torch para habilitar la segmentación.")
            elif self.image is None:
                 self.run_segmentation_button.setToolTip("Carga una imagen para ejecutar la segmentación.")
            else:
                 self.run_segmentation_button.setToolTip("Ejecuta el modelo Detectron2 para detectar y segmentar objetos.")

        elif self.radio_deeplab.isChecked():
            self._current_segmentation_method = 'deeplab'
            self.run_segmentation_button.setText("🚀 Ejecutar Segmentación DeepLab")
            # Enable run button only if DeepLab is available and an image is loaded
            self.run_segmentation_button.setEnabled(DEEPLAB_AVAILABLE and self.image is not None)
            if not DEEPLAB_AVAILABLE:
                 self.run_segmentation_button.setToolTip("Instala DeepLab para habilitar la segmentación.")
            elif self.image is None:
                 self.run_segmentation_button.setToolTip("Carga una imagen para ejecutar la segmentación.")
            else:
                 self.run_segmentation_button.setToolTip("Ejecuta el modelo DeepLab para detectar y segmentar objetos.")

        elif self.radio_maskrcnn.isChecked():
            self._current_segmentation_method = 'maskrcnn'
            self.run_segmentation_button.setText("🚀 Ejecutar Segmentación Mask R-CNN")
            self.run_segmentation_button.setEnabled(MASKRCNN_AVAILABLE and self.image is not None)
            if not MASKRCNN_AVAILABLE:
                self.run_segmentation_button.setToolTip("Instala torchvision y torch para habilitar Mask R-CNN.")
            elif self.image is None:
                self.run_segmentation_button.setToolTip("Carga una imagen para ejecutar la segmentación.")
            else:
                self.run_segmentation_button.setToolTip("Ejecuta el modelo Mask R-CNN para detectar y segmentar objetos.")

        # Clear previous output buttons when method changes
        self.clear_output_buttons()
        self.output_group.setEnabled(False)
        self.segment_threshold_button.setEnabled(False)
        self.segment_thresh_controls_group.setVisible(False)
        self._current_segment_threshold_key = None
        self._current_output_key = None
        # Keep the current image displayed, but reset info text
        if self._current_display_image is not None:
             self.display_processed_image(self._current_display_image, "Selecciona un Método y Ejecuta")
        else:
             self.processed_label.setPixmap(QPixmap())
             self.processed_label.setText("✨ Resultado de Segmentación Aquí ✨\n(Carga una imagen y ejecuta la segmentación)")
             self.processed_info_label.setText("⚙️ Resultado de Segmentación")


    # --- Run Selected Segmentation ---
    def run_selected_segmentation(self):
        """Runs segmentation using the currently selected method."""
        if self.image is None:
            QMessageBox.warning(self, "Advertencia", "Carga una imagen primero.")
            return

        if self._current_segmentation_method == 'yolo':
            if not YOLO_AVAILABLE:
                 QMessageBox.warning(self, "Librería Faltante", "La librería 'ultralytics' no está instalada.")
                 return
            if self.yolo_model is None and not self.load_yolo_model():
                 # Error message already shown in load_yolo_model
                 return
            self.process_yolo_segmentation()

        elif self._current_segmentation_method == 'detectron':
            if not DETECTRON_AVAILABLE:
                 QMessageBox.warning(self, "Librería Faltante", "Las librerías de Detectron2 o Torch no están instaladas.\n\nInstala las librerías necesarias. Puedes seguir las instrucciones de instalación de Detectron2:\nhttps://github.com/facebookresearch/detectron2/blob/main/INSTALL.md\nAsegúrate de instalar PyTorch compatible con tu configuración (CUDA si tienes GPU): https://pytorch.org/get-started/locally/")
                 return
            if self.detectron_model is None and not self.load_detectron_model():
                 # Error message already shown in load_detectron_model
                 return
            self.process_detectron_segmentation()

        elif self._current_segmentation_method == 'deeplab':
            if not DEEPLAB_AVAILABLE:
                 QMessageBox.warning(self, "Librería Faltante", "La librería 'deeplab' no está instalada.")
                 return
            if self.deeplab_model is None and not self.load_deeplab_model():
                 # Error message already shown in load_deeplab_model
                 return
            self.process_deeplab_segmentation()

        elif self._current_segmentation_method == 'maskrcnn':
            if not MASKRCNN_AVAILABLE:
                QMessageBox.warning(self, "Librería Faltante", "La librería 'torchvision' o 'torch' no está instalada.")
                return
            if self.maskrcnn_model is None and not self.load_maskrcnn_model():
                return
            self.process_maskrcnn_segmentation()
        else:
            QMessageBox.warning(self, "Error de Selección", "Método de segmentación no válido seleccionado.")


    # --- Process Segmentation with YOLOv8 ---
    def process_yolo_segmentation(self):
        """Runs YOLOv8 inference on the loaded image."""
        print("Ejecutando segmentación YOLOv8. Esto puede tardar un poco...")
        QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor
        QApplication.processEvents() # Keep UI responsive

        try:
            self.reset_segmentation_display() # Reset display before new processing

            results = self.yolo_model(self.image, imgsz=960, verbose=False)[0]

            # Target classes for YOLOv8 (based on COCO dataset)
            # person (0), car (2), motorcycle (3), traffic light (9)
            yolo_target_classes = [0, 2, 3, 9]

            mask_person = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask_vehicle = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask_traffic = np.zeros(self.image.shape[:2], dtype=np.uint8)

            count_person = 0
            count_vehicle = 0
            count_traffic = 0

            if results.masks is not None:
                for mask, cls in zip(results.masks.data, results.boxes.cls):
                    class_id = int(cls)
                    if class_id in yolo_target_classes:
                        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                        if mask_np.shape[:2] != self.image.shape[:2]:
                             mask_np = cv2.resize(mask_np, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)

                        if class_id == 0: # person
                            mask_person = cv2.bitwise_or(mask_person, mask_np)
                            count_person += 1
                        elif class_id in [2, 3]: # car, motorcycle
                            mask_vehicle = cv2.bitwise_or(mask_vehicle, mask_np)
                            count_vehicle += 1
                        elif class_id == 9: # traffic light
                            mask_traffic = cv2.bitwise_or(mask_traffic, mask_np)
                            count_traffic += 1

            self.yolo_counts['person'] = count_person
            self.yolo_counts['vehicle'] = count_vehicle
            self.yolo_counts['traffic'] = count_traffic

            self.person_count_label.setText(f"👨‍👩‍👧‍👦 Personas: {count_person}")
            self.vehicle_count_label.setText(f"🚗 Vehículos: {count_vehicle}")
            self.traffic_count_label.setText(f"🚦 Semáforos: {count_traffic}")


            # Generate and store different output visualizations (OpenCV format) for YOLOv8
            # Using standardized keys ('mask_..._gray')

            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.yolo_results['mask_person_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_person)
            self.yolo_results['mask_vehicle_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_vehicle)
            self.yolo_results['mask_traffic_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_traffic)

            color_combined = np.zeros_like(self.image)
            color_combined[mask_person > 0] = [0, 255, 0] # Green
            color_combined[mask_vehicle > 0] = [255, 0, 0] # Blue
            color_combined[mask_traffic > 0] = [0, 255, 255] # Yellow
            self.yolo_results['color_combined'] = color_combined

            mask_combined = cv2.bitwise_or(mask_person, mask_vehicle)
            mask_combined = cv2.bitwise_or(mask_combined, mask_traffic)
            self.yolo_results['mask_combined'] = mask_combined

            annotated_image = results.plot().copy().astype(np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_image, f"Personas: {count_person}", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Vehiculos: {count_vehicle}", (10, 60), font, 0.8, (255, 0, 0), 2)
            cv2.putText(annotated_image, f"Semaforos: {count_traffic}", (10, 90), font, 0.8, (0, 255, 255), 2)
            self.yolo_results['annotated'] = annotated_image


            # Populate output buttons for YOLOv8
            self.populate_output_buttons('yolo')

            # Display the default output (annotated image)
            self.display_segmentation_output('annotated', 'yolo')

            self.output_group.setEnabled(True)
            self.histogram_button.setEnabled(True)
            self.segment_threshold_button.setEnabled(True) # Enable main thresholding button


            print("Segmentación YOLOv8 completada.")

        except Exception as e:
            print(f"Error durante la segmentación YOLOv8: {e}")
            traceback.print_exc() # Print full traceback for debugging
            QMessageBox.critical(self, "Error YOLOv8", f"Ocurrió un error durante la segmentación YOLOv8: {e}")
            self.reset_segmentation_display() # Reset display on error

        finally:
             QApplication.restoreOverrideCursor() # Restore normal cursor

    # --- Process Segmentation with Detectron2 ---
    def process_detectron_segmentation(self):
        """Runs Detectron2 inference on the loaded image."""
        import traceback
        print("Ejecutando segmentación Detectron2. Esto puede tardar un poco...")
        QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor
        QApplication.processEvents() # Keep UI responsive

        try:
            self.reset_segmentation_display() # Reset display before new processing

            # Perform inference
            outputs = self.detectron_model(self.image)
            instances = outputs["instances"]

            # Initialize masks and counts for target classes
            mask_person = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask_vehicle = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask_traffic = np.zeros(self.image.shape[:2], dtype=np.uint8)

            count_person = 0
            count_vehicle = 0
            count_traffic = 0

            # Process detected instances
            if instances.has("pred_masks") and instances.has("pred_classes"):
                for mask, cls in zip(instances.pred_masks, instances.pred_classes):
                    class_id = int(cls)
                    # Convert mask tensor to numpy array and scale to 0-255
                    mask_np = (mask.cpu().numpy().astype(np.uint8) * 255)

                    # Ensure mask size matches image size (should usually match, but safety check)
                    if mask_np.shape[:2] != self.image.shape[:2]:
                         mask_np = cv2.resize(mask_np, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)


                    # Accumulate masks and count objects for target classes
                    if class_id in self.target_classes:
                        class_name = self.detectron_metadata.thing_classes[class_id] # Get class name from metadata
                        if class_name == 'person':
                            mask_person = cv2.bitwise_or(mask_person, mask_np)
                            count_person += 1
                        elif class_name in ['car', 'motorcycle']: # Group car and motorcycle as vehicles
                            mask_vehicle = cv2.bitwise_or(mask_vehicle, mask_np)
                            count_vehicle += 1
                        elif class_name == 'traffic light':
                            mask_traffic = cv2.bitwise_or(mask_traffic, mask_np)
                            count_traffic += 1

            # Store counts
            self.detectron_counts['person'] = count_person
            self.detectron_counts['vehicle'] = count_vehicle
            self.detectron_counts['traffic'] = count_traffic

            # Update count labels in the UI
            self.person_count_label.setText(f"👨‍👩‍👧‍👦 Personas: {count_person}")
            self.vehicle_count_label.setText(f"🚗 Vehículos: {count_vehicle}")
            self.traffic_count_label.setText(f"🚦 Semáforos: {count_traffic}")


            # --- Generate and Store Different Output Visualizations for Detectron2 ---
            # Using standardized keys ('mask_..._gray')

            # Grayscale versions of individual masks (for thresholding and display)
            # Apply the mask to the grayscale version of the original image
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.detectron_results['mask_person_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_person)
            self.detectron_results['mask_vehicle_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_vehicle)
            self.detectron_results['mask_traffic_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_traffic)


            # Combined color segmentation mask
            color_combined = np.zeros_like(self.image)
            # Assign distinct colors to each segment type
            color_combined[mask_person > 0] = [0, 255, 0]      # Green for persons
            color_combined[mask_vehicle > 0] = [255, 0, 0]     # Blue for vehicles
            color_combined[mask_traffic > 0] = [0, 255, 255]   # Yellow for traffic lights
            self.detectron_results['color_combined'] = color_combined

            # Combined binary mask (all detected segments)
            mask_combined = cv2.bitwise_or(mask_person, mask_vehicle)
            mask_combined = cv2.bitwise_or(mask_combined, mask_traffic)
            self.detectron_results['mask_combined'] = mask_combined


            # Annotated image with bounding boxes and masks from Detectron2 Visualizer
            # Use the original image (converted to RGB for Visualizer)
            v = Visualizer(self.image[:, :, ::-1], metadata=self.detectron_metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
            # Draw predictions - instances are already on CPU if using CPU device, otherwise move them
            output = v.draw_instance_predictions(instances.to("cpu"))
            # Get the annotated image (convert back to BGR for OpenCV/display)
            annotated_image = output.get_image()[:, :, ::-1].copy().astype(np.uint8)

            # Add counts as text overlay on the annotated image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_image, f"Personas: {count_person}", (10, 30), font, 0.8, (0, 255, 0), 2) # Green text
            cv2.putText(annotated_image, f"Vehiculos: {count_vehicle}", (10, 60), font, 0.8, (255, 0, 0), 2) # Blue text
            cv2.putText(annotated_image, f"Semaforos: {count_traffic}", (10, 90), font, 0.8, (0, 255, 255), 2) # Yellow text

            self.detectron_results['annotated'] = annotated_image


            # Populate output buttons for Detectron2
            self.populate_output_buttons('detectron')

            # Display the default output (annotated image)
            self.display_segmentation_output('annotated', 'detectron')

            self.output_group.setEnabled(True)
            self.histogram_button.setEnabled(True)
            self.segment_threshold_button.setEnabled(True) # Enable main thresholding button


            print("Segmentación Detectron2 completada.")

        except Exception as e:
            print(f"Error durante la segmentación Detectron2: {e}")
            traceback.print_exc() # Print full traceback for debugging
            QMessageBox.critical(self, "Error Detectron2", f"Ocurrió un error durante la segmentación Detectron2: {e}")
            self.reset_segmentation_display() # Reset display on error

        finally:
             QApplication.restoreOverrideCursor() # Restore normal cursor

    # --- Process Segmentation with DeepLab ---
    def process_deeplab_segmentation(self):
        """Runs DeepLab inference on the loaded image."""
        print("Ejecutando segmentación DeepLab. Esto puede tardar un poco...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            self.reset_segmentation_display()

            # Convertir imagen OpenCV a PIL
            img_pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            
            # Preprocesar y realizar inferencia de DeepLab
            input_tensor = self.deeplab_preprocess(img_pil).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                output = self.deeplab_model(input_tensor)
                semseg_predictions = output['out'][0].argmax(0).cpu().numpy()

            # Convertir la máscara semántica a imagen coloreada
            semseg_img_rgb = self.decode_semseg_mask(semseg_predictions, self.PASCAL_VOC_COLORMAP)
            semseg_img_bgr = cv2.cvtColor(semseg_img_rgb, cv2.COLOR_RGB2BGR)

            # COCO class indices for DeepLabV3:
            # person: 15, car: 13, motorcycle: 17, traffic light: 10
            mask_person = (semseg_predictions == 15).astype(np.uint8) * 255
            mask_vehicle = ((semseg_predictions == 13) | (semseg_predictions == 17)).astype(np.uint8) * 255
            mask_traffic = (semseg_predictions == 10).astype(np.uint8) * 255

            # --- Resize masks to original image size ---
            original_h, original_w = self.image.shape[:2]
            mask_person = cv2.resize(mask_person, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            mask_vehicle = cv2.resize(mask_vehicle, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            mask_traffic = cv2.resize(mask_traffic, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


            # --- Realizar detección de objetos con Faster R-CNN ---
            det_input_tensor = self.detection_preprocess(img_pil).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                det_prediction = self.detection_model(det_input_tensor)[0]

            # Filtrar predicciones por umbral de confianza y clases de interés
            detection_threshold = 0.3  # Reducido de 0.5 a 0.3 para detectar más objetos
            target_classes = ['person', 'car', 'motorcycle', 'traffic light']
            det_keep = (det_prediction['scores'] > detection_threshold) & \
                      (torch.tensor([self.detection_categories[label] in target_classes 
                                   for label in det_prediction['labels']], device=det_prediction['labels'].device))
            
            det_boxes = det_prediction['boxes'][det_keep].cpu().numpy()
            det_labels = det_prediction['labels'][det_keep].cpu().numpy()
            det_scores = det_prediction['scores'][det_keep].cpu().numpy()

            # --- Fusionar las detecciones de Faster R-CNN con las máscaras de DeepLab ---
            # Esto asegura que las máscaras individuales contengan todos los objetos detectados aunque DeepLab no los haya segmentado
            det_person_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            det_vehicle_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            det_traffic_mask = np.zeros((original_h, original_w), dtype=np.uint8)

            for i in range(len(det_boxes)):
                box = det_boxes[i]
                label_id = det_labels[i]
                class_name = self.detection_categories[label_id]
                x1, y1, x2, y2 = [int(coord) for coord in box]
                if class_name == 'person':
                    det_person_mask[y1:y2, x1:x2] = 255
                elif class_name in ['car', 'motorcycle']:
                    det_vehicle_mask[y1:y2, x1:x2] = 255
                elif class_name == 'traffic light':
                    det_traffic_mask[y1:y2, x1:x2] = 255

            # Unir las máscaras semánticas y de detección
            mask_person = cv2.bitwise_or(mask_person, det_person_mask)
            mask_vehicle = cv2.bitwise_or(mask_vehicle, det_vehicle_mask)
            mask_traffic = cv2.bitwise_or(mask_traffic, det_traffic_mask)

            # --- Dibujar cajas delimitadoras y etiquetas ---
            annotated_image = self.image.copy()
            
            # Contadores para cada clase
            count_person = 0
            count_vehicle = 0
            count_traffic = 0

            # Crear una imagen de segmentación a color personalizada usando colores más visibles
            color_combined = np.zeros_like(self.image)
            # Asignar colores específicos para cada clase (en formato BGR)
            color_combined[mask_person > 0] = [0, 255, 0]      # Verde para personas
            color_combined[mask_vehicle > 0] = [255, 0, 0]     # Azul para vehículos
            color_combined[mask_traffic > 0] = [0, 255, 255]   # Amarillo para semáforos

            # Mejorar la visibilidad de las máscaras combinándolas con la imagen original
            alpha = 0.5  # Reducido de 0.7 a 0.5 para mejor visibilidad
            color_combined = cv2.addWeighted(self.image, 1-alpha, color_combined, alpha, 0)

            # Dibujar las cajas delimitadoras y etiquetas
            for i in range(len(det_boxes)):
                box = det_boxes[i]
                label_id = det_labels[i]
                score = det_scores[i]
                class_name = self.detection_categories[label_id]

                # Coordenadas de la caja
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Actualizar contadores según la clase
                if class_name == 'person':
                    count_person += 1
                    color = (0, 255, 0)  # Verde para personas
                elif class_name in ['car', 'motorcycle']:
                    count_vehicle += 1
                    color = (255, 0, 0)  # Azul para vehículos
                elif class_name == 'traffic light':
                    count_traffic += 1
                    color = (0, 255, 255)  # Amarillo para semáforos
                    # Si es un semáforo y no fue detectado por DeepLab, usar la detección de Faster R-CNN
                    if not np.any(mask_traffic > 0):
                        mask_traffic[y1:y2, x1:x2] = 255
                        color_combined[y1:y2, x1:x2] = [0, 255, 255]  # Amarillo para semáforos

                # --- DIBUJAR CAJA Y ETIQUETA SOBRE LA IMAGEN ANOTADA ---
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                label_text = f"{class_name} ({score:.2f})"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(annotated_image, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Añadir contadores en la parte superior izquierda
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_image, f"Personas: {count_person}", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Vehiculos: {count_vehicle}", (10, 60), font, 0.8, (255, 0, 0), 2)
            cv2.putText(annotated_image, f"Semaforos: {count_traffic}", (10, 90), font, 0.8, (0, 255, 255), 2)

            # Store counts
            self.deeplab_counts['person'] = count_person
            self.deeplab_counts['vehicle'] = count_vehicle
            self.deeplab_counts['traffic'] = count_traffic

            # Update count labels
            self.person_count_label.setText(f"👨‍👩‍👧‍👦 Personas: {count_person}")
            self.vehicle_count_label.setText(f"🚗 Vehículos: {count_vehicle}")
            self.traffic_count_label.setText(f"🚦 Semáforos: {count_traffic}")

            # Grayscale image for mask overlays
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.deeplab_results['mask_person_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_person)
            self.deeplab_results['mask_vehicle_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_vehicle)
            self.deeplab_results['mask_traffic_gray'] = cv2.bitwise_and(gray_image, gray_image, mask=mask_traffic)


            # Color combined mask (usando colores personalizados)
            self.deeplab_results['color_combined'] = color_combined

            # Combined binary mask
            mask_combined = cv2.bitwise_or(mask_person, mask_vehicle)
            mask_combined = cv2.bitwise_or(mask_combined, mask_traffic)
            self.deeplab_results['mask_combined'] = mask_combined

            # Guardar la imagen anotada con las cajas delimitadoras
            self.deeplab_results['annotated'] = annotated_image

            # Populate output buttons for DeepLab
            self.populate_output_buttons('deeplab')

            # Display the default output (annotated image)
            self._current_output_key = 'annotated'
            self._current_segmentation_method = 'deeplab'
            self.display_segmentation_output('annotated', 'deeplab')

            self.output_group.setEnabled(True)
            self.histogram_button.setEnabled(True)
            self.segment_threshold_button.setEnabled(True)

            print("Segmentación DeepLab completada.")
            # Refuerzo: mostrar la imagen anotada en el panel central (por si el usuario espera la actualización inmediata)
            if 'annotated' in self.deeplab_results:
                self.display_processed_image(self.deeplab_results['annotated'], "✅ Resultado Anotado (DeepLab)")


        except Exception as e:
            print(f"Error durante la segmentación DeepLab: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error DeepLab", f"Ocurrió un error durante la segmentación DeepLab: {e}")
            self.reset_segmentation_display()

        finally:
             QApplication.restoreOverrideCursor()


    # --- Populate Output Buttons ---
    def populate_output_buttons(self, method):
        """Clears existing output buttons and adds new ones based on the method."""
        # Clear existing buttons from the layout
        while self.output_layout.count():
            item = self.output_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Add buttons based on the selected method
        if method == 'yolo':
            self.output_layout.addWidget(QPushButton("✅ Anotada (Bounding Boxes + Máscaras)", clicked=lambda: self.display_segmentation_output('annotated', 'yolo')))
            self.output_layout.addWidget(QPushButton("🌈 Segmentación por Color", clicked=lambda: self.display_segmentation_output('color_combined', 'yolo')))
            self.output_layout.addWidget(QPushButton("⚫⚪ Máscara Binaria Total", clicked=lambda: self.display_segmentation_output('mask_combined', 'yolo')))
            self.output_layout.addWidget(QPushButton("👤 Máscara de Personas (Gris)", clicked=lambda: self.display_segmentation_output('mask_person_gray', 'yolo')))
            self.output_layout.addWidget(QPushButton("🚗 Máscara de Vehículos (Gris)", clicked=lambda: self.display_segmentation_output('mask_vehicle_gray', 'yolo')))
            self.output_layout.addWidget(QPushButton("🚦 Máscara de Semáforos (Gris)", clicked=lambda: self.display_segmentation_output('mask_traffic_gray', 'yolo')))

        elif method == 'detectron':
            self.output_layout.addWidget(QPushButton("✅ Anotada (Bounding Boxes + Máscaras)", clicked=lambda: self.display_segmentation_output('annotated', 'detectron')))
            self.output_layout.addWidget(QPushButton("🌈 Segmentación por Color", clicked=lambda: self.display_segmentation_output('color_combined', 'detectron')))
            self.output_layout.addWidget(QPushButton("⚫⚪ Máscara Binaria Total", clicked=lambda: self.display_segmentation_output('mask_combined', 'detectron')))
            self.output_layout.addWidget(QPushButton("👤 Máscara de Personas (Gris)", clicked=lambda: self.display_segmentation_output('mask_person_gray', 'detectron')))
            self.output_layout.addWidget(QPushButton("🚗 Máscara de Vehículos (Gris)", clicked=lambda: self.display_segmentation_output('mask_vehicle_gray', 'detectron')))
            self.output_layout.addWidget(QPushButton("🚦 Máscara de Semáforos (Gris)", clicked=lambda: self.display_segmentation_output('mask_traffic_gray', 'detectron')))

        elif method == 'deeplab':
            self.output_layout.addWidget(QPushButton("✅ Anotada (Bounding Boxes + Máscaras)", clicked=lambda: self.display_segmentation_output('annotated', 'deeplab')))
            self.output_layout.addWidget(QPushButton("🌈 Segmentación por Color", clicked=lambda: self.display_segmentation_output('color_combined', 'deeplab')))
            self.output_layout.addWidget(QPushButton("⚫⚪ Máscara Binaria Total", clicked=lambda: self.display_segmentation_output('mask_combined', 'deeplab')))
            self.output_layout.addWidget(QPushButton("👤 Máscara de Personas (Gris)", clicked=lambda: self.display_segmentation_output('mask_person_gray', 'deeplab')))
            self.output_layout.addWidget(QPushButton("🚗 Máscara de Vehículos (Gris)", clicked=lambda: self.display_segmentation_output('mask_vehicle_gray', 'deeplab')))
            self.output_layout.addWidget(QPushButton("🚦 Máscara de Semáforos (Gris)", clicked=lambda: self.display_segmentation_output('mask_traffic_gray', 'deeplab')))

        elif method == 'maskrcnn':
            self.output_layout.addWidget(QPushButton("✅ Anotada (Bounding Boxes + Máscaras)", clicked=lambda: self.display_segmentation_output('annotated', 'maskrcnn')))
            self.output_layout.addWidget(QPushButton("🎨 Solo Máscaras", clicked=lambda: self.display_segmentation_output('only_masks', 'maskrcnn')))

        self.output_layout.addStretch(1) # Add stretch to push buttons to the top


    # --- Clear Output Buttons ---
    def clear_output_buttons(self):
        """Removes all buttons from the output group layout."""
        while self.output_layout.count():
            item = self.output_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


    # --- Display Selected Segmentation Output ---
    def display_segmentation_output(self, output_key, method):
        """Displays the selected segmentation output image in the main panel."""
        results_dict = self.yolo_results if method == 'yolo' else self.detectron_results if method == 'detectron' else self.deeplab_results

        if output_key in results_dict:
            img = results_dict[output_key]
            # Map output keys to display text
            info_map = {
                'annotated': "✅ Resultado Anotado",
                'color_combined': "🌈 Segmentación por Color",
                'mask_combined': "⚫⚪ Máscara Binaria Total",
                'mask_person_gray': "👤 Máscara Personas (Gris)",
                'mask_vehicle_gray': "🚗 Máscara Vehículos (Gris)",
                'mask_traffic_gray': "🚦 Máscaras Semáforos (Gris)"
            }
            method_name = "YOLOv8" if self._current_segmentation_method == 'yolo' else "Detectron2" if self._current_segmentation_method == 'detectron' else "DeepLab" # Use the current method for info text
            info_text = f"{info_map.get(output_key, 'Resultado')} ({method_name})"

            self.display_processed_image(img, info_text) # display_processed_image updates _current_display_image
            self._current_segmentation_method = method # Store the method that produced this output
            self._current_output_key = output_key # Store the key of the currently displayed output

            # Show/hide segment thresholding controls based on the displayed output key
            if output_key in ['mask_person_gray', 'mask_vehicle_gray', 'mask_traffic_gray']:
                 self.segment_threshold_button.setEnabled(True) # Ensure the main button is enabled
                 # Keep the controls group visible if it was already visible, otherwise it will be shown by clicking segment_threshold_button
                 # self.segment_thresh_controls_group.setVisible(True) # Don't force visible, let the button control it
                 # If a specific mask is displayed, set it as the current threshold key
                 self._current_segment_threshold_key = output_key
                 # Re-apply threshold if controls are visible to reflect current slider/checkbox state
                 if self.segment_thresh_controls_group.isVisible():
                      self.apply_segment_threshold()

            else:
                 # Hide segment thresholding controls if a non-mask output is displayed
                 self.segment_thresh_controls_group.setVisible(False)
                 self.segment_threshold_button.setEnabled(False) # Disable main button
                 self._current_segment_threshold_key = None # Reset segment key


        else:
             self.display_processed_image(None, "Resultado no disponible")
             print(f"Error: Resultado '{output_key}' no disponible para el método '{method}'.")
             self._current_segmentation_method = None
             self._current_output_key = None
             self.segment_thresh_controls_group.setVisible(False)
             self.segment_threshold_button.setEnabled(False)
             self._current_segment_threshold_key = None


    # --- Individual Segment Thresholding Methods ---

    # Show/Hide the segment thresholding controls group
    def show_segment_threshold_controls(self):
        """Shows or hides the segment thresholding controls group."""
        # Hide other conditional groups
        self.histogram_controls_group.setVisible(False)

        # Toggle the visibility of the segment thresholding group
        is_visible = self.segment_thresh_controls_group.isVisible()
        self.segment_thresh_controls_group.setVisible(not is_visible)

        # If the group is shown and a segment mask key is stored, re-apply the threshold
        # This ensures the displayed image matches the controls when the group is revealed
        if self.segment_thresh_controls_group.isVisible() and self._current_segment_threshold_key is not None:
             self.apply_segment_threshold()
        elif self.segment_thresh_controls_group.isVisible() and self._current_segment_threshold_key is None:
             # If showing controls but no segment is selected, prompt the user
             self.display_processed_image(None, "Selecciona un Segmento para Umbralizar")


    # Update the label for the segment threshold slider
    def update_segment_threshold_label(self, value):
        """Updates the label next to the segment threshold slider."""
        self.segment_thresh_slider_label.setText(f"Umbral: {value}")


    # Apply threshold to the currently selected segment mask
    def apply_segment_threshold(self, segment_key=None):
        """
        Applies thresholding to a specific segment mask.

        Args:
            segment_key (str, optional): The key of the segment mask to threshold
                                         (e.g., 'mask_person_gray'). If None,
                                         uses the currently stored key (_current_segment_threshold_key).
        """
        # If a key is provided, update the current segment key
        if segment_key is not None:
            self._current_segment_threshold_key = segment_key

        # Check if a segment is selected for thresholding
        if self._current_segment_threshold_key is None:
            print("No hay segmento seleccionado para umbralización.")
            # If controls are visible but no segment is selected, display a message
            if self.segment_thresh_controls_group.isVisible():
                 self.display_processed_image(None, "Selecciona un Segmento para Umbralizar")
            return

        # Determine which results dictionary to use based on the current method
        results_dict = None
        if self._current_segmentation_method == 'yolo':
             results_dict = self.yolo_results
        elif self._current_segmentation_method == 'detectron':
             results_dict = self.detectron_results
        elif self._current_segmentation_method == 'deeplab':
             results_dict = self.deeplab_results
        else:
             print("Error: Método de segmentación desconocido para umbralización.")
             self._current_segment_threshold_key = None # Reset key
             self.display_processed_image(None, "Error Umbralizando Segmento")
             return


        # Get the mask for the current segment
        if self._current_segment_threshold_key not in results_dict:
             print(f"Error: Máscara '{self._current_segment_threshold_key}' no disponible para umbralizar con el método '{self._current_segmentation_method}'.")
             QMessageBox.warning(self, "Advertencia", f"La máscara '{self._current_segment_threshold_key}' no está disponible para el método seleccionado.")
             self._current_segment_threshold_key = None # Reset the key
             self.display_processed_image(None, "Error Umbralizando Segmento") # Clear display
             return

        mask_to_process = results_dict[self._current_segment_threshold_key]

        # Ensure the mask is grayscale (2D) - the stored masks are already grayscale
        if mask_to_process.ndim != 2:
            print(f"Error: La imagen para umbralizar '{self._current_segment_threshold_key}' no es en escala de grises ({mask_to_process.ndim} dimensiones).")
            QMessageBox.warning(self, "Advertencia", f"La máscara '{self._current_segment_threshold_key}' no es en escala de grises.")
            self._current_segment_threshold_key = None # Reset the key
            self.display_processed_image(None, "Error Umbralizando Segmento")
            return

        # Get current slider and checkbox values
        threshold_value = self.segment_thresh_slider.value()
        max_value = 255 # Binary masks are 0 or 255

        # Determine the threshold type
        if self.segment_thresh_inv_checkbox.isChecked():
             thresh_type = cv2.THRESH_BINARY_INV
        else:
             thresh_type = cv2.THRESH_BINARY

        # Apply thresholding
        _, thresholded_mask = cv2.threshold(mask_to_process, threshold_value, max_value, thresh_type)

        # Map internal keys to display names for info text
        info_text_base = self.segment_mask_display_names.get(self._current_segment_threshold_key, "Umbral Segmento")
        method_name = "YOLOv8" if self._current_segmentation_method == 'yolo' else "Detectron2" if self._current_segmentation_method == 'detectron' else "DeepLab"
        info_text = f"{info_text_base} ({threshold_value}) ({method_name})"
        if self.segment_thresh_inv_checkbox.isChecked():
             info_text += " Invertido"

        # Display the thresholded result (which is a 2D mask)
        self.display_processed_image(thresholded_mask, info_text) # display_processed_image updates _current_display_image
        # When thresholding, the displayed image is the thresholded mask, not a result from yolo_results/detectron_results
        # So we don't update _current_output_key here, only _current_display_image


    # --- Histogram Methods ---

    def show_histogram_controls(self):
        """Shows or hides the histogram controls group."""
        self.segment_thresh_controls_group.setVisible(False) # Ocultar controles de umbralización de segmentos

        is_visible = self.histogram_controls_group.isVisible()
        self.histogram_controls_group.setVisible(not is_visible)

        # If the histogram group is shown, update the main display info text
        if self.histogram_controls_group.isVisible():
             img_to_show = self._current_display_image if self._current_display_image is not None else self.image
             info_text = "Generar Histograma"
             if self._current_display_image is not None:
                  # Get info text from the current processed info label
                  current_info = self.processed_info_label.text().replace('⚙️ ', '')
                  info_text = f"Generar Histograma ({current_info})"
             elif self.image is not None:
                  info_text = "Generar Histograma (Imagen Original)"

             # Display the current image with updated info text (copy for safety)
             if img_to_show is not None:
                 self.display_processed_image(img_to_show.copy(), info_text)
             else:
                 # If no image is loaded, clear the central panel
                 self.display_processed_image(None, info_text)


    # Generar y mostrar el histograma usando la ventana de Histograma avanzada
    def generate_and_display_histogram(self, image_source):
        """
        Generates and displays a histogram for the selected image source using the advanced window.

        Args:
            image_source (str): 'original' or 'processed'.
        """
        img_to_process = None
        title = "Histograma"

        # Determine which image to use based on the selected source
        if image_source == 'original': # Usar explícitamente la Imagen original cargada
            if self.image is not None:
                img_to_process = self.image.copy() # Work on a copy
                title = "Histograma - Imagen Original"
                print("Generando histograma para Imagen Original...")
            else:
                QMessageBox.warning(self, "Advertencia", "Carga una imagen primero para ver su histograma.")
                print("Advertencia: No hay Imagen Original cargada.")
                return
        elif image_source == 'processed':
            # Use the image currently displayed in the central panel (stored in _current_display_image)
            if self._current_display_image is not None:
                 img_to_process = self._current_display_image.copy() # Work on a copy
                 # Get the info text from the central panel for the histogram title
                 info_text = self.processed_info_label.text().replace('⚙️ ', '') # Clean up emoji/symbol
                 title = f"Histograma - {info_text}"
                 print(f"Generando histograma para Imagen Procesada ({info_text})...")
            else:
                QMessageBox.warning(self, "Advertencia", "No hay imagen procesada visible para generar histograma.")
                print("Advertencia: No hay Imagen Procesada disponible para histograma.")
                return
        else:
            print(f"Error: Fuente '{image_source}' desconocida para histograma.")
            return


        if img_to_process is None:
             print("Error: No se pudo obtener la imagen para el histograma.")
             return


        # *** Crear y mostrar la ventana de Histograma avanzada ***
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor) # Mostrar cursor de ocupado
            # Si la ventana ya existe y está visible, la cerramos primero para abrir una nueva con los datos actualizados
            if self._histogram_window is not None and self._histogram_window.isVisible():
                 print("Cerrando ventana de histograma existente para abrir una nueva...")
                 self._histogram_window.close() # Esto activará _on_histogram_window_closed y limpiará la referencia
                 # Esperar un momento para que la ventana se cierre completamente si es necesario, o confiar en el evento finished

            # Crear una nueva instancia de la ventana de Histograma avanzada
            # Pasar una copia de los datos de la imagen a la ventana de Histograma
            self._histogram_window = HistogramWindow(img_to_process, self) # Pass the image copy
            # Connect the 'finished' signal of the histogram window to a slot to clear the reference
            self._histogram_window.finished.connect(self._on_histogram_window_closed)

            self._histogram_window.setWindowTitle(title) # Set the window title
            # The new Histogram window calculates and displays the histogram in its __init__

            self._histogram_window.show() # Show the window (non-modal)
            self._histogram_window.raise_() # Bring the window to the front
            self._histogram_window.activateWindow() # Activate the window


            QApplication.restoreOverrideCursor() # Restore cursor


        except Exception as e:
            QApplication.restoreOverrideCursor() # Restore cursor before showing the message
            print(f"Error al crear/mostrar ventana de histograma: {e}")
            traceback.print_exc() # Imprimir traceback completo para depuración
            QMessageBox.critical(self, "Error de Histograma", f"Ocurrió un error al crear la ventana del histograma: {e}")


    # --- Slot to clear the histogram window reference ---
    def _on_histogram_window_closed(self):
        """Establece self._histogram_window a None cuando la ventana del histograma se cierra."""
        print("Ventana de histograma cerrada. Limpiando referencia.")
        self._histogram_window = None


    # Clean up resources when the widget is destroyed
    def closeEvent(self, event):
        """Handles cleanup when the widget is closed."""
        # Close the histogram window if it's open
        if self._histogram_window is not None:
            print("Cerrando ventana de histograma al cerrar la aplicación principal.")
            self._histogram_window.close() # Cerrar la ventana de diálogo

        cv2.destroyAllWindows() # Close any leftover OpenCV windows

        # Allow the close event to proceed (close the widget)
        super().closeEvent(event)

    def save_current_image(self):
        """Permite al usuario guardar la imagen actualmente mostrada en el panel central."""
        if self._current_display_image is None:
            QMessageBox.warning(self, "Sin imagen", "No hay ninguna imagen para guardar.")
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Imagen", "resultado.png", "Archivos de Imagen (*.png *.jpg *.bmp)", options=options)
        if file_path:
            # Convertir la imagen a formato BGR si es necesario
            img_to_save = self._current_display_image
            # Si es una imagen de un solo canal, convertir a 3 canales para guardar como PNG/JPG
            if len(img_to_save.shape) == 2:
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
            try:
                cv2.imwrite(file_path, img_to_save)
                QMessageBox.information(self, "Imagen guardada", f"Imagen guardada en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error al guardar", f"No se pudo guardar la imagen.\nError: {e}")

# --- Main Application Entry Point ---
if __name__ == "__main__":
    # Settings for high-density pixel displays (recommended for PyQt)
    # This helps GUI elements appear correctly on high-resolution screens
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create the application instance
    app = QApplication(sys.argv)
    # Create the main window instance
    window = ImageProcessor()
    # Show the window
    window.show()
    # Start the main application event loop
    sys.exit(app.exec_())
    