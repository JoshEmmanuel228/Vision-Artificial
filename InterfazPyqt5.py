import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import traceback

# PyQt5 imports
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QLabel, QComboBox, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QSlider,
    QRadioButton, QButtonGroup, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QScrollArea, QFrame, QLineEdit,
    QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QDoubleValidator

# Define OpenCV color space constants
COLOR_BGR2HSV = cv2.COLOR_BGR2HSV
COLOR_BGR2LAB = cv2.COLOR_BGR2LAB
COLOR_BGR2YCrCb = cv2.COLOR_BGR2YCrCb
COLOR_BGR2HLS = cv2.COLOR_BGR2HLS
COLOR_BGR2YUV = cv2.COLOR_BGR2YUV
COLOR_BGRA2BGR = cv2.COLOR_BGRA2BGR
COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
COLOR_BGRA2GRAY = cv2.COLOR_BGRA2GRAY
COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
COLOR_BGRA2RGBA = cv2.COLOR_BGRA2RGBA
COLOR_GRAY2BGR = cv2.COLOR_GRAY2BGR

# Define OpenCV normalization constants
NORM_MINMAX = cv2.NORM_MINMAX
CV_8U = cv2.CV_8U

# Define OpenCV threshold constants
THRESH_BINARY = cv2.THRESH_BINARY
THRESH_BINARY_INV = cv2.THRESH_BINARY_INV

# Define OpenCV interpolation constants
INTER_AREA = cv2.INTER_AREA
INTER_LINEAR = cv2.INTER_LINEAR

# Define OpenCV image reading constants
IMREAD_UNCHANGED = cv2.IMREAD_UNCHANGED

# Define color spaces and their channels
COLOR_SPACES = {
    "HSV": {
        "channels": ["Hue", "Saturation", "Value"],
        "cv_code": COLOR_BGR2HSV
    },
    "LAB": {
        "channels": ["L", "A", "B"],
        "cv_code": COLOR_BGR2LAB
    },
    "YCrCb": {
        "channels": ["Y", "Cr", "Cb"],
        "cv_code": COLOR_BGR2YCrCb
    },
    "HLS": {
        "channels": ["Hue", "Lightness", "Saturation"],
        "cv_code": COLOR_BGR2HLS
    },
    "YUV": {
        "channels": ["Y", "U", "V"],
        "cv_code": COLOR_BGR2YUV
    }
}

# Importar la clase ImageProcessor de YolovDetectron
try:
    from YolovDetectron import ImageProcessor as YoloImageProcessor
    print("YolovDetectron import successful")
except ImportError:
    print("Warning: YolovDetectron module not found. Some features may be limited.")
    YoloImageProcessor = None

# Configurar el backend de matplotlib para usar Qt5Agg
import matplotlib
matplotlib.use('Qt5Agg')

# Importar scipy con manejo de errores
try:
    import scipy
    print(f"SciPy import successful. Version: {scipy.__version__}")
except ImportError:
    print("Warning: SciPy module not found. Some features may be limited.")
    scipy = None

# Desactivar barras de herramientas por defecto de Matplotlib
plt.rcParams["toolbar"] = "None"

# Importaciones PyQt5: Aseguramos que están QDialog, QComboBox y otros que usamos
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QLabel, QComboBox
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# Importar la clase ImageProcessor de YolovDetectron
try:
    from YolovDetectron import ImageProcessor as SegmentationProcessor
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False
    print("Advertencia: No se pudo importar YolovDetectron.py")
    print("La funcionalidad de segmentación no estará disponible.")

# Importaciones para Matplotlib
# --- CORRECCIÓN: Especificar el backend de Matplotlib ANTES de importar otros módulos de Matplotlib ---
import matplotlib
matplotlib.use('Qt5Agg') # Asegura que Matplotlib use el backend de PyQt5
# --- Fin CORRECCIÓN ---

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure # Necesario para crear la figura del gráfico
import matplotlib.pyplot as plt # Necesario para plotear

# Intenta importar scipy, necesario para el filtro de moda
try:
    from scipy import ndimage, stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Removed import from bordes-edge.py

# Define OpenCV color space constants
COLOR_BGR2HSV = cv2.COLOR_BGR2HSV
COLOR_BGR2LAB = cv2.COLOR_BGR2LAB
COLOR_BGR2YCrCb = cv2.COLOR_BGR2YCrCb
COLOR_BGR2HLS = cv2.COLOR_BGR2HLS
COLOR_BGR2YUV = cv2.COLOR_BGR2YUV
COLOR_BGRA2BGR = cv2.COLOR_BGRA2BGR
COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
COLOR_BGRA2GRAY = cv2.COLOR_BGRA2GRAY
COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
COLOR_BGRA2RGBA = cv2.COLOR_BGRA2RGBA
COLOR_GRAY2BGR = cv2.COLOR_GRAY2BGR

# Define OpenCV normalization constants
NORM_MINMAX = cv2.NORM_MINMAX
CV_8U = cv2.CV_8U

# Define OpenCV threshold constants
THRESH_BINARY = cv2.THRESH_BINARY
THRESH_BINARY_INV = cv2.THRESH_BINARY_INV

# Define OpenCV interpolation constants
INTER_AREA = cv2.INTER_AREA
INTER_LINEAR = cv2.INTER_LINEAR

# Define OpenCV image reading constants
IMREAD_UNCHANGED = cv2.IMREAD_UNCHANGED

class ColorSpaceConverter:
    """Clase dedicada para manejar conversiones de espacio de color de manera robusta."""
    
    @staticmethod
    def get_conversion_flag(space_name):
        """Obtiene el código de conversión de OpenCV para el espacio de color especificado."""
        conversion_flags = {
            'HSV': cv2.COLOR_BGR2HSV,
            'LAB': cv2.COLOR_BGR2LAB,
            'YCrCb': cv2.COLOR_BGR2YCrCb,
            'HLS': cv2.COLOR_BGR2HLS,
            'YUV': cv2.COLOR_BGR2YUV
        }
        return conversion_flags.get(space_name)
    
    @staticmethod
    def get_channel_names(space_name):
        """Obtiene los nombres de los canales para el espacio de color especificado."""
        channel_names = {
            'HSV': ['Hue', 'Saturation', 'Value'],
            'LAB': ['L', 'A', 'B'],
            'YCrCb': ['Y', 'Cr', 'Cb'],
            'HLS': ['Hue', 'Lightness', 'Saturation'],
            'YUV': ['Y', 'U', 'V']
        }
        return channel_names.get(space_name, [])
    
    @staticmethod
    def normalize_channel(channel, space_name, channel_index):
        """Normaliza un canal específico según su espacio de color."""
        # Convert to float for calculations
        channel_float = channel.astype(np.float32)
        
        # Apply specific normalization based on color space and channel
        if space_name == 'HSV':
            if channel_index == 0:  # Hue
                channel_float = channel_float * 2  # Scale 0-179 to 0-255
            elif channel_index in [1, 2]:  # Saturation and Value
                channel_float = channel_float * 255 / 255  # Already 0-255
        elif space_name == 'LAB':
            if channel_index == 0:  # L
                channel_float = channel_float * 255 / 100  # Scale 0-100 to 0-255
            else:  # a and b
                channel_float = (channel_float + 128) * 255 / 255  # Scale -128 to 127 to 0-255
        elif space_name == 'YCrCb':
            if channel_index == 0:  # Y
                channel_float = channel_float  # Already 0-255
            else:  # Cr and Cb
                channel_float = (channel_float + 128) * 255 / 255  # Scale -128 to 127 to 0-255
        elif space_name == 'HLS':
            if channel_index == 0:  # Hue
                channel_float = channel_float * 2  # Scale 0-179 to 0-255
            else:  # Lightness and Saturation
                channel_float = channel_float * 255 / 255  # Already 0-255
        elif space_name == 'YUV':
            if channel_index == 0:  # Y
                channel_float = channel_float  # Already 0-255
            else:  # U and V
                channel_float = (channel_float + 128) * 255 / 255  # Scale -128 to 127 to 0-255
        
        # Normalize to 0-255 using min-max scaling
        normalized_channel_output = cv2.normalize(channel_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return normalized_channel_output
    
    @staticmethod
    def convert_image(image, space_name):
        """Convierte una imagen al espacio de color especificado."""
        if image is None:
            raise ValueError("No se proporcionó una imagen para convertir")
        
        conversion_flag = ColorSpaceConverter.get_conversion_flag(space_name)
        if conversion_flag is None:
            raise ValueError(f"Espacio de color no soportado: {space_name}")
        
        try:
            # Ensure image is in BGR format
            if len(image.shape) == 2:  # If grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # If RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Convert to target color space
            converted = cv2.cvtColor(image, conversion_flag)
            return converted
            
        except Exception as e:
            raise Exception(f"Error al convertir imagen a {space_name}: {str(e)}")



# --- Custom QLabel para hacerlo clickable (usado para miniaturas) ---
class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self.setText("")  # Clear text when setting an image

class ZoomableLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.offset = QtCore.QPoint(0, 0)
        self.last_mouse_pos = None
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self.zoom_factor = 1.0
        self.offset = QtCore.QPoint(0, 0)
        self.update()

    def wheelEvent(self, event):
        if self.pixmap() is None:
            return

        # Zoom factor change
        zoom_delta = 0.1
        if event.angleDelta().y() > 0:
            self.zoom_factor += zoom_delta
        else:
            self.zoom_factor = max(0.1, self.zoom_factor - zoom_delta)

        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.last_mouse_pos = None
            self.setCursor(QtCore.Qt.OpenHandCursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        if not self.pixmap():
            super().paintEvent(event)
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Calculate scaled size maintaining aspect ratio
        scaled_size = self.pixmap().size() * self.zoom_factor

        # Calculate position to center the image
        x = (self.width() - scaled_size.width()) / 2 + self.offset.x()
        y = (self.height() - scaled_size.height()) / 2 + self.offset.y()

        # Draw the scaled pixmap
        try:
            painter.drawPixmap(QtCore.QRect(int(x), int(y), int(scaled_size.width()), int(scaled_size.height())),
                              self.pixmap(),
                              self.pixmap().rect())
        except Exception as e:
            print(f"Error in paintEvent: {e}")
            # Optionally, show a message box or log the error
            QtWidgets.QMessageBox.critical(self, "Error de Renderizado", f"Ocurrió un error al renderizar la imagen: {e}")


# --- Clase para la Ventana de Histograma (AVANZADA) ---
class HistogramWindow(QtWidgets.QDialog):
    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histograma")
        self.setGeometry(200, 200, 750, 550)

        self.image_data = image_data
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        # --- Layout de Controles del Histograma ---
        controls_layout = QtWidgets.QVBoxLayout()

        # Checkboxes para controlar la visibilidad de canales/luminancia
        histogram_checkbox_layout = QtWidgets.QHBoxLayout()
        self.cb_red = QtWidgets.QCheckBox("Rojo")
        self.cb_red.setChecked(True)
        self.cb_green = QtWidgets.QCheckBox("Verde")
        self.cb_green.setChecked(True)
        self.cb_blue = QtWidgets.QCheckBox("Azul")
        self.cb_blue.setChecked(True)
        self.cb_luminance = QtWidgets.QCheckBox("Luminancia")
        self.cb_luminance.setChecked(True)

        histogram_checkbox_layout.addWidget(self.cb_red)
        histogram_checkbox_layout.addWidget(self.cb_green)
        histogram_checkbox_layout.addWidget(self.cb_blue)
        histogram_checkbox_layout.addWidget(self.cb_luminance)
        histogram_checkbox_layout.addStretch()
        controls_layout.addLayout(histogram_checkbox_layout)

        # Controles Adicionales (Tipo de Plot, Valores, Grid, Etiquetas Ejes, Límite Y, Guardar)
        additional_controls_layout = QtWidgets.QHBoxLayout()

        self.cb_show_bars = QtWidgets.QCheckBox("Mostrar como Barras")
        self.cb_show_bars.setChecked(False)
        self.cb_show_values = QtWidgets.QCheckBox("Mostrar Valores")
        self.cb_show_values.setChecked(False)
        self.cb_grid = QtWidgets.QCheckBox("Mostrar Grid")
        self.cb_grid.setChecked(True)
        self.cb_axis_labels = QtWidgets.QCheckBox("Mostrar Etiquetas")
        self.cb_axis_labels.setChecked(True)
        self.cb_show_max_min = QtWidgets.QCheckBox("Mostrar Max/Min")
        self.cb_show_max_min.setChecked(False)

        additional_controls_layout.addWidget(self.cb_show_bars)
        additional_controls_layout.addWidget(self.cb_show_values)
        additional_controls_layout.addWidget(self.cb_grid)
        additional_controls_layout.addWidget(self.cb_axis_labels)
        additional_controls_layout.addWidget(self.cb_show_max_min)

        self.y_limit_label = QtWidgets.QLabel("Límite Y:")
        self.y_limit_combobox = QtWidgets.QComboBox()
        y_values = []
        y_values.extend(range(100, 1001, 100))
        y_values.extend(range(2000, 10001, 1000))
        y_values.extend(range(20000, 100001, 10000))
        if image_data is not None and len(image_data.shape) > 0:
            total_pixels = image_data.shape[0] * image_data.shape[1]
            if total_pixels > 100000:
                step = 50000 if total_pixels < 500000 else 100000
                for val in range(110000, total_pixels + step, step):
                    if val > (y_values[-1] if y_values else 0):
                        y_values.append(val)
        y_values = sorted(list(set(y_values)))
        self.y_limit_combobox.addItems([str(val) for val in y_values])
        self.save_button = QtWidgets.QPushButton("Guardar Histograma")
        self.save_button.clicked.connect(self.save_histogram_plot)
        additional_controls_layout.addWidget(self.y_limit_label)
        additional_controls_layout.addWidget(self.y_limit_combobox)
        additional_controls_layout.addWidget(self.save_button)
        additional_controls_layout.addStretch()
        controls_layout.addLayout(additional_controls_layout)

        self.cb_red.stateChanged.connect(self.update_plot)
        self.cb_green.stateChanged.connect(self.update_plot)
        self.cb_blue.stateChanged.connect(self.update_plot)
        self.cb_luminance.stateChanged.connect(self.update_plot)
        self.cb_show_bars.stateChanged.connect(self.update_plot)
        self.cb_show_values.stateChanged.connect(self.update_plot)
        self.cb_grid.stateChanged.connect(self.update_plot)
        self.cb_axis_labels.stateChanged.connect(self.update_plot)
        self.cb_show_max_min.stateChanged.connect(self.update_plot)
        self.y_limit_combobox.currentTextChanged.connect(self.update_plot)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(controls_layout)

        self.calculate_histograms()
        self.set_initial_y_limit()
        self.update_plot()

    def calculate_histograms(self):
        self.hist_blue = self.hist_green = self.hist_red = self.hist_luminance = None
        is_color_image = False
        if self.image_data is not None and len(self.image_data.shape) > 0:
            if len(self.image_data.shape) >= 3:
                is_color_image = True
                if self.image_data.shape[2] == 4:
                    img_bgr = cv2.cvtColor(self.image_data, cv2.COLOR_BGRA2BGR)
                else:
                    img_bgr = self.image_data
                self.hist_blue = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])
                self.hist_green = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])
                self.hist_red = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])
                gray_image = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
                self.hist_luminance = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            elif len(self.image_data.shape) == 2:
                self.hist_luminance = cv2.calcHist([self.image_data], [0], None, [256], [0, 256])
        self.cb_red.setEnabled(is_color_image)
        self.cb_green.setEnabled(is_color_image)
        self.cb_blue.setEnabled(is_color_image)
        self.cb_luminance.setEnabled(self.hist_luminance is not None)

    def set_initial_y_limit(self):
        max_count = 0
        if self.hist_red is not None:
            max_count = max(max_count, np.max(self.hist_red))
        if self.hist_green is not None:
            max_count = max(max_count, np.max(self.hist_green))
        if self.hist_blue is not None:
            max_count = max(max_count, np.max(self.hist_blue))
        if self.hist_luminance is not None:
            max_count = max(max_count, np.max(self.hist_luminance))
        if max_count > 0:
            initial_limit_target = int(max_count * 1.05)
            selected_index = 0
            for i in range(self.y_limit_combobox.count()):
                item_value = int(self.y_limit_combobox.itemText(i))
                if item_value >= initial_limit_target:
                    selected_index = i
                    break
                if i == self.y_limit_combobox.count() - 1:
                    selected_index = i
            self.y_limit_combobox.setCurrentIndex(selected_index)
        else:
            default_index = self.y_limit_combobox.findText("1000")
            if default_index != -1:
                self.y_limit_combobox.setCurrentIndex(default_index)
            else:
                if self.y_limit_combobox.count() > 0:
                    self.y_limit_combobox.setCurrentIndex(0)

    def update_plot(self):
        self.axes.clear()
        show_bars = self.cb_show_bars.isChecked()
        show_values = self.cb_show_values.isChecked()
        show_grid = self.cb_grid.isChecked()
        show_axis_labels = self.cb_axis_labels.isChecked()
        show_max_min = self.cb_show_max_min.isChecked()
        self.cb_show_values.setEnabled(show_bars)
        bin_edges = np.arange(256).reshape(-1, 1)
        if self.hist_blue is not None and self.cb_blue.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_blue.flatten(), color='blue', alpha=0.7, width=1, label='Azul')
                if show_values:
                    for i, count in enumerate(self.hist_blue.flatten()):
                        if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='blue')
            else:
                self.axes.plot(self.hist_blue, color='blue', label='Azul')
            if show_max_min and self.hist_blue is not None:
                max_count = np.max(self.hist_blue)
                min_count = np.min(self.hist_blue)
                max_indices = np.where(self.hist_blue == max_count)[0]
                min_indices = np.where(self.hist_blue == min_count)[0]
                for idx in max_indices:
                    self.axes.plot(idx, max_count, 'o', color='darkblue', markersize=8, label=f'Azul Max ({idx}, {int(max_count)})')
                if min_count > 0:
                    for idx in min_indices:
                        self.axes.plot(idx, min_count, 'x', color='darkblue', markersize=8, label=f'Azul Min ({idx}, {int(min_count)})')
        if self.hist_green is not None and self.cb_green.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_green.flatten(), color='green', alpha=0.7, width=1, label='Verde')
                if show_values:
                    for i, count in enumerate(self.hist_green.flatten()):
                        if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='green')
            else:
                self.axes.plot(self.hist_green, color='green', label='Verde')
            if show_max_min and self.hist_green is not None:
                max_count = np.max(self.hist_green)
                min_count = np.min(self.hist_green)
                max_indices = np.where(self.hist_green == max_count)[0]
                min_indices = np.where(self.hist_green == min_count)[0]
                for idx in max_indices:
                    self.axes.plot(idx, max_count, 'o', color='darkgreen', markersize=8, label=f'Verde Max ({idx}, {int(max_count)})')
                if min_count > 0:
                    for idx in min_indices:
                        self.axes.plot(idx, min_count, 'x', color='darkgreen', markersize=8, label=f'Verde Min ({idx}, {int(min_count)})')
        if self.hist_red is not None and self.cb_red.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_red.flatten(), color='red', alpha=0.7, width=1, label='Rojo')
                if show_values:
                    for i, count in enumerate(self.hist_red.flatten()):
                        if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='red')
            else:
                self.axes.plot(self.hist_red, color='red', label='Rojo')
            if show_max_min and self.hist_red is not None:
                max_count = np.max(self.hist_red)
                min_count = np.min(self.hist_red)
                max_indices = np.where(self.hist_red == max_count)[0]
                min_indices = np.where(self.hist_red == min_count)[0]
                for idx in max_indices:
                    self.axes.plot(idx, max_count, 'o', color='darkred', markersize=8, label=f'Rojo Max ({idx}, {int(max_count)})')
                if min_count > 0:
                    for idx in min_indices:
                        self.axes.plot(idx, min_count, 'x', color='darkred', markersize=8, label=f'Rojo Min ({idx}, {int(min_count)})')
        if self.hist_luminance is not None and self.cb_luminance.isChecked():
            if show_bars:
                self.axes.bar(bin_edges.flatten(), self.hist_luminance.flatten(), color='gray', alpha=0.7, width=1, label='Luminancia')
                if show_values:
                    for i, count in enumerate(self.hist_luminance.flatten()):
                        if count > 0:
                            self.axes.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=6, color='gray')
            else:
                self.axes.plot(self.hist_luminance, color='gray', label='Luminancia')
            if show_max_min and self.hist_luminance is not None:
                max_count = np.max(self.hist_luminance)
                min_count = np.min(self.hist_luminance)
                max_indices = np.where(self.hist_luminance == max_count)[0]
                min_indices = np.where(self.hist_luminance == min_count)[0]
                for idx in max_indices:
                    self.axes.plot(idx, max_count, 'o', color='black', markersize=8, label=f'Lum Max ({idx}, {int(max_count)})')
                if min_count > 0:
                    for idx in min_indices:
                        self.axes.plot(idx, min_count, 'x', color='black', markersize=8, label=f'Lum Min ({idx}, {int(min_count)})')
        if show_grid:
            self.axes.grid(True, alpha=0.3)
        else:
            self.axes.grid(False)
        if show_axis_labels:
            self.axes.set_title("Histograma de Canales y Luminancia")
            self.axes.set_xlabel("Nivel de Intensidad")
            self.axes.set_ylabel("Cantidad de Pixeles")
            self.axes.tick_params(axis='x', labelbottom=True)
            self.axes.tick_params(axis='y', labelleft=True)
            handles, labels = self.axes.get_legend_handles_labels()
            if handles:
                self.axes.legend()
        else:
            self.axes.set_title("")
            self.axes.set_xlabel("")
            self.axes.set_ylabel("")
            self.axes.tick_params(axis='x', labelbottom=False)
            self.axes.tick_params(axis='y', labelleft=False)
            if self.axes.get_legend():
                self.axes.get_legend().remove()
        y_limit_str = self.y_limit_combobox.currentText()
        if y_limit_str:
            try:
                y_limit = int(y_limit_str)
                self.axes.set_ylim(0, y_limit)
            except ValueError:
                print(f"Invalid Y limit value in combobox: {y_limit_str}")
        self.axes.set_xlim(-1, 256)
        plt.tight_layout()
        self.canvas.draw()

    def save_histogram_plot(self):
        # Asegurarse de que haya datos de histograma para guardar
        if self.hist_blue is None and self.hist_green is None and self.hist_red is None and self.hist_luminance is None:
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos de histograma para guardar.")
            return
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar Histograma", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options)
        if file_name:
            self.figure.savefig(file_name)
            QtWidgets.QMessageBox.information(self, "Éxito", "Histograma guardado exitosamente.")


# --- Worker para Umbralización (Lo mantenemos como estaba en tu código) ---
class ThresholdWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(np.ndarray, str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, active_image, threshold_value, invert, parent=None):
        super().__init__(parent)
        self.active_image = active_image
        self.threshold_value = threshold_value
        self.invert = invert
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def is_canceled(self):
        return self._is_canceled

    def run(self):
        try:
            # Convert to grayscale if needed
            if self.active_image.ndim == 3 and self.active_image.shape[2] == 4:
                gray = cv2.cvtColor(self.active_image, COLOR_BGRA2GRAY)
            elif self.active_image.ndim == 3 and self.active_image.shape[2] == 3:
                gray = cv2.cvtColor(self.active_image, COLOR_BGR2GRAY)
            else:
                gray = self.active_image.copy()
            
            # Apply threshold with appropriate type
            max_value = 255
            thresh_type = THRESH_BINARY_INV if self.invert else THRESH_BINARY
            _, thresholded_img = cv2.threshold(gray, self.threshold_value, max_value, thresh_type)

            if not self.is_canceled():
                info_text = f"Umbralización {'Invertida' if self.invert else ''} ({self.threshold_value})"
                self.result_ready.emit(thresholded_img, info_text)

        except Exception as e:
            if not self.is_canceled():
                self.error_occurred.emit(str(e))

    def __del__(self):
        self.wait()

# --- Fin Worker Umbralización ---

# --- Funciones para añadir ruido (Las mantenemos como estaba en tu código) ---
def add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, amount):
    noisy_image = np.copy(image)
    total_image_pixels = image.shape[0] * image.shape[1]
    num_noise_pixels = int(amount * total_image_pixels)

    num_salt = int(salt_vs_pepper_ratio * num_noise_pixels)
    row_coords_salt = np.random.randint(0, image.shape[0], num_salt)
    col_coords_salt = np.random.randint(0, image.shape[1], num_salt)
    for i in range(num_salt):
        if image.ndim == 2:
            noisy_image[row_coords_salt[i], col_coords_salt[i]] = 255
        else:
             noisy_image[row_coords_salt[i], col_coords_salt[i], :] = 255


    num_pepper = num_noise_pixels - num_salt
    row_coords_pepper = np.random.randint(0, image.shape[0], num_pepper)
    col_coords_pepper = np.random.randint(0, image.shape[1], num_pepper)
    for i in range(num_pepper):
         if image.ndim == 2:
             noisy_image[row_coords_pepper[i], col_coords_pepper[i]] = 0
         else:
              noisy_image[row_coords_pepper[i], col_coords_pepper[i], :] = 0

    return noisy_image


def add_gaussian_noise(image, mean=0, std_dev=25):
    img_float = image.astype(np.float32)
    gaussian_noise = np.random.normal(mean, std_dev, img_float.shape).astype(np.float32)
    noisy_image_float = img_float + gaussian_noise
    noisy_image = np.clip(noisy_image_float, 0, 255).astype(np.uint8)
    return noisy_image
# --- Fin Funciones para añadir ruido ---

# Define a consistent intermediate size for thumbnails
THUMBNAIL_INTERMEDIATE_SIZE = (150, 150) # width, height


class ImageProcessor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Imágenes - Gamma + Ruido + Brillo/Contraste") # Updated title
        self.image = None  # Stores the first original loaded image (OpenCV format)
        self.image2 = None # Stores the second original loaded image (OpenCV format)
        self.processed_image = None # Stores the currently displayed image in the central panel (OpenCV format) - This is the result *before* brightness/contrast display adjustments
        self._last_arithmetic_op = None # To track the last arithmetic operation

        self._threshold_worker = None # Referencia al worker de umbralización

        self.active_image = None # Imagen activa para procesamiento base (Imagen 1 o Imagen 2)

        self._is_noisy = False # Bandera para saber si la imagen procesada tiene ruido

        # *** Referencia a la ventana del histograma avanzada ***
        self.histogram_window = None

        # *** Referencia a la ventana de segmentación ***
        self.segmentation_window = None

        # Parámetros para el filtro Canny (valores por defecto)
        self.canny_low_threshold = 100
        self.canny_high_threshold = 200
        self._canny_sliders_visible = False # Track Canny slider visibility


        # *** Parámetros de Brillo y Contraste (Para ajuste de visualización) ***
        self._brightness_value = 0 # Rango: -100 a 100 (aproximado)
        self._contrast_factor = 1.0 # Rango: 0.0 a 2.0 (aproximado)

        # *** Parámetros Aritméticos (Escalar) ***
        self._add_scalar = 0
        self._subtract_scalar = 0
        self._multiply_scalar = 1.0
        self._divide_scalar = 1.0

        # --- Kernels for new edge filters (Robinson, Kirsch, Roberts) ---
        # Robinson kernels (compass masks)
        self.robinson_kernels = {
            "North": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
            "NorthEast": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32),
            "East": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
            "SouthEast": np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
            "South": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
            "SouthWest": np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
            "West": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            "NorthWest": np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
        }

        # Kirsch kernels (compass masks)
        self.kirsch_kernels = {
            "North": np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            "NorthEast": np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            "East": np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            "SouthEast": np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            "South": np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            "SouthWest": np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
            "West": np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            "NorthWest": np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32)
        }

        # Roberts kernels
        self.roberts_kernels = {
            "Diagonal1": np.array([[1, 0], [0, -1]], dtype=np.float32),
            "Diagonal2": np.array([[0, 1], [-1, 0]], dtype=np.float32)
        }

        # Add channel buttons container
        self.channel_buttons_widget = QtWidgets.QWidget()
        self.channel_buttons_layout = QtWidgets.QHBoxLayout()
        self.channel_buttons_widget.setLayout(self.channel_buttons_layout)
        self.channel_buttons_widget.hide()  # Initially hidden
        
        # Store the current color space
        self.current_color_space = None
        self.converted_image = None
        
        self.init_ui()
        self.apply_styles()

        self.hide_all_controls() # Ocultar todos los controles al inicio
        self.histogram_button.setEnabled(False) # Deshabilitar botón de histograma al inicio
        self.set_operation_buttons_enabled(False) # Deshabilitar botones de operación al inicio
        # Habilitar los sliders de brillo y contraste si hay una imagen cargada (se ajusta en set_operation_buttons_enabled)
        self.brightness_slider.setEnabled(False)
        self.contrast_slider.setEnabled(False)

        # Crear grupo para botones direccionales de Robinson
        self.robinson_directions_group = QGroupBox("Máscaras Direccionales de Robinson")
        self.robinson_directions_group.setVisible(False)  # Inicialmente oculto
        robinson_layout = QGridLayout()

        # Crear botones direccionales de Robinson
        self.btn_robinson_norte = QPushButton("Norte")
        self.btn_robinson_noreste = QPushButton("Noreste")
        self.btn_robinson_este = QPushButton("Este")
        self.btn_robinson_sureste = QPushButton("Sureste")
        self.btn_robinson_sur = QPushButton("Sur")
        self.btn_robinson_suroeste = QPushButton("Suroeste")
        self.btn_robinson_oeste = QPushButton("Oeste")
        self.btn_robinson_noroeste = QPushButton("Noroeste")
        self.btn_robinson_completo = QPushButton("Robinson Completo")

        # Organizar botones en grid
        robinson_layout.addWidget(self.btn_robinson_norte, 0, 1)
        robinson_layout.addWidget(self.btn_robinson_noreste, 0, 2)
        robinson_layout.addWidget(self.btn_robinson_este, 1, 2)
        robinson_layout.addWidget(self.btn_robinson_sureste, 2, 2)
        robinson_layout.addWidget(self.btn_robinson_sur, 2, 1)
        robinson_layout.addWidget(self.btn_robinson_suroeste, 2, 0)
        robinson_layout.addWidget(self.btn_robinson_oeste, 1, 0)
        robinson_layout.addWidget(self.btn_robinson_noroeste, 0, 0)
        robinson_layout.addWidget(self.btn_robinson_completo, 1, 1)

        self.robinson_directions_group.setLayout(robinson_layout)

        # Conectar botones direccionales de Robinson
        self.btn_robinson_norte.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_noreste.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_este.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_sureste.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_sur.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_suroeste.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_oeste.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_noroeste.clicked.connect(self.apply_robinson_filter)
        self.btn_robinson_completo.clicked.connect(self.apply_robinson_filter)

        # Agregar el grupo de Robinson al layout de filtros de bordes
        self.edge_filters_group.layout().addWidget(self.robinson_directions_group)


    def init_ui(self):
        # --- Widgets ---

        self.processed_label = ZoomableLabel(self)
        self.processed_label.setText("✨ Visualización Principal ✨\n(Carga una imagen y aplica operaciones)")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_label.setProperty("cssClass", "image-display")

        self.processed_info_label = QLabel("⚙️ Imagen Visualizada")
        self.processed_info_label.setAlignment(Qt.AlignCenter)
        self.processed_info_label.setStyleSheet("font-weight: bold;")


        center_panel_layout = QVBoxLayout()
        center_panel_layout.addWidget(QLabel("<h2>Visualización Principal</h2>", alignment=Qt.AlignCenter))
        center_panel_layout.addWidget(self.processed_info_label)
        center_panel_layout.addWidget(self.processed_label)


        # --- Left Panel (Gallery) ---
        self.load_button1 = QPushButton("📂 Cargar Imagen 1")
        self.load_button1.setToolTip("Cargar la primera imagen para procesar")
        self.image1_thumbnail_label = ClickableLabel(self) # Usamos ClickableLabel
        self.image1_thumbnail_label.setFixedSize(120, 120)
        self.image1_thumbnail_label.setStyleSheet("border: 1px solid lightgray; background-color: #f0f0f0;")
        self.image1_thumbnail_label.setAlignment(Qt.AlignCenter)
        self.image1_thumbnail_label.setText("📁 Imagen 1\n(Click para ver)")
        self.image1_thumbnail_label.setScaledContents(True)
        self.image1_thumbnail_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image1_loaded_label = QLabel("Imagen 1: No Cargada") # NEW: Initialize this label
        self.image1_loaded_label.setAlignment(Qt.AlignCenter) # Align text to center

        self.load_button2 = QPushButton("📂 Cargar Imagen 2")
        self.load_button2.setToolTip("Cargar una segunda imagen para operaciones binarias o como imagen de trabajo")
        self.image2_thumbnail_label = ClickableLabel(self) # Usamos ClickableLabel
        self.image2_thumbnail_label.setFixedSize(120, 120)
        self.image2_thumbnail_label.setStyleSheet("border: 1px solid lightgray; background-color: #f0f0f0;")
        self.image2_thumbnail_label.setAlignment(Qt.AlignCenter)
        self.image2_thumbnail_label.setText("📁 Imagen 2\n(Click para ver)")
        self.image2_thumbnail_label.setScaledContents(True)
        self.image2_thumbnail_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image2_loaded_label = QLabel("Imagen 2: No Cargada") # NEW: Initialize this label
        self.image2_loaded_label.setAlignment(Qt.AlignCenter) # Align text to center


        left_panel_layout = QVBoxLayout()
        left_panel_layout.addWidget(QLabel("<h2>Galería</h2>", alignment=Qt.AlignCenter))
        left_panel_layout.addWidget(self.load_button1)
        left_panel_layout.addWidget(self.image1_thumbnail_label)
        left_panel_layout.addWidget(self.image1_loaded_label) # NEW: Add to layout
        left_panel_layout.addWidget(self.load_button2)
        left_panel_layout.addWidget(self.image2_thumbnail_label)
        left_panel_layout.addWidget(self.image2_loaded_label) # NEW: Add to layout
        left_panel_layout.addStretch(1)


        # --- Right Panel (Controls) ---
        # The main right panel layout will contain the title and the scroll area
        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(QLabel("<h2>Controles</h2>", alignment=Qt.AlignCenter))

        # Create a container widget for all the control groups that will be scrollable
        right_controls_container = QWidget()
        right_controls_layout = QVBoxLayout(right_controls_container) # Layout for the container


        # --- Main Operation Buttons ---
        self.gray_button = QPushButton("🔘 Escala de Grises")
        self.gray_button.setObjectName("gray_button")
        self.gray_button.setStyleSheet("background-color: #808080; color: white; padding: 8px; border-radius: 4px;")
        self.gray_button.setToolTip("Abrir escala de grises")
        
        self.threshold_button = QPushButton("⚫️⚪️ Umbralización")
        self.threshold_button.setObjectName("threshold_button")
        self.threshold_button.setStyleSheet("background-color: #800080; color: white; padding: 8px; border-radius: 4px;")
        self.threshold_button.setToolTip("Abrir umbralización")
        
        self.split_channels_button = QPushButton("🟥🟩🟦 Separar Canales (RGB)")
        self.split_channels_button.setObjectName("split_channels_button")
        self.split_channels_button.setStyleSheet("background-color: #DAA520; color: white; padding: 8px; border-radius: 4px;") # Goldenrod
        self.split_channels_button.setToolTip("Abrir separación de canales RGB")
        
        self.color_space_button = QPushButton("🎨 Espacios de Color")
        self.color_space_button.setObjectName("color_space_button")
        self.color_space_button.setStyleSheet("background-color: #008080; color: white; padding: 8px; border-radius: 4px;") # Teal
        self.color_space_button.setToolTip("Abrir espacios de color")
        
        self.arithmetic_button = QPushButton("🔢 Operaciones Aritméticas")
        self.arithmetic_button.setObjectName("arithmetic_button")
        self.arithmetic_button.setStyleSheet("background-color: #FF0000; color: white; padding: 8px; border-radius: 4px;")
        self.arithmetic_button.setToolTip("Abrir operaciones aritméticas")
        
        self.morph_operations_button = QPushButton("🔧 Operaciones Morfológicas")
        self.morph_operations_button.setObjectName("morph_operations_button")
        self.morph_operations_button.setStyleSheet("background-color: #006400; color: white; padding: 8px; border-radius: 4px;") # DarkGreen
        self.morph_operations_button.setToolTip("Abrir ventana de operaciones morfológicas")
        self.morph_operations_button.clicked.connect(self.toggle_morph_operations_panel)

        self.logical_button = QPushButton("🤝 Operaciones Lógicas")
        self.logical_button.setObjectName("logical_button")
        self.logical_button.setStyleSheet("background-color: #FFA500; color: white; padding: 8px; border-radius: 4px;") # Orange
        self.logical_button.setToolTip("Abrir operaciones lógicas")
        
        self.noise_button = QPushButton("🦠 Añadir Ruido")
        self.noise_button.setObjectName("noise_button")
        self.noise_button.setStyleSheet("background-color: #4B0082; color: white; padding: 8px; border-radius: 4px;") # Indigo
        self.noise_button.setToolTip("Abrir opciones de ruido")
        
        self.filters_button = QPushButton("🪄 Filtros")
        self.filters_button.setObjectName("filters_button")
        self.filters_button.setStyleSheet("background-color: #FF8C00; color: white; padding: 8px; border-radius: 4px;") # DarkOrange
        self.filters_button.setToolTip("Abrir filtros")
        
        self.histogram_button = QPushButton("📊 Histograma")
        self.histogram_button.setObjectName("histogram_button")
        self.histogram_button.setStyleSheet("background-color: #0000FF; color: white; padding: 8px; border-radius: 4px;")
        self.histogram_button.setToolTip("Abrir histograma")
        
        self.connected_components_button = QPushButton("🔗 Etiquetado de Componentes Conexas")
        self.connected_components_button.setObjectName("connected_components_button")
        self.connected_components_button.setStyleSheet("background-color: #8B4513; color: white; padding: 8px; border-radius: 4px;")
        self.connected_components_button.setToolTip("Abrir etiquetado de componentes conexas")
        
        self.segmentation_button = QPushButton("🎯 Segmentación, Detección y Clasificación")
        self.segmentation_button.setObjectName("segmentation_button")
        self.segmentation_button.setStyleSheet("background-color: #FF1493; color: white; padding: 8px; border-radius: 4px;")
        self.segmentation_button.setToolTip("Abrir segmentación, detección y clasificación")
        
        self.load_image1_button = QPushButton("📂 Cargar Imagen 1")
        self.load_image1_button.setObjectName("load_image1_button")
        self.load_image1_button.setStyleSheet("background-color: #4169E1; color: white; padding: 8px; border-radius: 4px;")
        
        self.load_image2_button = QPushButton("📂 Cargar Imagen 2")
        self.load_image2_button.setObjectName("load_image2_button")
        self.load_image2_button.setStyleSheet("background-color: #20B2AA; color: white; padding: 8px; border-radius: 4px;")

        # Add segmentation button
        self.segmentation_button = QPushButton("🎯 Segmentación, Detección y Clasificación")
        self.segmentation_button.setToolTip("Abre la ventana de segmentación con YOLOv8 y Detectron2")
        self.segmentation_button.clicked.connect(self.show_segmentation_window)

        # Añadir botones de operación principal al layout del CONTENEDOR de controles
        for btn in [self.gray_button, self.threshold_button, self.split_channels_button,
            self.color_space_button, # Replaced individual buttons
            self.arithmetic_button, self.morph_operations_button, self.logical_button, self.noise_button,
            self.filters_button, self.histogram_button, self.connected_components_button,
            self.segmentation_button]: # Ahora incluye el botón de operaciones morfológicas
            right_controls_layout.addWidget(btn)

        right_controls_layout.addSpacing(10)


        # --- Grupo de Control de Brillo y Contraste (NUEVO) ---
        self.brightness_contrast_group = QGroupBox("")
        brightness_contrast_layout = QVBoxLayout()
        brightness_contrast_layout.setContentsMargins(5, 5, 5, 5)
        brightness_contrast_layout.setSpacing(5)

        # Control de Brillo
        brightness_hbox = QHBoxLayout()
        brightness_hbox.setSpacing(5)
        self.brightness_label = QLabel(f"Brillo: {self._brightness_value}")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(self._brightness_value)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.setFixedHeight(20)  # Altura fija para el slider
        self.brightness_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Solo expandir horizontalmente

        brightness_hbox.addWidget(self.brightness_label)
        brightness_hbox.addWidget(self.brightness_slider)
        brightness_contrast_layout.addLayout(brightness_hbox)

        # Control de Contraste
        contrast_hbox = QHBoxLayout()
        contrast_hbox.setSpacing(5)
        self.contrast_label = QLabel(f"Contraste: {self._contrast_factor:.1f}")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(int(self._contrast_factor * 100))
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.setFixedHeight(20)  # Altura fija para el slider
        self.contrast_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Solo expandir horizontalmente

        contrast_hbox.addWidget(self.contrast_label)
        contrast_hbox.addWidget(self.contrast_slider)
        brightness_contrast_layout.addLayout(contrast_hbox)

        self.brightness_contrast_group.setLayout(brightness_contrast_layout)
        self.brightness_contrast_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Altura fija para el grupo


        # --- Grupos de Control (Inicialmente Ocultos) ---

        # Grupo: Escala de Grises
        self.gray_controls_group = QGroupBox("Opciones Escala de Grises")
        self.gray_inv_checkbox = QCheckBox("Invertir Colores")
        gray_vbox = QVBoxLayout()
        gray_vbox.addWidget(self.gray_inv_checkbox)
        self.gray_controls_group.setLayout(gray_vbox)

        # Grupo: Umbralización
        self.thresh_controls_group = QGroupBox("Opciones Umbralización")
        self.thresh_slider_label = QLabel("Umbral: 127")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(127)
        self.thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.thresh_slider.setTickInterval(10)
        self.thresh_slider.setFixedHeight(20)  # Altura fija para el slider
        self.thresh_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Solo expandir horizontalmente
        self.thresh_inv_checkbox = QCheckBox("Umbralización Invertida")
        
        # Crear un layout vertical con espaciado reducido
        thresh_vbox = QVBoxLayout()
        thresh_vbox.setSpacing(5)  # Reducir el espaciado vertical
        thresh_vbox.setContentsMargins(5, 5, 5, 5)  # Reducir los márgenes
        
        # Crear un layout horizontal para el label y el slider
        thresh_hbox = QHBoxLayout()
        thresh_hbox.setSpacing(5)  # Espaciado reducido entre label y slider
        thresh_hbox.addWidget(self.thresh_slider_label)
        thresh_hbox.addWidget(self.thresh_slider)
        
        # Añadir los layouts al layout vertical
        thresh_vbox.addLayout(thresh_hbox)
        thresh_vbox.addWidget(self.thresh_inv_checkbox)
        
        self.thresh_controls_group.setLayout(thresh_vbox)
        self.thresh_controls_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Altura fija para el grupo

        # Grupo: Separar Canales
        self.channel_controls_group = QGroupBox("Mostrar Canal RGB")
        self.blue_channel_button = QPushButton("🔵 Azul (B)")
        self.green_channel_button = QPushButton("🟢 Verde (G)")
        self.red_channel_button = QPushButton("🔴 Rojo (R)")
        channel_hbox = QHBoxLayout()
        channel_hbox.addWidget(self.blue_channel_button)
        channel_hbox.addWidget(self.green_channel_button)
        channel_hbox.addWidget(self.red_channel_button)
        self.channel_controls_group.setLayout(channel_hbox)

        # Grupo: Espacios de Color
        self.color_space_controls_group = QtWidgets.QGroupBox("Espacios de Color")
        color_space_vbox = QtWidgets.QVBoxLayout()
        
        # Create color space buttons
        hsv_button_cs = QtWidgets.QPushButton("🌈 HSV")
        lab_button_cs = QtWidgets.QPushButton("🔬 LAB")
        ycrcb_button_cs = QtWidgets.QPushButton("📺 YCrCb")
        hls_button_cs = QtWidgets.QPushButton("💡 HLS")
        yuv_button_cs = QtWidgets.QPushButton("📹 YUV")
        
        # Style the buttons
        for btn in [hsv_button_cs, lab_button_cs, ycrcb_button_cs, hls_button_cs, yuv_button_cs]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a4a4a;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #5a5a5a;
                }
                QPushButton:pressed {
                    background-color: #3a3a3a;
                }
            """)
        
        # Add buttons to layout
        color_space_vbox.addWidget(hsv_button_cs)
        color_space_vbox.addWidget(lab_button_cs)
        color_space_vbox.addWidget(ycrcb_button_cs)
        color_space_vbox.addWidget(hls_button_cs)
        color_space_vbox.addWidget(yuv_button_cs)
        
        # Add channel buttons widget
        self.channel_buttons_widget = QtWidgets.QWidget()
        self.channel_buttons_layout = QtWidgets.QVBoxLayout()
        self.channel_buttons_widget.setLayout(self.channel_buttons_layout)
        self.channel_buttons_widget.hide()  # Initially hidden
        color_space_vbox.addWidget(self.channel_buttons_widget)
        
        self.color_space_controls_group.setLayout(color_space_vbox)
        
        # Connect color space buttons
        hsv_button_cs.clicked.connect(lambda: self.show_specific_color_space('HSV'))
        lab_button_cs.clicked.connect(lambda: self.show_specific_color_space('LAB'))
        ycrcb_button_cs.clicked.connect(lambda: self.show_specific_color_space('YCrCb'))
        hls_button_cs.clicked.connect(lambda: self.show_specific_color_space('HLS'))
        yuv_button_cs.clicked.connect(lambda: self.show_specific_color_space('YUV'))

        # Grupo: Operaciones Aritméticas (Escalar)
        self.arithmetic_controls_group = QGroupBox("Opciones Aritméticas (Escalar)")
        _add_scalar = 0
        _subtract_scalar = 0
        _multiply_scalar = 1.0
        _divide_scalar = 1.0

        arithmetic_layout = QVBoxLayout()

        add_overall_vbox = QVBoxLayout()
        add_controls_hbox = QHBoxLayout()

        self.add_button = QPushButton("➕ Suma")
        self.add_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        self.add_scalar_input = QLineEdit(str(_add_scalar))
        self.add_scalar_input.setValidator(QDoubleValidator())
        self.add_scalar_input.setMinimumWidth(40)

        self.add_scalar_slider = QSlider(Qt.Horizontal)
        self.add_scalar_slider.setRange(-255, 255)
        self.add_scalar_slider.setValue(int(np.clip(_add_scalar, -255, 255)))
        self.add_scalar_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        add_controls_hbox.addWidget(self.add_button)
        add_controls_hbox.addWidget(self.add_scalar_input)
        add_controls_hbox.addWidget(self.add_scalar_slider)

        add_overall_vbox.addLayout(add_controls_hbox)

        self.add_scalar_label = QLabel(f"Valor: {_add_scalar}")
        self.add_scalar_label.setAlignment(Qt.AlignCenter)
        self.add_scalar_label.setProperty("cssClass", "arithmetic-scalar-label")
        add_overall_vbox.addWidget(self.add_scalar_label)

        arithmetic_layout.addLayout(add_overall_vbox)

        subtract_overall_vbox = QVBoxLayout()
        subtract_controls_hbox = QHBoxLayout()

        self.subtract_button = QPushButton("➖ Resta")
        self.subtract_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        self.subtract_scalar_input = QLineEdit(str(_subtract_scalar))
        self.subtract_scalar_input.setValidator(QDoubleValidator())
        self.subtract_scalar_input.setMinimumWidth(40)

        self.subtract_scalar_slider = QSlider(Qt.Horizontal)
        self.subtract_scalar_slider.setRange(-255, 255)
        self.subtract_scalar_slider.setValue(int(np.clip(_subtract_scalar, -255, 255)))
        self.subtract_scalar_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        subtract_controls_hbox.addWidget(self.subtract_button)
        subtract_controls_hbox.addWidget(self.subtract_scalar_input)
        subtract_controls_hbox.addWidget(self.subtract_scalar_slider)

        subtract_overall_vbox.addLayout(subtract_controls_hbox)

        self.subtract_scalar_label = QLabel(f"Valor: {_subtract_scalar}")
        self.subtract_scalar_label.setAlignment(Qt.AlignCenter)
        self.subtract_scalar_label.setProperty("cssClass", "arithmetic-scalar-label")
        subtract_overall_vbox.addWidget(self.subtract_scalar_label)

        arithmetic_layout.addLayout(subtract_overall_vbox)

        multiply_vbox = QVBoxLayout()
        multiply_hbox = QHBoxLayout()
        self.multiply_button = QPushButton("✖️ Multiplicación")
        self.multiply_scalar_input = QLineEdit(str(_multiply_scalar))
        self.multiply_scalar_input.setValidator(QDoubleValidator())
        multiply_hbox.addWidget(self.multiply_button)
        multiply_hbox.addWidget(self.multiply_scalar_input)
        multiply_vbox.addLayout(multiply_hbox)
        self.multiply_scalar_label = QLabel(f"Valor: {_multiply_scalar}")
        self.multiply_scalar_label.setAlignment(Qt.AlignCenter)
        self.multiply_scalar_label.setProperty("cssClass", "arithmetic-scalar-label")
        multiply_vbox.addWidget(self.multiply_scalar_label)
        arithmetic_layout.addLayout(multiply_vbox)

        divide_vbox = QVBoxLayout()
        divide_hbox = QHBoxLayout()
        self.divide_button = QPushButton("➗ División")
        self.divide_scalar_input = QLineEdit(str(_divide_scalar))
        self.divide_scalar_input.setValidator(QDoubleValidator())
        divide_hbox.addWidget(self.divide_button)
        divide_hbox.addWidget(self.divide_scalar_input)
        divide_vbox.addLayout(divide_hbox)
        self.divide_scalar_label = QLabel(f"Valor: {_divide_scalar}")
        self.divide_scalar_label.setAlignment(Qt.AlignCenter)
        self.divide_scalar_label.setProperty("cssClass", "arithmetic-scalar-label")
        divide_vbox.addWidget(self.divide_scalar_label)
        arithmetic_layout.addLayout(divide_vbox)

        self.arithmetic_controls_group.setLayout(arithmetic_layout)


        # Grupo: Operaciones Lógicas (Imagen 1 vs Imagen 2)
        self.logical_controls_group = QGroupBox("Operaciones Lógicas (Im1 vs Im2, y NOT en Activa)")
        self.and_button = QPushButton("🤝 AND (Im1, Im2)")
        self.or_button = QPushButton("🔀 OR (Im1, Im2)")
        self.xor_button = QPushButton("↔️ XOR (Im1, Im2)")
        self.not_button = QPushButton("🔄 NOT (Imagen Activa)")
        logical_vbox = QVBoxLayout()
        logical_vbox.addWidget(self.and_button)
        logical_vbox.addWidget(self.or_button)
        logical_vbox.addWidget(self.xor_button)
        logical_vbox.addWidget(self.not_button)
        self.logical_controls_group.setLayout(logical_vbox)

        # --- Grupo de Control de Ruido ---
        self.noise_controls_group = QGroupBox("Opciones de Ruido")
        noise_layout = QVBoxLayout()

        # Selección de Tipo de Ruido
        noise_type_layout = QHBoxLayout()
        noise_type_label = QLabel("Tipo:")
        self.radio_salt_pepper = QRadioButton("Sal y Pimienta")
        self.radio_gaussian = QRadioButton("Gaussiano")
        self.radio_gaussian.setChecked(True) # Gaussiano por defecto
        noise_type_layout.addWidget(noise_type_label)
        noise_type_layout.addWidget(self.radio_salt_pepper)
        noise_type_layout.addWidget(self.radio_gaussian)
        noise_type_layout.addStretch()

        noise_layout.addLayout(noise_type_layout)

        # Parámetros de Sal y Pimienta
        self.salt_pepper_params_layout = QVBoxLayout()
        self.salt_pepper_params_layout.setSpacing(5)  # Reducir el espaciado vertical
        self.sp_amount_label = QLabel("Cantidad: 0.0 %")
        self.sp_amount_slider = QSlider(Qt.Horizontal)
        self.sp_amount_slider.setRange(0, 100)
        self.sp_amount_slider.setValue(0)
        self.sp_amount_slider.setTickPosition(QSlider.TicksBelow)
        self.sp_amount_slider.setTickInterval(10)
        self.sp_amount_slider.setFixedHeight(20)  # Altura fija para el slider
        self.sp_amount_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Solo expandir horizontalmente

        self.sp_ratio_label = QLabel("Proporción Sal: 50.0 %")
        self.sp_ratio_slider = QSlider(Qt.Horizontal)
        self.sp_ratio_slider.setRange(0, 100)
        self.sp_ratio_slider.setValue(50)
        self.sp_ratio_slider.setTickPosition(QSlider.TicksBelow)
        self.sp_ratio_slider.setTickInterval(10)
        self.sp_ratio_slider.setFixedHeight(20)  # Altura fija para el slider
        self.sp_ratio_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Solo expandir horizontalmente

        self.salt_pepper_params_layout.addWidget(self.sp_amount_label)
        self.salt_pepper_params_layout.addWidget(self.sp_amount_slider)
        self.salt_pepper_params_layout.addWidget(self.sp_ratio_label)
        self.salt_pepper_params_layout.addWidget(self.sp_ratio_slider)


        # Parámetros Gaussianos
        self.gaussian_params_layout = QVBoxLayout()
        self.gaussian_params_layout.setSpacing(5)  # Reducir el espaciado vertical
        self.gauss_stddev_label = QLabel("Desv. Estándar: 25")
        self.gauss_stddev_slider = QSlider(Qt.Horizontal)
        self.gauss_stddev_slider.setRange(0, 100)
        self.gauss_stddev_slider.setValue(25)
        self.gauss_stddev_slider.setTickPosition(QSlider.TicksBelow)
        self.gauss_stddev_slider.setTickInterval(10)
        self.gauss_stddev_slider.setFixedHeight(20)  # Altura fija para el slider
        self.gauss_stddev_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Solo expandir horizontalmente

        self.gaussian_params_layout.addWidget(self.gauss_stddev_label)
        self.gaussian_params_layout.addWidget(self.gauss_stddev_slider)


        # Botón Aplicar Ruido
        self.apply_noise_button = QPushButton("✨ Aplicar Ruido Seleccionado")


        noise_layout.addLayout(self.salt_pepper_params_layout)
        noise_layout.addLayout(self.gaussian_params_layout)
        noise_layout.addWidget(self.apply_noise_button)
        noise_layout.addStretch()

        self.noise_controls_group.setLayout(noise_layout)
        # Fin Grupo de Control de Ruido


        # --- Controles de Filtro (Estructura de Categorías) ---

        # 1. Botones de Categoría de Filtro (aparecen al hacer clic en "Filtros")
        self.linear_filters_button = QPushButton("1. Filtros Lineales (Pasa-bajas)")
        self.nonlinear_filters_button = QPushButton("2. Filtros No Lineales (De Orden)")
        # Nuevo botón de categoría para Filtros de Borde
        self.edge_filters_button = QPushButton("3. Filtros Detectores de Borde")


        # 2. Grupos para cada Categoría de Filtro

        # Grupo: Filtros Lineales
        self.linear_filters_group = QGroupBox("Filtros Lineales")
        self.blur_button = QPushButton("💧 Filtro Promediador (Blur)")
        self.blur_button.setObjectName("blur_button")
        self.blur_button.setStyleSheet("background-color: #6495ED; color: white; padding: 8px; border-radius: 4px;") # CornflowerBlue
        
        self.weighted_avg_button = QPushButton("⚖️ Filtro Promediador Pesado")
        self.weighted_avg_button.setObjectName("weighted_avg_button")
        self.weighted_avg_button.setStyleSheet("background-color: #40E0D0; color: white; padding: 8px; border-radius: 4px;") # Turquoise
        
        self.gauss_button = QPushButton("🌫️ Filtro Gaussiano")
        self.gauss_button.setObjectName("gauss_button")
        self.gauss_button.setStyleSheet("background-color: #CCCCFF; color: black; padding: 8px; border-radius: 4px;") # Lavender
        
        linear_filters_layout = QVBoxLayout()
        linear_filters_layout.addWidget(self.blur_button)
        linear_filters_layout.addWidget(self.weighted_avg_button)
        linear_filters_layout.addWidget(self.gauss_button)
        self.linear_filters_group.setLayout(linear_filters_layout)

        # Grupo: Filtros No Lineales
        self.nonlinear_filters_group = QGroupBox("Filtros No Lineales")
        self.median_button = QPushButton("🧼 Filtro de Mediana")
        self.median_button.setObjectName("median_button")
        self.median_button.setStyleSheet("background-color: #DEB887; color: black; padding: 8px; border-radius: 4px;") # BurlyWood
        
        self.mode_filter_button = QPushButton("📊 Filtro de Moda")
        self.mode_filter_button.setObjectName("mode_filter_button")
        self.mode_filter_button.setStyleSheet("background-color: #F08080; color: white; padding: 8px; border-radius: 4px;") # LightCoral
        
        self.max_filter_button = QPushButton("🌟 Filtro Máximo")
        self.max_filter_button.setObjectName("max_filter_button")
        self.max_filter_button.setStyleSheet("background-color: #FFD700; color: black; padding: 8px; border-radius: 4px;") # Gold
        
        self.min_filter_button = QPushButton("🍂 Filtro Mínimo")
        self.min_filter_button.setObjectName("min_filter_button")
        self.min_filter_button.setStyleSheet("background-color: #A0522D; color: white; padding: 8px; border-radius: 4px;") # Sienna
        
        nonlinear_filters_layout = QVBoxLayout()
        nonlinear_filters_layout.addWidget(self.median_button)
        nonlinear_filters_layout.addWidget(self.mode_filter_button)
        nonlinear_filters_layout.addWidget(self.max_filter_button)
        nonlinear_filters_layout.addWidget(self.min_filter_button)
        self.nonlinear_filters_group.setLayout(nonlinear_filters_layout)

        # Deshabilitar botón de Moda si scipy no está disponible
        if not SCIPY_AVAILABLE:
            self.mode_filter_button.setEnabled(False)
            self.mode_filter_button.setToolTip("Requiere instalar la librería 'scipy'")


        # --- Grupo: Filtros Detectores de Borde (Nuevo) ---
        self.edge_filters_group = QGroupBox("Filtros Detectores de Borde")
        edge_filters_layout = QGridLayout() # Usamos QGridLayout para 2 columnas

        # Botones de Filtro de Borde individuales
        self.btn_canny = QPushButton("✏️ Canny")
        self.btn_canny.setObjectName("btn_canny")
        self.btn_canny.setStyleSheet("background-color: #483D8B; color: white; padding: 8px; border-radius: 4px;") # DarkSlateBlue
        
        self.btn_horizontal = QPushButton("↔️ Borde Horizontal")
        self.btn_horizontal.setObjectName("btn_horizontal")
        self.btn_horizontal.setStyleSheet("background-color: #2F4F4F; color: white; padding: 8px; border-radius: 4px;") # DarkSlateGray
        
        self.btn_vertical = QPushButton("↕️ Borde Vertical")
        self.btn_vertical.setObjectName("btn_vertical")
        self.btn_vertical.setStyleSheet("background-color: #8FBC8F; color: black; padding: 8px; border-radius: 4px;") # DarkSeaGreen
        
        self.btn_sobel = QPushButton("✨ Sobel")
        self.btn_sobel.setObjectName("btn_sobel")
        self.btn_sobel.setStyleSheet("background-color: #BA55D3; color: white; padding: 8px; border-radius: 4px;") # MediumOrchid
        
        self.btn_prewitt = QPushButton("📐 Prewitt")
        self.btn_prewitt.setObjectName("btn_prewitt")
        self.btn_prewitt.setStyleSheet("background-color: #BDB76B; color: black; padding: 8px; border-radius: 4px;") # DarkKhaki
        
        self.btn_laplacian = QPushButton("⚪⚫ Laplaciano")
        self.btn_laplacian.setObjectName("btn_laplacian")
        self.btn_laplacian.setStyleSheet("background-color: #778899; color: white; padding: 8px; border-radius: 4px;") # LightSlateGray
        
        self.btn_scharr = QPushButton("⚡ Scharr")
        self.btn_scharr.setObjectName("btn_scharr")
        self.btn_scharr.setStyleSheet("background-color: #CD5C5C; color: white; padding: 8px; border-radius: 4px;") # IndianRed
        
        # Añadir botones para Robinson, Kirsch, y Roberts
        self.btn_robinson = QPushButton("🧭 Robinson")
        self.btn_robinson.setObjectName("btn_robinson")
        self.btn_robinson.setStyleSheet("background-color: #E9967A; color: black; padding: 8px; border-radius: 4px;") # DarkSalmon
        
        self.btn_kirsch = QPushButton("☸️ Kirsch")
        self.btn_kirsch.setObjectName("btn_kirsch")
        self.btn_kirsch.setStyleSheet("background-color: #F4A460; color: black; padding: 8px; border-radius: 4px;") # SandyBrown
        
        self.btn_roberts = QPushButton("✝️ Roberts")
        self.btn_roberts.setObjectName("btn_roberts")
        self.btn_roberts.setStyleSheet("background-color: #DA70D6; color: white; padding: 8px; border-radius: 4px;") # Orchid
        
        # Canny Parameters (Labels and Sliders)
        self.canny_low_thresh_label = QLabel(f"Umbral Bajo:") # Etiqueta sin valor inicial aquí
        self.canny_low_thresh_slider = QSlider(Qt.Horizontal)
        self.canny_low_thresh_slider.setRange(0, 255)
        self.canny_low_thresh_slider.setValue(self.canny_low_threshold)
        self.canny_low_thresh_slider.setToolTip("Umbral bajo para el detector Canny")

        self.canny_high_thresh_label = QLabel(f"Umbral Alto:") # Etiqueta sin valor inicial aquí
        self.canny_high_thresh_slider = QSlider(Qt.Horizontal)
        self.canny_high_thresh_slider.setRange(0, 255)
        self.canny_high_thresh_slider.setValue(self.canny_high_threshold)
        self.canny_high_thresh_slider.setToolTip("Umbral alto para el detector Canny")


        # Add buttons to the grid layout (in 2 columns)
        row = 0
        col = 0
        edge_filters_layout.addWidget(self.btn_canny, row, col)
        col += 1
        edge_filters_layout.addWidget(self.btn_horizontal, row, col)
        row += 1
        col = 0
        edge_filters_layout.addWidget(self.btn_vertical, row, col)
        col += 1
        edge_filters_layout.addWidget(self.btn_sobel, row, col)
        row += 1
        col = 0
        edge_filters_layout.addWidget(self.btn_prewitt, row, col)
        col += 1
        edge_filters_layout.addWidget(self.btn_laplacian, row, col)
        row += 1
        col = 0
        edge_filters_layout.addWidget(self.btn_scharr, row, col)
        col += 1
        edge_filters_layout.addWidget(self.btn_robinson, row, col) # Add Robinson button
        row += 1
        col = 0
        edge_filters_layout.addWidget(self.btn_kirsch, row, col) # Add Kirsch button
        col += 1
        edge_filters_layout.addWidget(self.btn_roberts, row, col) # Add Roberts button


        # Add Canny sliders and labels to the layout (placing label and slider on the same row)
        # Start adding Canny controls after the last filter button row
        row += 1
        col = 0
        edge_filters_layout.addWidget(self.canny_low_thresh_label, row, col) # Label in col 0
        col += 1
        edge_filters_layout.addWidget(self.canny_low_thresh_slider, row, col) # Slider in col 1 (spans remaining space)

        row += 1 # Move to the next row for the high threshold controls
        col = 0
        edge_filters_layout.addWidget(self.canny_high_thresh_label, row, col) # Label in col 0
        col += 1
        edge_filters_layout.addWidget(self.canny_high_thresh_slider, row, col) # Slider in col 1


        # Initially hide the Canny threshold controls
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)


        self.edge_filters_group.setLayout(edge_filters_layout)
        # --- Fin Grupo: Filtros Detectores de Borde ---


        # Grupo: Histograma (Botones para elegir de qué imagen mostrar histograma)
        self.histogram_controls_group = QGroupBox("Generar Histograma para:")
        self.hist_original_button = QPushButton("Imagen Original 1")
        self.hist_processed_button = QPushButton("Imagen Procesada (Panel Central)")
        hist_vbox = QVBoxLayout()
        hist_vbox.addWidget(self.hist_original_button)
        hist_vbox.addWidget(self.hist_processed_button)
        self.histogram_controls_group.setLayout(hist_vbox)

        # --- NEW: Grupo de Controles para Etiquetado de Componentes Conexas ---
        self.connected_components_group = QGroupBox("Etiquetado de Componentes Conexas")
        connected_components_layout = QVBoxLayout()

        # Buttons for showing specific results
        self.btn_bin = QPushButton("Mostrar Imagen Binarizada")
        self.btn_bin.setObjectName("btn_bin")
        self.btn_bin.setStyleSheet("background-color: #AFEEEE; color: black; padding: 8px; border-radius: 4px;") # PaleTurquoise
        
        self.btn_4 = QPushButton("Mostrar Etiquetas Vecindad-4")
        self.btn_4.setObjectName("btn_4")
        self.btn_4.setStyleSheet("background-color: #7FFFD4; color: black; padding: 8px; border-radius: 4px;") # Aquamarine
        
        self.btn_8 = QPushButton("Mostrar Etiquetas Vecindad-8")
        self.btn_8.setObjectName("btn_8")
        self.btn_8.setStyleSheet("background-color: #40E0D0; color: black; padding: 8px; border-radius: 4px;") # Turquoise
        
        self.btn_contours = QPushButton("Mostrar Contornos y Numeración")
        self.btn_contours.setObjectName("btn_contours")
        self.btn_contours.setStyleSheet("background-color: #66CDAA; color: white; padding: 8px; border-radius: 4px;") # MediumAquaMarine
        
        connected_components_layout.addWidget(self.btn_bin)
        connected_components_layout.addWidget(self.btn_4)
        connected_components_layout.addWidget(self.btn_8)
        connected_components_layout.addWidget(self.btn_contours)

        self.connected_components_group.setLayout(connected_components_layout)
        # --- FIN NEW Grupo de Controles para Etiquetado de Componentes Conexas ---


        # --- Añadir grupos de control al layout del CONTENEDOR de controles ---
        # Add brightness/contrast group permanently to the right panel container
        right_controls_layout.addWidget(self.brightness_contrast_group) # Add the new group here

        right_controls_layout.addWidget(self.gray_controls_group)
        right_controls_layout.addWidget(self.thresh_controls_group)
        right_controls_layout.addWidget(self.channel_controls_group)
        right_controls_layout.addWidget(self.color_space_controls_group)
        right_controls_layout.addWidget(self.arithmetic_controls_group)
        right_controls_layout.addWidget(self.logical_controls_group)
        right_controls_layout.addWidget(self.noise_controls_group)

        # --- NEW: Morphological Operations Group ---
        self.morph_operations_group = QGroupBox("Operaciones Morfológicas")
        morph_layout = QVBoxLayout()

        # --- Operadores morfológicos ---
        op_groupbox = QGroupBox("Operador Morfológico")
        op_layout = QVBoxLayout()
        self.morph_op_buttons = {
            "Dilatación": QRadioButton("Dilatación"),
            "Erosión": QRadioButton("Erosión"),
            "Apertura": QRadioButton("Apertura"),
            "Cierre": QRadioButton("Cierre"),
            "Gradiente": QRadioButton("Gradiente"),  # Nuevo botón para Gradiente
        }
        self.morph_op_buttons["Dilatación"].setChecked(True)
        for rb in self.morph_op_buttons.values():
            rb.toggled.connect(self.apply_morphology_panel)
            op_layout.addWidget(rb)
        op_groupbox.setLayout(op_layout)
        morph_layout.addWidget(op_groupbox)

        # Botón extra para operaciones morfológicas avanzadas
        self.btn_advanced_morph = QPushButton("Operaciones Morfológicas Avanzadas")
        self.btn_advanced_morph.setObjectName("btn_advanced_morph")
        self.btn_advanced_morph.setStyleSheet("background-color: #7B68EE; color: white; padding: 8px; border-radius: 4px;")  # MediumSlateBlue
        morph_layout.addWidget(self.btn_advanced_morph)

        # --- Panel de operaciones avanzadas (creado solo al pulsar el botón) ---
        def show_advanced_morph_dialog():
            dialog = QDialog(self)
            dialog.setWindowTitle("Operaciones Morfológicas Avanzadas")
            layout = QVBoxLayout(dialog)
            btn_distance = QPushButton("Transformada de Distancia", dialog)
            btn_watershed = QPushButton("Transformada Watershed", dialog)
            btn_distance.setStyleSheet("background-color: #4682B4; color: white; padding: 8px; border-radius: 4px;")
            btn_watershed.setStyleSheet("background-color: #228B22; color: white; padding: 8px; border-radius: 4px;")
            layout.addWidget(btn_distance)
            layout.addWidget(btn_watershed)
            dialog.setLayout(layout)

            def apply_distance_transform():
                if self.active_image is None:
                    QMessageBox.warning(self, "Advertencia", "No hay imagen activa para procesar.")
                    return
                gray = cv2.cvtColor(self.active_image, cv2.COLOR_BGR2GRAY) if self.active_image.ndim == 3 else self.active_image.copy()
                _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
                dist_display = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                # Display in both the morph panel and the main visualization panel
                self._display_cv_image_in_label(dist_display, self.morph_processed_label)
                self._display_cv_image_in_label(dist_display, self.processed_label)
                self.processed_image = dist_display.copy()  # Update app state
                dialog.accept()

            def apply_watershed():
                if self.active_image is None:
                    QMessageBox.warning(self, "Advertencia", "No hay imagen activa para procesar.")
                    return
                gray = cv2.cvtColor(self.active_image, cv2.COLOR_BGR2GRAY) if self.active_image.ndim == 3 else self.active_image.copy()
                _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((3, 3), np.uint8)
                dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                sure_bg = cv2.dilate(binary_image, kernel, iterations=3)
                unknown = cv2.subtract(sure_bg, sure_fg)
                ret, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                watershed_input_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                markers = cv2.watershed(watershed_input_color, markers)
                watershed_result_image = gray.copy()
                watershed_result_image[markers == -1] = 128
                # Display in both the morph panel and the main visualization panel
                self._display_cv_image_in_label(watershed_result_image, self.morph_processed_label)
                self._display_cv_image_in_label(watershed_result_image, self.processed_label)
                self.processed_image = watershed_result_image.copy()  # Update app state
                dialog.accept()

            btn_distance.clicked.connect(apply_distance_transform)
            btn_watershed.clicked.connect(apply_watershed)
            dialog.exec_()
        self.btn_advanced_morph.clicked.connect(show_advanced_morph_dialog)

        # Método para mostrar imágenes de OpenCV en un QLabel
        def _display_cv_image_in_label(self, cv_img, label):
            # Admite imágenes en escala de grises (2D) y color (3D)
            from PyQt5.QtGui import QImage, QPixmap
            if cv_img is None:
                label.clear()
                label.setText("Sin imagen")
                return
            if len(cv_img.shape) == 2:
                height, width = cv_img.shape
                bytes_per_line = width
                q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = cv_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._display_cv_image_in_label = _display_cv_image_in_label.__get__(self)

        # --- Slider tamaño kernel ---
        kernel_groupbox = QGroupBox("Tamaño del Kernel (n x n)")
        kernel_layout = QVBoxLayout()
        self.morph_kernel_slider = QSlider(Qt.Horizontal)
        self.morph_kernel_slider.setMinimum(1)
        self.morph_kernel_slider.setMaximum(31)
        self.morph_kernel_slider.setValue(5)
        self.morph_kernel_slider.setTickInterval(2)
        self.morph_kernel_slider.setTickPosition(QSlider.TicksBelow)
        self.morph_kernel_slider.valueChanged.connect(self.update_kernel_label_and_apply_panel)
        self.morph_kernel_label = QLabel(f"Tamaño: {self.morph_kernel_slider.value()}x{self.morph_kernel_slider.value()}")
        self.morph_kernel_label.setAlignment(Qt.AlignCenter)
        kernel_layout.addWidget(self.morph_kernel_label)
        kernel_layout.addWidget(self.morph_kernel_slider)
        kernel_groupbox.setLayout(kernel_layout)
        morph_layout.addWidget(kernel_groupbox)

        # --- Etiquetas de imagen ---
        img_layout = QHBoxLayout()
        self.morph_original_label = QLabel("Imagen Original")
        self.morph_original_label.setAlignment(Qt.AlignCenter)
        self.morph_original_label.setStyleSheet("border: 2px solid #aaa; background-color: #f0f0f0;")
        self.morph_original_label.setMinimumSize(180, 180)
        self.morph_processed_label = QLabel("Imagen Procesada")
        self.morph_processed_label.setAlignment(Qt.AlignCenter)
        self.morph_processed_label.setStyleSheet("border: 2px solid #aaa; background-color: #f0f0f0;")
        self.morph_processed_label.setMinimumSize(180, 180)
        self.morph_processed_label.mousePressEvent = self.set_main_view_to_morph_result
        img_layout.addWidget(self.morph_original_label)
        img_layout.addWidget(self.morph_processed_label)
        morph_layout.addLayout(img_layout)

        self.morph_operations_group.setLayout(morph_layout)
        self.morph_operations_group.setVisible(False)  # Oculto por defecto
        right_controls_layout.addWidget(self.morph_operations_group)
        # --- FIN NEW ---

        # Grupos de filtro específicos (ocultos inicialmente)
        right_controls_layout.addWidget(self.linear_filters_button)
        right_controls_layout.addWidget(self.linear_filters_group)
        right_controls_layout.addWidget(self.nonlinear_filters_button)
        right_controls_layout.addWidget(self.nonlinear_filters_group)
        # Añadir el nuevo botón de categoría de bordes y su grupo
        right_controls_layout.addWidget(self.edge_filters_button)
        right_controls_layout.addWidget(self.edge_filters_group)

        # Añadir el grupo de controles de histograma
        right_controls_layout.addWidget(self.histogram_controls_group)

        # --- NEW: Add the Connected Components group ---
        right_controls_layout.addWidget(self.connected_components_group)
        # --- FIN NEW ---

        right_controls_layout.addStretch(1) # Empuja los controles hacia arriba dentro del CONTENEDOR


        # --- Crear QScrollArea para el contenido del panel derecho ---
        self.right_panel_scroll_area = QScrollArea()
        self.right_panel_scroll_area.setWidgetResizable(True) # Allow the widget inside to be resized
        self.right_panel_scroll_area.setWidget(right_controls_container) # Set the container as the scrollable widget
        # Set size policy for the scroll area within the main right panel layout
        self.right_panel_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # --- Layout Principal (Usando QSplitter) ---
        self.main_splitter = QSplitter(Qt.Horizontal)

        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)

        center_panel_widget = QWidget()
        center_panel_widget.setLayout(center_panel_layout)

        # The right panel widget now just contains the scroll area
        right_panel_widget = QWidget()
        right_panel_main_layout = QVBoxLayout(right_panel_widget)
        right_panel_main_layout.setContentsMargins(0, 0, 0, 0) # No extra margins needed here
        right_panel_main_layout.setSpacing(0) # No extra spacing needed here

        # Add the initial "Controles" title to the main right panel layout (outside the scroll area)
        right_panel_main_layout.addWidget(QLabel("<h2>Controles</h2>", alignment=Qt.AlignCenter))
        # Add the scroll area containing all other controls
        right_panel_main_layout.addWidget(self.right_panel_scroll_area)


        self.main_splitter.addWidget(left_panel_widget)
        self.main_splitter.addWidget(center_panel_widget)
        self.main_splitter.addWidget(right_panel_widget) # Add the widget containing the scroll area

        self.main_splitter.setSizes([150, 600, 250]) # Ajustar tamaños si es necesario
        self.main_splitter.setStretchFactor(1, 1) # Permitir que el panel central se expanda


        window_layout = QVBoxLayout()
        window_layout.addWidget(self.main_splitter)

        self.setLayout(window_layout)
        self.setMinimumSize(1000, 700) # Tamaño mínimo de la ventana


        # --- Conectar señales ---
        self.load_button1.clicked.connect(self.load_image1)
        self.load_button2.clicked.connect(self.load_image2)

        # Conexiones para los botones de operación principal
        self.gray_button.clicked.connect(self.show_gray_controls)
        self.threshold_button.clicked.connect(self.show_threshold_controls)
        self.split_channels_button.clicked.connect(self.show_channel_controls)
        self.color_space_button.clicked.connect(self.show_color_space_controls)
        self.arithmetic_button.clicked.connect(self.show_arithmetic_controls)
        self.logical_button.clicked.connect(self.show_logical_controls)
        self.noise_button.clicked.connect(self.show_noise_controls)
        self.filters_button.clicked.connect(self.show_filter_categories) # Filtros button shows categories
        # El botón principal de histograma ahora solo muestra el grupo de controles
        self.histogram_button.clicked.connect(self.show_histogram_controls)

        # *** Conexiones para los sliders de Brillo y Contraste ***
        self.brightness_slider.valueChanged.connect(self.update_brightness_label)
        self.brightness_slider.valueChanged.connect(self.apply_display_adjustments)
        self.contrast_slider.valueChanged.connect(self.update_contrast_label)
        self.contrast_slider.valueChanged.connect(self.apply_display_adjustments)


        # Conexiones para la estructura de categorías de filtro
        self.linear_filters_button.clicked.connect(self.show_linear_filters_group)
        self.nonlinear_filters_button.clicked.connect(self.show_nonlinear_filters_group)
        # Conexión para el nuevo botón de categoría de bordes
        self.edge_filters_button.clicked.connect(self.show_edge_filters_group)


        # Conexiones para los botones de filtro específicos (dentro de sus grupos)
        # Lineales
        self.blur_button.clicked.connect(self.apply_blur)
        self.weighted_avg_button.clicked.connect(self.apply_weighted_average)
        self.gauss_button.clicked.connect(self.apply_gaussian)
        # No Lineales
        self.median_button.clicked.connect(self.apply_median)
        self.mode_filter_button.clicked.connect(self.apply_mode_filter)
        self.max_filter_button.clicked.connect(self.apply_max_filter)
        self.min_filter_button.clicked.connect(self.apply_min_filter)

        # --- Conexiones para los botones de Filtro de Borde ---
        # Modificado para llamar a toggle_canny_controls_visibility y luego aplicar canny
        self.btn_canny.clicked.connect(self.toggle_canny_controls_visibility)
        self.btn_canny.clicked.connect(self.apply_canny_filter)

        self.btn_horizontal.clicked.connect(self.apply_horizontal_filter)
        self.btn_vertical.clicked.connect(self.apply_vertical_filter)
        self.btn_sobel.clicked.connect(self.apply_sobel_filter)
        self.btn_prewitt.clicked.connect(self.apply_prewitt_filter)
        self.btn_laplacian.clicked.connect(self.apply_laplacian_filter)
        self.btn_scharr.clicked.connect(self.apply_scharr_filter)

        # --- Add connections for new edge filters ---
        self.btn_robinson.clicked.connect(self.apply_robinson_filter)
        self.btn_kirsch.clicked.connect(self.apply_kirsch_filter)
        self.btn_roberts.clicked.connect(self.apply_roberts_filter)
        # --- End add connections for new edge filters ---

        # Conexiones de miniaturas (clic para mostrar en panel principal y actualizar active_image)
        self.image1_thumbnail_label.clicked.connect(self.show_original_in_processed)
        self.image2_thumbnail_label.clicked.connect(self.show_image2_in_processed)

        # --- Conexiones de Controles Condicionales ---

        # Escala de Grises
        self.gray_inv_checkbox.stateChanged.connect(self.apply_gray)

        # Umbralización (Usando ThresholdWorker)
        self.thresh_slider.valueChanged.connect(self.update_threshold_label)
        self.thresh_slider.valueChanged.connect(self._start_threshold_worker)
        self.thresh_inv_checkbox.stateChanged.connect(self._start_threshold_worker)

        # Separar Canales
        self.blue_channel_button.clicked.connect(lambda: self.show_specific_channel('B'))
        self.green_channel_button.clicked.connect(lambda: self.show_specific_channel('G'))
        self.red_channel_button.clicked.connect(lambda: self.show_specific_channel('R'))

        # Espacios de Color
        
        # Only connect the color space buttons that are actually created as local variables in the color space group.

        # Operaciones Aritméticas
        self.add_button.clicked.connect(lambda: self.apply_arithmetic_op('add'))
        self.subtract_button.clicked.connect(lambda: self.apply_arithmetic_op('subtract'))
        self.multiply_button.clicked.connect(lambda: self.apply_arithmetic_op('multiply'))
        self.divide_button.clicked.connect(lambda: self.apply_arithmetic_op('divide'))
        self.add_scalar_slider.valueChanged.connect(self.update_add_scalar_from_slider)
        self.add_scalar_input.editingFinished.connect(lambda: self.update_add_scalar_from_input(self.add_scalar_input.text()))
        self.subtract_scalar_slider.valueChanged.connect(self.update_subtract_scalar_from_slider)
        self.subtract_scalar_input.editingFinished.connect(lambda: self.update_subtract_scalar_from_input(self.subtract_scalar_input.text()))
        self.multiply_scalar_input.editingFinished.connect(lambda: self.update_multiply_scalar_from_input(self.multiply_scalar_input.text()))
        self.divide_scalar_input.editingFinished.connect(lambda: self.update_divide_scalar_from_input(self.divide_scalar_input.text()))
        self.add_scalar_slider.valueChanged.connect(lambda: self.apply_arithmetic_op('add') if self.arithmetic_controls_group.isVisible() and self._last_arithmetic_op == 'add' else None)
        self.subtract_scalar_slider.valueChanged.connect(lambda: self.apply_arithmetic_op('subtract') if self.arithmetic_controls_group.isVisible() and self._last_arithmetic_op == 'subtract' else None)

        # Operaciones Lógicas
        self.and_button.clicked.connect(lambda: self.apply_logical_operation('AND'))
        self.or_button.clicked.connect(lambda: self.apply_logical_operation('OR'))
        self.xor_button.clicked.connect(lambda: self.apply_logical_operation('XOR'))
        self.not_button.clicked.connect(lambda: self.apply_logical_operation('NOT'))

        # Controles de Ruido
        self.radio_salt_pepper.toggled.connect(self.on_noise_type_changed)
        self.radio_gaussian.toggled.connect(self.on_noise_type_changed)
        self.sp_amount_slider.valueChanged.connect(self.update_sp_amount_label)
        self.sp_ratio_slider.valueChanged.connect(self.update_sp_ratio_label)
        self.gauss_stddev_slider.valueChanged.connect(self.update_gauss_stddev_label)
        self.apply_noise_button.clicked.connect(self.apply_noise)
        self.on_noise_type_changed() # Establecer estado inicial

        # Controles de Histograma - Conectar botones dentro del grupo para activar la ventana avanzada
        self.hist_original_button.clicked.connect(lambda: self.generate_and_display_histogram('original1'))
        self.hist_processed_button.clicked.connect(lambda: self.generate_and_display_histogram('processed'))

        # --- NEW: Connect the new Connected Components button ---
        self.connected_components_button.clicked.connect(self.show_connected_components_controls)
        # --- FIN NEW ---

        # --- NEW: Connect the Connected Components display buttons ---
        self.btn_bin.clicked.connect(self.show_binary_window)
        self.btn_4.clicked.connect(self.show_vec4_window)
        self.btn_8.clicked.connect(self.show_vec8_window)
        self.btn_contours.clicked.connect(self.show_contours_window)
        # --- FIN NEW ---

        # Connect Canny button and sliders
        self.btn_canny.clicked.connect(self.toggle_canny_controls_visibility)
        self.canny_low_thresh_slider.valueChanged.connect(self.update_canny_low_thresh)
        self.canny_high_thresh_slider.valueChanged.connect(self.update_canny_high_thresh)
        
        # Set initial values for Canny sliders
        self.canny_low_thresh_slider.setValue(self.canny_low_threshold)
        self.canny_high_thresh_slider.setValue(self.canny_high_threshold)
        
        # Update labels with initial values
        self.canny_low_thresh_label.setText(f"Umbral Bajo: {self.canny_low_threshold}")
        self.canny_high_thresh_label.setText(f"Umbral Alto: {self.canny_high_threshold}")
        
        # Initially hide Canny controls
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)

        # Conectar el botón Canny para asegurar que el grupo de filtros esté visible
        self.btn_canny.clicked.disconnect()  # Desconectar conexiones anteriores
        self.btn_canny.clicked.connect(lambda: (self.show_edge_filters_group(), self.toggle_canny_controls_visibility()))


    # --- Helper para habilitar/deshabilitar botones de operación ---
    def set_operation_buttons_enabled(self, enabled):
        """ Habilita o deshabilita los botones de operación principal y sliders de brillo/contraste."""
        # self.threshold_button.setEnabled(enabled and self.active_image is not None) # Original line
        self.threshold_button.setEnabled(True) # Always enabled as requested

        # logical_button depends on image presence for binary ops, but NOT is always possible if button is enabled
        # self.logical_button.setEnabled(enabled and self.active_image is not None) # Original line
        self.logical_button.setEnabled(True) # Always enabled as requested

        # other_buttons_enabled = enabled and self.active_image is not None # Original line
        # self.gray_button.setEnabled(other_buttons_enabled) # Original line
        self.gray_button.setEnabled(True) # Always enabled as requested
        # self.split_channels_button.setEnabled(other_buttons_enabled) # Original line
        self.split_channels_button.setEnabled(True) # Always enabled as requested
        # self.color_spaces_button.setEnabled(other_buttons_enabled) # Original line
         # Always enabled as requested
        # self.arithmetic_button.setEnabled(other_buttons_enabled) # Original line
        self.arithmetic_button.setEnabled(True) # Always enabled as requested
        # self.noise_button.setEnabled(other_buttons_enabled) # Original line
        self.noise_button.setEnabled(True) # Always enabled as requested
        # self.filters_button.setEnabled(other_buttons_enabled) # Original line
        self.filters_button.setEnabled(True) # Always enabled as requested

        # Botón de Histograma se habilita si Imagen 1 o processed_image está disponible # Original comment
        # self.histogram_button.setEnabled(enabled and (self.image is not None or self.processed_image is not None)) # Original line
        self.histogram_button.setEnabled(True) # Always enabled as requested

        # Enable the new Connected Components button always as requested
        self.connected_components_button.setEnabled(True) # Always enabled as requested

        # The buttons *inside* the logical group still depend on image availability, as requested previously
        # These are handled within show_logical_controls when the group is shown.
        if self.logical_button.isEnabled(): # This check is now effectively always True
            can_do_binary_logical = self.image is not None and self.image2 is not None
            self.and_button.setEnabled(can_do_binary_logical)
            self.or_button.setEnabled(can_do_binary_logical)
            self.xor_button.setEnabled(can_do_binary_logical)
            self.not_button.setEnabled(True) # NOT always enabled if the main group button is enabled

        # *** Habilitar/Deshabilitar Sliders de Brillo y Contraste ***
        # These sliders should be enabled if there is *any* image loaded (image or image2)
        # because operan en la imagen actualmente mostrada (processed_image o active_image si processed_image es None)
        brightness_contrast_enabled = self.image is not None or self.image2 is not None
        self.brightness_slider.setEnabled(brightness_contrast_enabled)
        self.contrast_slider.setEnabled(brightness_contrast_enabled)

        # Re-evaluate button state to enable/disable edge buttons
        self.btn_canny.setEnabled(enabled and self.active_image is not None)
        self.btn_horizontal.setEnabled(enabled and self.active_image is not None)
        self.btn_vertical.setEnabled(enabled and self.active_image is not None)
        self.btn_sobel.setEnabled(enabled and self.active_image is not None)
        self.btn_prewitt.setEnabled(enabled and self.active_image is not None)
        self.btn_laplacian.setEnabled(enabled and self.active_image is not None)
        self.btn_scharr.setEnabled(enabled and self.active_image is not None)
        # Enable/disable new edge filter buttons
        self.btn_robinson.setEnabled(enabled and self.active_image is not None)
        self.btn_kirsch.setEnabled(enabled and self.active_image is not None)
        self.btn_roberts.setEnabled(enabled and self.active_image is not None)
        # Los sliders de Canny solo se habilitan si están visibles Y los botones de borde están habilitados
        self.canny_low_thresh_slider.setEnabled(self.canny_low_thresh_slider.isVisible() and enabled)
        self.canny_high_thresh_slider.setEnabled(self.canny_high_thresh_slider.isVisible() and enabled)


    # --- Aplicar Estilos QSS ---
    def apply_styles(self):
        # Estilo general para la aplicación
        self.setStyleSheet("""
            QWidget {
                background-color: #FF0000;
            }
            QLabel {
                color: white;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                padding: 10px;
                color: white;
                background-color: rgba(0, 0, 0, 0.2);
                font-weight: bold;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background-color: rgba(0, 0, 0, 0.4);
                border: 2px solid rgba(255, 255, 255, 0.7);
            }
            QGroupBox {
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                margin-top: 1ex;
                padding: 15px;
                color: white;
                background-color: rgba(0, 0, 0, 0.2);
                font-family: 'Segoe UI', Arial;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 8px;
                color: white;
                font-weight: bold;
                font-family: 'Segoe UI', Arial;
            }
            QSplitter::handle {
                background-color: rgba(255, 255, 255, 0.3);
                width: 3px;
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: rgba(255, 255, 255, 0.5);
            }
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 0.3);
                height: 8px;
                background: rgba(0, 0, 0, 0.2);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 2px solid rgba(255, 255, 255, 0.3);
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1976D2;
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QLineEdit {
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                padding: 8px;
                background: rgba(0, 0, 0, 0.2);
                color: white;
                font-family: 'Segoe UI', Arial;
            }
            QLineEdit:focus {
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QComboBox {
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                padding: 8px;
                background: rgba(0, 0, 0, 0.2);
                color: white;
                font-family: 'Segoe UI', Arial;
            }
            QComboBox:hover {
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 0.1);
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.5);
            }
            QScrollBar:horizontal {
                border: none;
                background: rgba(255, 255, 255, 0.1);
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(255, 255, 255, 0.3);
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(255, 255, 255, 0.5);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

    # --- Helper para convertir imagen OpenCV a QPixmap (Mantenemos la versión robusta) ---

    # --- Helper para convertir imagen OpenCV a QPixmap (Mantenemos la versión robusta) ---
    def cv_to_qpixmap(self, cv_image):
        if cv_image is None or cv_image.size == 0:
            return QPixmap() # Return empty pixmap for None or empty image

        # Aplica brillo y contraste antes de convertir
        img_for_display = cv_image.copy()
        if img_for_display.dtype != np.float32:
            img_for_display = img_for_display.astype(np.float32)
        alpha = getattr(self, '_contrast_factor', 1.0)
        beta = getattr(self, '_brightness_value', 0)
        img_for_display = cv2.addWeighted(img_for_display, alpha, img_for_display, 0, beta)
        img_for_display = np.clip(img_for_display, 0, 255).astype(np.uint8)

        # Asegurarse de que la imagen es contigua en memoria
        img_for_display = np.ascontiguousarray(img_for_display)
        h, w = img_for_display.shape[:2]

        # Convert based on image type and channels
        if img_for_display.ndim == 2: # Gray scale image
            bytes_per_line = w
            q_image = QImage(img_for_display.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif img_for_display.ndim == 3 and img_for_display.shape[2] == 3: # Color BGR (3 channels)
            bytes_per_line = 3 * w
            rgb_image = cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        elif img_for_display.ndim == 3 and img_for_display.shape[2] == 4: # Color BGRA (with alpha, 4 channels)
            bytes_per_line = 4 * w
            rgba_image = cv2.cvtColor(img_for_display, cv2.COLOR_BGRA2RGBA)
            q_image = QImage(rgba_image.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        else:
            print(f"Formato de imagen no soportado para conversión a QPixmap: {img_for_display.shape}, Dtype: {img_for_display.dtype}")
            return QPixmap() # Return empty pixmap for unsupported formats

        if q_image.isNull():
            print("Error: QImage es nula después de la conversión")
            return QPixmap()

        return QPixmap.fromImage(q_image)


    # --- Método para mostrar una imagen en la etiqueta procesada (Panel Central) ---
    # Modified to store the original processed image and apply display adjustments before showing
    def display_processed_image(self, img, info_text="Imagen Procesada"):
        # Store the image *before* brightness/contrast display adjustments
        self.processed_image = img.copy() if img is not None else None

        # Update the info label
        self.processed_info_label.setText(info_text)

        if img is None:
            self.processed_label.clear()
            self.processed_label.setText("✨ Visualización Principal ✨\n(Carga una imagen y aplica operaciones)")
            return

        # Convert to QPixmap and display
        pixmap = self.cv_to_qpixmap(img)
        if pixmap is not None and not pixmap.isNull():
            # Reset zoom and offset before setting new pixmap
            self.processed_label.zoom_factor = 1.0
            self.processed_label.offset = QPoint(0, 0)

            # Escalar la pixmap para que quepa en la etiqueta manteniendo el aspecto
            scaled_pixmap = pixmap.scaled(self.processed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_label.setPixmap(scaled_pixmap)
            self.processed_label.update()
        else:
            # If cv_to_qpixmap returned None or invalid pixmap, handle the error
            print("Error: cv_to_qpixmap returned None or invalid pixmap")
            self.processed_label.clear()
            self.processed_label.setText("Error al convertir imagen para visualización")
            self.processed_info_label.setText("❌ Error de Visualización")
            self.set_operation_buttons_enabled(False)


    # --- Redefinir resizeEvent para escalar la imagen principal ---
    def resizeEvent(self, event):
        # Cuando la ventana (o processed_label) es redimensionada, reescalar y reestablecer la pixmap
        # Now, resize event should redraw the *processed_image* applying current display adjustments
        if self.processed_image is not None: # Use processed_image as the base for redrawing
            # Re-display processed_image which will re-apply display adjustments and scale
            current_info_text = self.processed_info_label.text()
            # Remove the prefix "⚙️ " before passing to display_processed_image
            info_text = current_info_text.replace('⚙️ ', '')
            self.display_processed_image(self.processed_image.copy(), info_text) # Pass a copy to display
        elif self.active_image is not None: # If no processed_image, redraw active_image
             current_info_text = self.processed_info_label.text()
             info_text = current_info_text.replace('⚙️ ', '')
             self.display_processed_image(self.active_image.copy(), info_text) # Pass a copy to display
        else:
             # If no image is loaded, just let the default resize event handle it
             super().resizeEvent(event)


    # --- Método para ocultar todos los controles condicionales (en Panel Derecho) ---
    def hide_all_controls(self):
        """Oculta todos los grupos de control específicos y botones de categoría de filtro."""
        self.gray_controls_group.setVisible(False)
        self.thresh_controls_group.setVisible(False)
        self.channel_controls_group.setVisible(False)
        self.color_space_controls_group.setVisible(False)
        self.arithmetic_controls_group.setVisible(False)
        self.logical_controls_group.setVisible(False)
        self.noise_controls_group.setVisible(False)
        self.histogram_controls_group.setVisible(False) # Ocultar el grupo de controles de histograma
        # --- NEW: Hide Morphological Operations group ---
        if hasattr(self, 'morph_operations_group'):
            self.morph_operations_group.setVisible(False)
        # --- FIN NEW ---
        # Limpia imágenes del panel morfológico
        if hasattr(self, 'morph_original_label'):
            self.morph_original_label.clear()
            self.morph_original_label.setText("Imagen Original")
        if hasattr(self, 'morph_processed_label'):
            self.morph_processed_label.clear()
            self.morph_processed_label.setText("Imagen Procesada")

        # Ocultar botones de categoría de filtro y sus grupos
        self.linear_filters_button.setVisible(False)
        self.nonlinear_filters_button.setVisible(False)
        # Ocultar el botón de categoría de bordes
        self.edge_filters_button.setVisible(False)

        self._hide_filter_groups() # Ocultar también los grupos de filtro específicos

        # Resetear la última operación aritmética
        self._last_arithmetic_op = None

        # Do NOT hide the brightness/contrast group as it's meant to be always visible

        # --- NEW: Hide the Connected Components group as well ---
        self.connected_components_group.setVisible(False)
        # --- FIN NEW ---

        # --- NEW: Hide the Morphological Operations group as well ---
        if hasattr(self, 'morph_operations_group'):
            self.morph_operations_group.setVisible(False)
        # --- FIN NEW ---

    def _hide_filter_groups(self):
        """Oculta solo los QGroupBox específicos de filtro."""
        self.linear_filters_group.setVisible(False)
        self.nonlinear_filters_group.setVisible(False)
        # Ocultar el grupo de filtros de borde
        self.edge_filters_group.setVisible(False)
        # Asegurarse de que los sliders de Canny también se oculten cuando se oculta el grupo de bordes
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        # No ocultamos el grupo de botones direccionales de Robinson aquí


    # --- Método llamado cuando se hace clic en la miniatura de la imagen original 1 ---
    def show_original_in_processed(self):
        if self.image is not None:
            # Establecer Imagen 1 como la imagen activa
            self.active_image = self.image.copy()
            
            # Reset brightness and contrast to default when switching to original image
            self.reset_brightness_contrast()
            
            # Mostrar la imagen en el panel central
            self.display_processed_image(self.active_image, "Imagen Original 1")
            
            # Ocultar controles al volver a la original
            self.hide_all_controls()
            
            # Cancelar worker si está activo
            self._cancel_threshold_worker()
            
            # Re-evaluar estado de botones
            self.set_operation_buttons_enabled(True)
            
            # La imagen original no tiene ruido (aplicado por nosotros)
            self._is_noisy = False
            
            # display_processed_image already calls histogram update if open
        else:
            QMessageBox.warning(self, "Advertencia", "Primero carga la Imagen 1.")


    # --- Método llamado cuando se hace clic en la miniatura de la imagen original 2 ---
    def show_image2_in_processed(self):
        if self.image2 is not None:
            self.active_image = self.image2 # Establecer Imagen 2 como la imagen activa
            # Reset brightness and contrast to default when switching to original image
            self.reset_brightness_contrast()
            self.display_processed_image(self.image2.copy(), "Imagen Original 2")
            self.hide_all_controls() # Ocultar controles al cambiar
            self._cancel_threshold_worker() # Cancelar worker si está activo
            self.set_operation_buttons_enabled(True) # Re-evaluar estado de botones
            self._is_noisy = False # La imagen original no tiene ruido
            # display_processed_image already calls histogram update if open

        else:
            QMessageBox.warning(self, "Advertencia", "Primero carga la Imagen 2.")


    # --- Métodos de Operaciones Morfológicas ---
    def display_morph_images(self):
        """Muestra la imagen activa en el panel morfológico y aplica la operación seleccionada."""
        if self.active_image is not None:
            orig_pixmap = self.cv_to_qpixmap(self.active_image)
            self.morph_original_label.setPixmap(orig_pixmap.scaled(self.morph_original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.apply_morphology_panel()
        else:
            self.morph_original_label.clear()
            self.morph_original_label.setText("Imagen Original")
            self.morph_processed_label.clear()
            self.morph_processed_label.setText("Imagen Procesada")
        self._last_morph_result = None

    def set_main_view_to_morph_result(self, event):
        """Al hacer clic en la imagen procesada del panel morfológico, se muestra en el panel principal."""
        if hasattr(self, '_last_morph_result') and self._last_morph_result is not None:
            # Mostramos la imagen morfológica en el panel central
            self.display_processed_image(self._last_morph_result, "Imagen Morfológica Procesada")

    def apply_morphology_panel(self):
        """Aplica la operación morfológica seleccionada en el panel sobre la imagen activa."""
        if self.active_image is None:
            self.morph_processed_label.clear()
            self.morph_processed_label.setText("Imagen Procesada")
            self._last_morph_result = None
            return
        img = self.active_image.copy()
        k_size = self.morph_kernel_slider.value()
        if k_size % 2 == 0:
            k_size += 1
        kernel = np.ones((k_size, k_size), np.uint8)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        op_name = ""
        for name, rb in self.morph_op_buttons.items():
            if rb.isChecked():
                op_name = name
                break
        if op_name == "Dilatación":
            processed = cv2.dilate(gray_image, kernel, iterations=1)
        elif op_name == "Erosión":
            processed = cv2.erode(gray_image, kernel, iterations=1)
        elif op_name == "Apertura":
            processed = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        elif op_name == "Cierre":
            processed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        elif op_name == "Gradiente":
            processed = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        else:
            processed = gray_image
        self._last_morph_result = processed
        pixmap = self.cv_to_qpixmap(processed)
        self.morph_processed_label.setPixmap(pixmap.scaled(self.morph_processed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_kernel_label_and_apply_panel(self, value=None):
        """Actualiza la etiqueta del tamaño del kernel y aplica la morfología en el panel."""
        size = self.morph_kernel_slider.value()
        if size % 2 == 0:
            size += 1
        self.morph_kernel_label.setText(f"Tamaño: {size}x{size}")
        self.apply_morphology_panel()

    # --- Métodos de Carga de Imagen ---
    def toggle_morph_operations_panel(self):
        """
        Muestra u oculta el panel de operaciones morfológicas.
        """
        if not hasattr(self, 'morph_operations_group'):
            return
        # Oculta todos los demás controles primero
        self.hide_all_controls()
        # Alterna la visibilidad del grupo morfológico
        is_visible = self.morph_operations_group.isVisible()
        self.morph_operations_group.setVisible(not is_visible)
        # Si se muestra el panel y hay imagen activa, mostrarla
        if self.morph_operations_group.isVisible() and self.active_image is not None:
            self.display_morph_images()

    def load_image1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen 1", "", "Imagen (*.png *.jpg *.jpeg *.bmp *.tiff);;Todos los archivos (*)")
        if path:
            try:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # Try to load with alpha channel if exists
                if img is None:
                    QMessageBox.critical(self, "Error", f"No se pudo cargar la Imagen 1 desde '{path}'.")
                    self.image = None
                    self.image1_thumbnail_label.clear() # Clear thumbnail on error
                    self.image1_thumbnail_label.setText("Error al Cargar")
                    self.image1_loaded_label.setText("Imagen 1: No Cargada")
                    return

                # Store the original image
                self.image = img.copy()
                self.image1_loaded_label.setText(f"Imagen 1: {path.split('/')[-1]}")

                # Create a thumbnail for display
                thumb_display = img.copy()
                # Ensure the image is BGR 3 channels for consistent thumbnail processing
                if thumb_display.ndim == 2: # Grayscale
                    thumb_display = cv2.cvtColor(thumb_display, cv2.COLOR_GRAY2BGR)
                elif thumb_display.ndim == 3 and thumb_display.shape[2] == 4: # BGRA
                    thumb_display = cv2.cvtColor(thumb_display, cv2.COLOR_BGRA2BGR)
                
                # Resize for thumbnail
                thumb_resized = cv2.resize(thumb_display, (128, 128), interpolation=cv2.INTER_AREA)
                
                # Convert to QPixmap and display
                thumb_pixmap = self.cv_to_qpixmap_thumbnail(thumb_resized)
                if not thumb_pixmap.isNull():
                    self.image1_thumbnail_label.setPixmap(thumb_pixmap)
                    self.image1_thumbnail_label.setText("")
                else:
                    self.image1_thumbnail_label.setText("No Preview")
                
                # Update the active image
                self.active_image = img.copy()
                
                # Show the image in the processed area
                self.show_original_in_processed()
                
                # Enable operations
                self.set_operation_buttons_enabled(True)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error al cargar la imagen: {str(e)}")
                return
        
        # Close any OpenCV windows that might have been opened
        cv2.destroyAllWindows()

    def load_image2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen 2", "", "Imagen (*.png *.jpg *.jpeg *.bmp *.tiff);;Todos los archivos (*)")
        if path:
            try:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # Try to load with alpha channel if exists
                if img is None:
                    QMessageBox.critical(self, "Error", f"No se pudo cargar la Imagen 2 desde '{path}'.")
                    self.image2 = None
                    self.image2_thumbnail_label.clear() # Clear thumbnail on error
                    self.image2_thumbnail_label.setText("Error al Cargar")
                    self.image2_loaded_label.setText("Imagen 2: No Cargada")
                    return

                # Store the original image
                self.image2 = img.copy()
                self.image2_loaded_label.setText(f"Imagen 2: {path.split('/')[-1]}")

                # Create a thumbnail for display
                thumb_display = img.copy()
                # Ensure the image is BGR 3 channels for consistent thumbnail processing
                if thumb_display.ndim == 2: # Grayscale
                    thumb_display = cv2.cvtColor(thumb_display, cv2.COLOR_GRAY2BGR)
                elif thumb_display.ndim == 3 and thumb_display.shape[2] == 4: # BGRA
                    thumb_display = cv2.cvtColor(thumb_display, cv2.COLOR_BGRA2BGR)
                
                # Resize for thumbnail
                thumb_resized = cv2.resize(thumb_display, (128, 128), interpolation=cv2.INTER_AREA)
                
                # Convert to QPixmap and display
                thumb_pixmap = self.cv_to_qpixmap_thumbnail(thumb_resized)
                if not thumb_pixmap.isNull():
                    self.image2_thumbnail_label.setPixmap(thumb_pixmap)
                    self.image2_thumbnail_label.setText("")
                else:
                    self.image2_thumbnail_label.setText("No Preview")
                
                # Update the active image
                self.active_image = img.copy()
                
                # Show the image in the processed area
                self.show_image2_in_processed()
                
                # Enable operations
                self.set_operation_buttons_enabled(True)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error al cargar la imagen: {str(e)}")
                return
        
        # Close any OpenCV windows that might have been opened
        cv2.destroyAllWindows()

    def cv_to_qpixmap_thumbnail(self, cv_image):
         if cv_image is None or cv_image.size == 0:
             return QPixmap() # Return empty pixmap for None or empty image

         h, w = cv_image.shape[:2]
         # Convert based on image type and channels
         if cv_image.ndim == 2: # Gray scale image
             bytes_per_line = w
             q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
         if cv_image is None:
             return QPixmap()

         cv_image = np.ascontiguousarray(cv_image)

         if cv_image.ndim == 2: # Grayscale (1 channel)
             height, width = cv_image.shape
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
             print(f"Formato de imagen no soportado directamente para miniatura: {cv_image.shape}, Dtype: {cv_image.dtype}")
             return QPixmap() # Return empty pixmap for unsupported formats

         if q_image.isNull():
              print("QImage es nula después de la conversión para miniatura")
              return QPixmap()

         return QPixmap.fromImage(q_image)

    # *** End of new helper ***

    # --- Métodos de Procesamiento (Operan principalmente en self.active_image.copy() o self.processed_image.copy() según _is_noisy) ---

    # Mostrar Controles de Escala de Grises y Aplicar Escala de Grises (inicial)
    def show_gray_controls(self):
        # Esta operación debe trabajar en la imagen activa (revierte el ruido)
        if self.active_image is not None:
            self.hide_all_controls() # Ocultar otros grupos de control
            self.gray_controls_group.setVisible(True) # Mostrar controles de escala de grises
            # Aplicar escala de grises con el estado actual del checkbox a la imagen activa
            self.apply_gray()
            self._cancel_threshold_worker() # Cancelar cualquier worker de umbralización si está activo
        else:
            QMessageBox.warning(self, "Advertencia", "Carga una imagen de trabajo (Imagen 1 o Imagen 2) primero para aplicar Escala de Grises.")


    # Aplicar Escala de Grises (llamado por cambio de estado de checkbox o inicialmente por show_gray_controls)
    def apply_gray(self):
        # Operar en la imagen actualmente activa (revierte el ruido)
        if self.active_image is not None:
            img_to_process = self.active_image.copy() # Trabajar en una copia de la imagen activa

            gray = None
            info_text = "Escala de Grises"

            if img_to_process.ndim == 3:
                # Convertir a gris, manejando BGRA if needed
                if img_to_process.shape[2] == 4:
                     gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGRA2GRAY)
                elif img_to_process.shape[2] == 3:
                    gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
                else: # Unexpected color format
                     QMessageBox.warning(self, "Advertencia", f"Formato de imagen con {img_to_process.shape[2]} canales no válido para escala de grises.")
                     # Display the original active image with an info message
                     self.display_processed_image(self.active_image.copy(), f"Formato no compatible con Escala de Grises")
                     return
            elif img_to_process.ndim == 2: # If already grayscale, use it directly
                 gray = img_to_process.copy()
            else: # Unsupported format
                 QMessageBox.warning(self, "Advertencia", "Formato de imagen no válido para convertir a escala de grises.")
                 self.display_processed_image(self.active_image.copy(), f"Formato no compatible con Escala de Grises")
                 return

            # Ensure the resulting image is uint8 if it wasn't
            if gray.dtype != np.uint8:
                 gray = np.clip(gray, 0, 255).astype(np.uint8)


            if self.gray_inv_checkbox.isChecked():
                gray = cv2.bitwise_not(gray) # Invert if checkbox is checked
                info_text = "Escala de Grises Invertida"

            # Show the result in the central panel (this calls display_processed_image)
            self.display_processed_image(gray, f"{info_text}")
            self._cancel_threshold_worker() # Cancel any worker de umbralización if está activo
            self._is_noisy = False # After grayscale, the image is not "noisy" in the sense of the flag


        else:
             # This case should not happen if show_gray_controls checks, but for safety
             QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para aplicar escala de grises.")


    # Mostrar Controles de Umbral y Iniciar Worker
    def show_threshold_controls(self):
         # Umbralización now applies to the active image (reverts noise)
         if self.active_image is not None:
             self.hide_all_controls() # Hide other control groups
             self.thresh_controls_group.setVisible(True) # Show threshold controls
             # Start the threshold worker using self.active_image.copy()
             self._start_threshold_worker()
         else:
             # Warning message if no active image
             QMessageBox.warning(self, "Advertencia", "Carga una imagen de trabajo (Imagen 1 o Imagen 2) primero para umbralizar.")


    def update_threshold_label(self, value):
        self.thresh_slider_label.setText(f"Umbral: {value}")


    # --- Métodos de Umbralización con Worker (Ahora operan en self.active_image y resetean _is_noisy en el resultado) ---
    # Modified to use self.active_image
    def _start_threshold_worker(self):
        # Start the threshold worker
        if self.active_image is None:
            # If no active image, cannot threshold
            print("Advertencia: Intento de iniciar worker de umbralización sin imagen de trabajo.")
            return

        # Cancel any previous threshold worker if running
        self._cancel_threshold_worker()

        # Get current slider and checkbox values
        threshold_value = self.thresh_slider.value()
        invert = self.thresh_inv_checkbox.isChecked()

        # Create and connect the worker
        # The worker receives the active_image (original base image) for processing
        self._threshold_worker = ThresholdWorker(self.active_image.copy(), threshold_value, invert)
        self._threshold_worker.result_ready.connect(self._display_thresholded_result)
        self._threshold_worker.error_occurred.connect(self._handle_threshold_error)
        self._threshold_worker.finished.connect(self._threshold_worker_finished) # Connect finished signal
        # Start the worker
        self._threshold_worker.start()


    def _display_thresholded_result(self, thresholded_img, info_text):
        # Slot to receive result from worker and display it
        # display_processed_image is called, which will apply display adjustments
        self.display_processed_image(thresholded_img, info_text)
        self._is_noisy = False # After thresholding, the image is not noisy
        # The worker is cleaned up in _threshold_worker_finished


    def _handle_threshold_error(self, error_message):
        # Slot to handle worker errors
        QMessageBox.critical(self, "Error de Umbralización", error_message)
        # Revert to the active image if possible, otherwise show empty
        if self.active_image is not None:
            self.display_processed_image(self.active_image.copy(), "Error Umbralización")
        else:
            self.display_processed_image(None, "Error Umbralización")
        self._is_noisy = False # Reset noisy state on error
        # The worker is cleaned up in _threshold_worker_finished


    def _cancel_threshold_worker(self):
        # Method to cancel a threshold worker if active
        if self._threshold_worker is not None and self._threshold_worker.isRunning():
            print("Cancelando worker de umbralización...")
            self._threshold_worker.cancel()
            # We do not use wait() here on the main thread to avoid freezing the UI.
            # The actual cleanup (setting _threshold_worker to None) happens in _threshold_worker_finished.

    def _threshold_worker_finished(self):
         # This slot is called when the threshold worker finishes (normally or by cancellation)
         print("Worker de umbralización finalizado.")
         self._threshold_worker = None # Clear the reference

    # --- End Thresholding Methods with Worker ---


    # Mostrar Controles de Separar Canales
    def show_channel_controls(self, ):
        # This operation should work on the active image (reverts noise)
        if self.active_image is not None:
            # Check if the active image is color (3 or 4 channels)
            if not (self.active_image.ndim == 3 and (self.active_image.shape[2] == 3 or self.active_image.shape[2] == 4)):
                QMessageBox.warning(self, "Advertencia", "La imagen de trabajo seleccionada no es a color (BGR/BGRA) para separar canales.")
                return

            self.hide_all_controls() # Hide other control groups
            self.channel_controls_group.setVisible(True) # Show channel controls
            # Show the current active image as a placeholder
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(self.active_image.copy(), f"Selecciona un Canal (R, G, B)")
            self._cancel_threshold_worker()
        else:
             # Warning message if no active image
             QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo seleccionada para procesar.")


    # Mostrar un canal específico
    def show_specific_channel(self, channel_name):
        # Operate on the currently active image (reverts noise)
        if self.active_image is not None and self.active_image.ndim == 3 and (self.active_image.shape[2] == 3 or self.active_image.shape[2] == 4):
            img_to_process = self.active_image.copy()

            # Ensure we have 3 channels for split/merge if the active image has 4
            if img_to_process.shape[2] == 4:
                 img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGRA2BGR)

            B, G, R = cv2.split(img_to_process)
            zeros = np.zeros_like(B) # Create a zeros array of the same size and type as the channel

            channel_img = None
            info_text = ""

            # Merge the selected channel with zero matrices to create a BGR image
            if channel_name == 'B':
                channel_img = cv2.merge([B, zeros, zeros]) # Only Blue channel
                info_text = "Canal Azul (B)"
            elif channel_name == 'G':
                channel_img = cv2.merge([zeros, G, zeros]) # Only Green channel
                info_text = "Canal Verde (G)"
            elif channel_name == 'R':
                channel_img = cv2.merge([zeros, zeros, R]) # Only Red channel
                info_text = "Canal Rojo (R)"

            if channel_img is not None:
                # display_processed_image is called, which will apply display adjustments
                self.display_processed_image(channel_img, f"{info_text}")
                self._cancel_threshold_worker()
                self._is_noisy = False # After separating channels, the image is not noisy


            else:
                 print(f"Error: Unknown channel '{channel_name}'")


        else:
             QMessageBox.warning(self, "Advertencia", "La imagen de trabajo no es a color.")


    # Mostrar Controles de Espacios de Color
    def show_color_space_controls(self):
        if self.active_image is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo seleccionada para procesar.")
            return

        self.hide_all_controls() # Hide other groups
        self.color_space_controls_group.setVisible(True) # Show color space controls
        # Show the current active image as a placeholder
        # display_processed_image is called, which will apply display adjustments
        self.display_processed_image(self.active_image.copy(), f"Selecciona un Espacio de Color")
        self._cancel_threshold_worker()


    # Mostrar un espacio de color específico
    def show_specific_color_space(self, space_name):
        """Muestra la imagen convertida al espacio de color especificado y crea los botones de canales."""
        if self.active_image is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo seleccionada para procesar.")
            return

        try:
            # Convert image to selected color space
            self.converted_image = ColorSpaceConverter.convert_image(self.active_image.copy(), space_name)
            self.current_color_space = space_name
            
            # Create channel buttons
            self.create_channel_buttons(space_name)
            
            # Show the converted image
            self.display_processed_image(self.converted_image, f"Espacio de Color: {space_name}")
            self._cancel_threshold_worker()
            self._is_noisy = False
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al convertir a {space_name}: {str(e)}")
            self.display_processed_image(None, f"Error en Espacio {space_name}")
            self._cancel_threshold_worker()
            self._is_noisy = False

    def create_channel_buttons(self, space_name):
        # Limpiar botones de canal existentes
        for i in reversed(range(self.channel_buttons_layout.count())): 
            widget_to_remove = self.channel_buttons_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()

        channel_names = ColorSpaceConverter.get_channel_names(space_name)
        if not channel_names:
            self.channel_buttons_widget.hide()
            return

        # Crear nuevos botones de canal
        for i, name in enumerate(channel_names):
            btn = QPushButton(f"Canal: {name}")
            btn.clicked.connect(lambda checked, index=i: self.show_channel(index))
            self.channel_buttons_layout.addWidget(btn)
        
        self.channel_buttons_widget.show()

    def show_channel(self, channel_index):
        """Muestra el canal específico del espacio de color actual."""
        if self.converted_image is None or self.current_color_space is None:
            return
            
        try:
            # Extract and normalize the channel
            channel = self.converted_image[:, :, channel_index]
            normalized_channel = ColorSpaceConverter.normalize_channel(
                channel, 
                self.current_color_space, 
                channel_index
            )
            
            # Create a 3-channel image for display
            display_image = cv2.merge([normalized_channel] * 3)
            
            # Show the channel
            channel_name = ColorSpaceConverter.get_channel_names(self.current_color_space)[channel_index]
            self.display_processed_image(
                display_image, 
                f"Canal {channel_name} de {self.current_color_space}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Error", 
                f"Error al mostrar el canal: {str(e)}"
            )

    # Mostrar Controles Aritméticos
    def show_arithmetic_controls(self):
        # This operation should work on the active image (reverts noise)
        self.hide_all_controls()
        self.arithmetic_controls_group.show()
        # Initialize the values shown in the input fields and labels
        self.update_arithmetic_ui_values()
        # Reset the last arithmetic operation
        self._last_arithmetic_op = None

        # Show the current processed image if available, otherwise use active_image
        img_to_show = self.processed_image.copy() if isinstance(self.processed_image, np.ndarray) else (self.active_image.copy() if isinstance(self.active_image, np.ndarray) else None)
        # display_processed_image is called, which will apply display adjustments
        self.display_processed_image(img_to_show, f"Selecciona una Operación Aritmética (Escalar)")
        self._cancel_threshold_worker()

    def update_arithmetic_ui_values(self):
        """Updates the arithmetic inputs and sliders with stored values."""
        self.add_scalar_input.setText(str(self._add_scalar))
        self.add_scalar_label.setText(f"Valor: {self._add_scalar}")
        # Temporarily block signals to prevent the slider from firing apply_arithmetic_op initially
        self.add_scalar_slider.blockSignals(True)
        self.add_scalar_slider.setValue(int(np.clip(self._add_scalar, self.add_scalar_slider.minimum(), self.add_scalar_slider.maximum())))
        self.add_scalar_slider.blockSignals(False)

        self.subtract_scalar_input.setText(str(self._subtract_scalar))
        self.subtract_scalar_label.setText(f"Valor: {self._subtract_scalar}")
        self.subtract_scalar_slider.blockSignals(True)
        self.subtract_scalar_slider.setValue(int(np.clip(self._subtract_scalar, self.subtract_scalar_slider.minimum(), self.subtract_scalar_slider.maximum())))
        self.subtract_scalar_slider.blockSignals(False)

        self.multiply_scalar_input.setText(str(self._multiply_scalar))
        self.multiply_scalar_label.setText(f"Valor: {self._multiply_scalar}")

        self.divide_scalar_input.setText(str(self._divide_scalar))
        self.divide_scalar_label.setText(f"Valor: {self._divide_scalar}")


    # --- Métodos para actualizar valores escalares individuales ---
    def update_add_scalar_from_slider(self, value):
        self._add_scalar = value
        self.add_scalar_input.setText(str(value))
        self.add_scalar_label.setText(f"Valor: {value}")

    def update_add_scalar_from_input(self, text):
        try:
            value = float(text)
            self._add_scalar = value
            int_value = int(np.clip(value, self.add_scalar_slider.minimum(), self.add_scalar_slider.maximum()))
            self.add_scalar_slider.blockSignals(True)
            self.add_scalar_slider.setValue(int_value)
            self.add_scalar_slider.blockSignals(False)
            self.add_scalar_label.setText(f"Valor: {value}")

            if self.arithmetic_controls_group.isVisible() and self._last_arithmetic_op == 'add':
                 self.apply_arithmetic_op('add')

        except ValueError:
            self.add_scalar_label.setText("Valor: Inválido")

    def update_subtract_scalar_from_slider(self, value):
        self._subtract_scalar = value
        self.subtract_scalar_input.setText(str(value))
        self.subtract_scalar_label.setText(f"Valor: {value}")

    def update_subtract_scalar_from_input(self, text):
        try:
            value = float(text)
            self._subtract_scalar = value
            int_value = int(np.clip(value, self.subtract_scalar_slider.minimum(), self.subtract_scalar_slider.maximum()))
            self.subtract_scalar_slider.blockSignals(True)
            self.subtract_scalar_slider.setValue(int_value)
            self.subtract_scalar_slider.blockSignals(False)
            self.subtract_scalar_label.setText(f"Valor: {value}")

            if self.arithmetic_controls_group.isVisible() and self._last_arithmetic_op == 'subtract':
                 self.apply_arithmetic_op('subtract')

        except ValueError:
            self.subtract_scalar_label.setText("Valor: Inválido")

    def update_multiply_scalar_from_input(self, text):
        try:
            value = float(text)
            self._multiply_scalar = value
            self.multiply_scalar_label.setText(f"Valor: {value}")
            if self.arithmetic_controls_group.isVisible() and self._last_arithmetic_op == 'multiply':
                 self.apply_arithmetic_op('multiply')

        except ValueError:
            self.multiply_scalar_label.setText("Valor: Inválido")

    def update_divide_scalar_from_input(self, text):
        try:
            value = float(text)
            if abs(value) < 1e-6: # Use tolerance
                 self.divide_scalar_label.setText("Valor: Cero! (División por cero)")
                 return
            else:
                 self._divide_scalar = value
                 self.divide_scalar_label.setText(f"Valor: {value}")
                 if self.arithmetic_controls_group.isVisible() and self._last_arithmetic_op == 'divide':
                      self.apply_arithmetic_op('divide')

        except ValueError:
            self.divide_scalar_label.setText("Valor: Inválido")


    # --- Métodos para aplicar operaciones aritméticas usando valores individuales ---
    def apply_arithmetic_op(self, op_type):
        # Operate on the currently active image (reverts noise)
        if self.active_image is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para aplicar {op_type}.")
            return

        try:
            # Store the operation type for reapplication
            self._last_arithmetic_op = op_type

            # Get the image to process
            img_to_process = self.active_image.copy()

            # Apply the selected operation
            result = None
            info_text = ""

            if op_type == 'add':
                scalar_value = self._add_scalar # Use the individually stored value (float)
                result = cv2.add(img_to_process, scalar_value)
                info_text = f"Suma (+{scalar_value})"

            elif op_type == 'subtract':
                scalar_value = self._subtract_scalar # Use the individually stored value (float)
                result = cv2.subtract(img_to_process, scalar_value)
                info_text = f"Resta (-{scalar_value})"

            elif op_type == 'multiply':
                scalar_value = self._multiply_scalar # Use the individually stored value (float)
                result = cv2.multiply(img_to_process, scalar_value)
                info_text = f"Multiplicación (×{scalar_value})"

            elif op_type == 'divide':
                scalar_value = self._divide_scalar # Use the individually stored value (float)
                if abs(scalar_value) < 1e-6: # Use tolerance
                     QMessageBox.warning(self, "Advertencia", "División por cero evitada.")
                     # Display the last processed image or active image
                     img_to_show_on_error = self.processed_image.copy() if self.processed_image is not None else (self.active_image.copy() if self.active_image is not None else None)
                     self.display_processed_image(img_to_show_on_error, "División por Cero Evitada")
                     self._last_arithmetic_op = None # Reset if there's an error or avoided operation
                     self._cancel_threshold_worker()
                     self._is_noisy = False # Reset noisy state on error
                     return # Exit if division by zero is detected
                else:
                    # Convert to float before dividing. cv2.divide with a float scalar and uint8 produces float.
                    result_float = cv2.divide(img_to_process.astype(np.float32), scalar_value)
                    result = np.clip(result_float, 0, 255).astype(np.uint8)
                    info_text = f"División (/{scalar_value})"

            if result is not None:
                # display_processed_image is called, which will apply display adjustments
                self.display_processed_image(result, f"{info_text}")

            self._cancel_threshold_worker() # Cancel any threshold worker if active
            self._is_noisy = False

        except cv2.error as e:
            QMessageBox.critical(self, f"Error en {op_type.capitalize()}", f"Ocurrió un error durante la operación {op_type}: {e}")
            self.display_processed_image(None, f"Error en {op_type.capitalize()}") # Show error in panel
            self._last_arithmetic_op = None # Reset if there's an error
            self._cancel_threshold_worker()
            self._is_noisy = False # Reset noisy state on error
        except Exception as e:
             QMessageBox.critical(self, f"Error General en {op_type.capitalize()}", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, f"Error en {op_type.capitalize()}") # Show error in panel
             self._last_arithmetic_op = None
             self._cancel_threshold_worker()
             self._is_noisy = False # Reset noisy state on error


    # --- Logical Operation Methods (Operate on self.image and self.image2, or active_image for NOT) ---
    # These operations explicitly use self.image and self.image2 for binary ops, active_image for NOT
    def show_logical_controls(self):
        # Logical operations require at least the active image (for NOT)
        # Binary operations require both original images
        if self.active_image is None:
            QMessageBox.warning(self, "Advertencia", "Carga una imagen de trabajo (Imagen 1 o Imagen 2) primero para operaciones lógicas.")
            return

        self.hide_all_controls() # Hide other groups
        self.logical_controls_group.setVisible(True) # Show logical controls

        # Update enabled state of buttons within the logical group
        can_do_binary_logical = self.image is not None and self.image2 is not None
        self.and_button.setEnabled(can_do_binary_logical)
        self.or_button.setEnabled(can_do_binary_logical)
        self.xor_button.setEnabled(can_do_binary_logical)
        self.not_button.setEnabled(True) # NOT only needs the active image, which exists here

        # Show the active image as a placeholder for logical operations
        # display_processed_image is called, which will apply display adjustments
        self.display_processed_image(self.active_image.copy(), "Selecciona Op. Lógica (Binaria en Im1/Im2, NOT en Activa)")
        self._cancel_threshold_worker() # Ensure the threshold worker is canceled if running
        # No change to _is_noisy here as showing controls doesn't apply an operation


    def apply_logical_operation(self, op_type):
        result = None
        info_text = ""

        try:
            if op_type == 'NOT':
                # Logical NOT operates on the active image
                if self.active_image is None:
                    QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo seleccionada para aplicar NOT.")
                    return

                img_to_process = self.active_image.copy()

                # Ensure the image is uint8 for bitwise_not
                if img_to_process.dtype != np.uint8:
                    QMessageBox.warning(self, "Advertencia", f"La operación NOT requiere imágenes de 8-bit (uint8). Imagen de trabajo es {img_to_process.dtype}. Intentando convertir...")
                    try:
                        img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
                        print("Imagen converted to uint8 for NOT operation.")
                    except Exception as e_conv:
                         QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para NOT: {e_conv}")
                         self.display_processed_image(None, f"Error al aplicar NOT")
                         self._cancel_threshold_worker()
                         self._is_noisy = False # Reset noisy state on error
                         return


                result = cv2.bitwise_not(img_to_process)
                info_text = "Op. Lógica: NOT (Imagen Activa)"

            elif op_type in ['AND', 'OR', 'XOR']:
                # Binary logical operations operate on Image 1 and Image 2
                if self.image is None or self.image2 is None:
                    QMessageBox.warning(self, "Advertencia", f"Carga Imagen 1 e Imagen 2 primero para la operación {op_type}.")
                    return

                img1 = self.image.copy() # Work on copies of the original loaded images
                img2 = self.image2.copy()

                # --- Handle Size Discrepancy ---
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]

                if (h1, w1) != (h2, w2):
                    reply = QMessageBox.question(self, "Discrepancia de Tamaño",
                                                 f"Imagen 2 ({w2}x{h2}) vs Imagen 1 ({w1}x{h1}).\n¿Redimensionar Imagen 2 para que coincida con Imagen 1?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if reply == QMessageBox.Yes:
                        try:
                            # Use INTER_AREA for reduction and INTER_LINEAR for enlargement (good practice)
                            interpolation = cv2.INTER_AREA if (w2*h2 > w1*h1) else cv2.INTER_LINEAR
                            img2 = cv2.resize(img2, (w1, h1), interpolation=interpolation)
                            print("Imagen 2 redimensionada.")
                        except Exception as e:
                             print(f"Error al redimensionar Imagen 2: {e}")
                             QMessageBox.critical(self, "Error Redimensionando", f"No se pudo redimensionar Imagen 2: {e}")
                             self.display_processed_image(None, "Error de Redimensionamiento")
                             self._cancel_threshold_worker()
                             self._is_noisy = False # Reset noisy state on error
                             return
                    else:
                        QMessageBox.information(self, "Operación Cancelada", "Operación lógica binaria cancelada.")
                        return

                # --- Handle Channel Discrepancy (Convert to BGR if necessary) ---
                # Bitwise logical operations usually require images of the same type and number of channels.
                # Attempt to convert both to BGR if they are not of the same type or have different channels.
                ndim1, ndim2 = img1.ndim, img2.ndim
                channels1 = img1.shape[2] if ndim1 == 3 else 1
                channels2 = img2.shape[2] if ndim2 == 3 else 1

                # Convert BGRA to BGR if applicable
                if ndim1 == 3 and channels1 == 4:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
                    channels1 = 3
                if ndim2 == 3 and channels2 == 4:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
                    channels2 = 3

                # If after handling BGRA, channels still don't match, try converting both to BGR (if possible)
                if channels1 != channels2 or img1.dtype != img2.dtype:
                     print(f"Advertencia: Canales o tipos de dato diferentes ({img1.shape}, {img1.dtype} vs {img2.shape}, {img2.dtype}). Intentando homogeneizar a BGR.")
                     try:
                        # Convert to BGR if grayscale
                        if img1.ndim == 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                        if img2.ndim == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

                        # If after BGR conversion they still don't have the same shape (they should if resizing worked)
                        if img1.shape != img2.shape:
                             raise ValueError("Las imágenes no tienen la misma forma después de intentar homogeneizar canales.")

                        # Ensure both are uint8 (required by bitwise_*)
                        if img1.dtype != np.uint8: img1 = np.clip(img1, 0, 255).astype(np.uint8)
                        if img2.dtype != np.uint8: img2 = np.clip(img2, 0, 255).astype(np.uint8)

                     except Exception as e:
                         QMessageBox.critical(self, "Error de Canales/Tipo", f"No se pudieron homogeneizar las imágenes para operación lógica: {e}")
                         self.display_processed_image(None, "Error de Canales/Tipo")
                         self._cancel_threshold_worker()
                         self._is_noisy = False # Reset noisy state on error
                         return


                # Apply the logical operation
                if op_type == 'AND':
                    result = cv2.bitwise_and(img1, img2)
                elif op_type == 'OR':
                    result = cv2.bitwise_or(img1, img2)
                elif op_type == 'XOR':
                    result = cv2.bitwise_xor(img1, img2)

                info_text = f"Op. Lógica: Im1 {op_type} Im2"

            else:
                 # Should not happen with current buttons
                 print(f"Error: Unknown logical operation type '{op_type}'")
                 return


            if result is not None:
                # display_processed_image is called, which will apply display adjustments
                self.display_processed_image(result, info_text) # Show result
                self._cancel_threshold_worker() # Ensure the threshold worker is canceled if running
                self._is_noisy = False # After logical operation, the image is not noisy


        except cv2.error as e:
            # Handle OpenCV errors during the operation
            QMessageBox.critical(self, "Error de OpenCV", f"Error applying logical operation: {e}")
            self.display_processed_image(None, f"Error en {info_text}") # Show error in the panel
            self._cancel_threshold_worker()
            self._is_noisy = False # Reset noisy state on error
        except Exception as e:
             QMessageBox.critical(self, "General Error", f"Ocurrió un error inesperado durante la operación lógica: {e}")
             self.display_processed_image(None, f"Error en {info_text}") # Show error in the panel
             self._cancel_threshold_worker()
             self._is_noisy = False # Reset noisy state on error


    # --- Noise Operations ---
    def show_noise_controls(self):
        """Shows the noise controls group."""
        # This operation should work on the currently processed image
        if self.active_image is not None: # Still need a loaded base image
            self.hide_all_controls() # Hide other control groups
            self.noise_controls_group.setVisible(True) # Show noise controls
            # Show the currently processed image as a placeholder for noise application
            # Use processed_image if available, otherwise use active_image
            img_to_show = self.processed_image.copy() if isinstance(self.processed_image, np.ndarray) else (self.active_image.copy() if isinstance(self.active_image, np.ndarray) else None)
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(img_to_show, f"Añadir Ruido")
            self._cancel_threshold_worker() # Cancel any threshold worker if active
            # Ensure visibility of noise parameter layouts is correct based on default radio button
            self.on_noise_type_changed()
        else:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo seleccionada para añadir ruido.")

    def on_noise_type_changed(self):
        # Show/hide parameter layouts based on selected radio button
        if self.radio_salt_pepper.isChecked():
            # Using layouts directly instead of groupboxes
            self.salt_pepper_params_layout.setEnabled(True)
            self.gaussian_params_layout.setEnabled(False)
            # We need to iterate through the widgets in the layout to set visibility
            for i in range(self.salt_pepper_params_layout.count()):
                widget = self.salt_pepper_params_layout.itemAt(i).widget()
                if widget: widget.setVisible(True) # Check if widget exists

            for i in range(self.gaussian_params_layout.count()):
                 widget = self.gaussian_params_layout.itemAt(i).widget()
                 if widget: widget.setVisible(False) # Check if widget exists


        elif self.radio_gaussian.isChecked():
            self.salt_pepper_params_layout.setEnabled(False)
            self.gaussian_params_layout.setEnabled(True)
            for i in range(self.salt_pepper_params_layout.count()):
                 widget = self.salt_pepper_params_layout.itemAt(i).widget()
                 if widget: widget.setVisible(False) # Check if widget exists
            for i in range(self.gaussian_params_layout.count()):
                 widget = self.gaussian_params_layout.itemAt(i).widget()
                 if widget: widget.setVisible(True) # Check if widget exists


    def apply_noise(self):
        """Applies the selected noise to the currently processed image."""
        # Operate on the currently processed image
        # Use the processed_image if available, otherwise use active_image
        img_to_process = self.processed_image.copy() if isinstance(self.processed_image, np.ndarray) else (self.active_image.copy() if isinstance(self.active_image, np.ndarray) else None)

        if img_to_process is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen para añadir ruido.")
            return

        # Ensure image is uint8 before applying noise functions
        if img_to_process.dtype != np.uint8:
            QMessageBox.warning(self, "Advertencia", f"El ruido requiere imágenes de 8-bit (uint8). Imagen de trabajo es {img_to_process.dtype}. Intentando convertir...")
            try:
                img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
                print("Imagen converted to uint8 for noise application.")
            except Exception as e_conv:
                 QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para ruido: {e_conv}")
                 self.display_processed_image(None, f"Error al aplicar Ruido")
                 self._cancel_threshold_worker()
                 self._is_noisy = False # Reset noisy state on error
                 return


        applied_noise_type = ""
        noisy_image = None # Use a local variable for the result

        try:
            if self.radio_salt_pepper.isChecked():
                amount = self.sp_amount_slider.value() / 100.0 # Convert 0-100 to 0-1
                ratio = self.sp_ratio_slider.value() / 100.0   # Convert 0-100 to 0-1
                noisy_image = add_salt_and_pepper_noise(img_to_process, ratio, amount)
                applied_noise_type = f"Sal y Pimienta ({amount*100:.1f}%)"

            elif self.radio_gaussian.isChecked():
                std_dev = self.gauss_stddev_slider.value()
                mean = 0 # Keep mean at 0 for simplicity
                noisy_image = add_gaussian_noise(img_to_process, mean, std_dev)
                applied_noise_type = f"Gaussiano (StdDev={std_dev})"

            # Show the noisy image (this calls display_processed_image)
            if noisy_image is not None:
                self.display_processed_image(noisy_image, f"Imagen con Ruido: {applied_noise_type}")
                self._cancel_threshold_worker() # Cancel any threshold worker if active
                self._is_noisy = True # Mark as noisy

            else:
                 # This case should ideally not be reached if add_noise functions handle None input
                 print("Error: noisy_image is None after applying noise function.")
                 self.display_processed_image(self.active_image.copy() if self.active_image is not None else None, f"Error al aplicar Ruido")
                 self._cancel_threshold_worker()
                 self._is_noisy = False # Reset noisy state on error

        except Exception as e:
             QMessageBox.critical(self, "Error al Aplicar Ruido", f"Ocurrió un error al aplicar ruido: {e}")
             print(f"Error al aplicar ruido: {e}")
             traceback.print_exc() # Print full traceback for debugging
             self.display_processed_image(self.active_image.copy() if self.active_image is not None else None, f"Error al aplicar Ruido")
             self._cancel_threshold_worker()
             self._is_noisy = False # Reset noisy state on error


    # --- Slider Label Update Methods for Noise Controls ---
    def update_sp_amount_label(self, value):
        self.sp_amount_label.setText(f"Cantidad: {value / 100.0:.1f} %")

    def update_sp_ratio_label(self, value):
         self.sp_ratio_label.setText(f"Proporción Sal: {value / 100.0:.1f} %")

    def update_gauss_stddev_label(self, value):
         self.gauss_stddev_label.setText(f"Desv. Estándar: {value}")

    # --- End Noise Operations ---


    # --- Methods to show filter categories and groups ---

    def show_filter_categories(self):
        """Shows the filter category buttons."""
        # This operation should work on the active image
        if self.active_image is not None:
            self.hide_all_controls()
            self.linear_filters_button.setVisible(True)
            self.nonlinear_filters_button.setVisible(True)
            self.edge_filters_button.setVisible(True) # Show the edge category button
            # Display the current processed image, or active image if no processed image
            img_to_show = self.processed_image.copy() if isinstance(self.processed_image, np.ndarray) else (self.active_image.copy() if isinstance(self.active_image, np.ndarray) else None)
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(img_to_show, f"Selecciona una Categoría de Filtro")
            self._cancel_threshold_worker()
        else:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo seleccionada para aplicar filtros.")

    # _hide_filter_groups is defined above along with hide_all_controls

    def show_linear_filters_group(self):
        """Shows the linear filters group and hides the others."""
        # This operation should work on the active image
        if self.active_image is not None:
            # It's good practice to hide *all* controls first, then show the desired ones
            self.hide_all_controls()
            # Then, show the linear category button and its group
            self.linear_filters_button.setVisible(True) # Keep the active category button visible
            self.linear_filters_group.setVisible(True)
            # Display the currently processed image, or active image if no processed image
            img_to_show = self.processed_image.copy() if isinstance(self.processed_image, np.ndarray) else (self.active_image.copy() if isinstance(self.active_image, np.ndarray) else None)
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(img_to_show, f"Filtros Lineales (Pasa-bajas)")
            self._cancel_threshold_worker()


    def show_nonlinear_filters_group(self):
        """Shows the non-linear filters group and hides the others."""
        # This operation should work on the active image
        if self.active_image is not None:
            self.hide_all_controls()
            self.nonlinear_filters_button.setVisible(True) # Keep visible
            self.nonlinear_filters_group.setVisible(True)
            # Display the currently processed image, or active image if no processed image
            img_to_show = self.processed_image.copy() if isinstance(self.processed_image, np.ndarray) else (self.active_image.copy() if isinstance(self.active_image, np.ndarray) else None)
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(img_to_show, f"Filtros No Lineales (De Orden)")
            self._cancel_threshold_worker()

    # --- New method to show the edge filters group ---
    def show_edge_filters_group(self):
        self.hide_all_controls()
        self.edge_filters_group.setVisible(True)
        self.set_operation_buttons_enabled(True)

        # Habilitar/deshabilitar botones según haya imagen activa
        has_active_image = self.active_image is not None
        self.btn_canny.setEnabled(has_active_image)
        self.btn_sobel.setEnabled(has_active_image)
        self.btn_prewitt.setEnabled(has_active_image)
        self.btn_laplacian.setEnabled(has_active_image)
        self.btn_robinson.setEnabled(has_active_image)
        self.btn_kirsch.setEnabled(has_active_image)

    # --- FILTER FUNCTIONS (Operate on self.processed_image.copy() if noisy, otherwise self.active_image.copy()) ---

    # Helper to get the correct image for filter operations
    # This helper is crucial. It should return the image *before* display adjustments
    def _get_image_for_filter(self):
        """Returns the image that will be used for filter operations based on the noisy state."""
        # If the current displayed image is marked as noisy, use the processed image as the base for filtering
        if self._is_noisy and isinstance(self.processed_image, np.ndarray):
            print("Aplicando filtro a la imagen procesada ruidosa.")
            return self.processed_image.copy()
        # Otherwise, use the active image (original or last non-noisy result)
        elif isinstance(self.active_image, np.ndarray):
            print("Aplicando filtro a la active image.")
            return self.active_image.copy()
        else:
            print("No hay imagen disponible para _get_image_for_filter.")
            return None # No image available for processing


    # Apply Averaging Filter (Blur)
    def apply_blur(self):
        print("Attempting to apply Averaging Filter...")
        img_to_process = self._get_image_for_filter() # Get the correct image based on noisy state
        if img_to_process is None:
             QMessageBox.warning(self, "Advertencia", "No hay imagen para aplicar filtro Promediador.")
             print("apply_blur: img_to_process is None.")
             return

        kernel_size = (5, 5) # Kernel size
        try:
             print(f"Aplicando cv2.blur with kernel {kernel_size} to image of shape {img_to_process.shape} and dtype {img_to_process.dtype}")
             blur = cv2.blur(img_to_process, kernel_size)
             if blur is not None:
                 print(f"cv2.blur applied. Result shape: {blur.shape}, dtype: {blur.dtype}")
                 # display_processed_image is called, which will apply display adjustments
                 self.display_processed_image(blur, f"Filtro Promediador ({kernel_size[0]}x{kernel_size[1]})")
                 self._cancel_threshold_worker()
                 self._is_noisy = False
             else:
                 print("cv2.blur returned None.")
                 self.display_processed_image(None, f"Error Filtro Promediador")
                 self._cancel_threshold_worker()
                 self._is_noisy = False


        except cv2.error as e:
             print(f"OpenCV error in blur: {e}")
             QMessageBox.critical(self,"Error en Promediador", f"Error de OpenCV en blur: {e}")
             self.display_processed_image(None, f"Error Filtro Promediador")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             print(f"General error in apply_blur: {e}")
             traceback.print_exc()
             QMessageBox.critical(self,"Error General en Promediador", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, f"Error Filtro Promediador")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Weighted Averaging Filter
    def apply_weighted_average(self):
        print("Attempting to apply Weighted Averaging Filter...")
        img_to_process = self._get_image_for_filter() # Get the correct image based on noisy state
        if img_to_process is None:
             QMessageBox.warning(self, "Advertencia", "No hay imagen para aplicar filtro Promediador Pesado.")
             print("apply_weighted_average: img_to_process is None.")
             return

        kernel = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype=np.float32)
        kernel = kernel / np.sum(kernel) # Normalize the kernel so the sum is 1
        print(f"Normalized Weighted Average Kernel:\n{kernel}")
        try:
             print(f"Aplicando cv2.filter2D with kernel to image of shape {img_to_process.shape} and dtype {img_to_process.dtype}")
             weighted_avg = cv2.filter2D(img_to_process, -1, kernel) # Apply 2D filter with kernel
             if weighted_avg is not None:
                 print(f"cv2.filter2D applied. Result shape: {weighted_avg.shape}, dtype: {weighted_avg.dtype}")
                 # display_processed_image is called, which will apply display adjustments
                 self.display_processed_image(weighted_avg, f"Filtro Promediador Pesado (3x3)")
                 self._cancel_threshold_worker()
                 self._is_noisy = False
             else:
                 print("cv2.filter2D returned None.")
                 self.display_processed_image(None, f"Error Filtro Promediador Pesado")
                 self._cancel_threshold_worker()
                 self._is_noisy = False


        except cv2.error as e:
             print(f"OpenCV error in filter2D: {e}")
             QMessageBox.critical(self,"Error en Promediador Pesado", f"Error de OpenCV en filter2D: {e}")
             self.display_processed_image(None, f"Error Filtro Promediador Pesado")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             print(f"General error in apply_weighted_average: {e}")
             traceback.print_exc()
             QMessageBox.critical(self,"Error General en Promediador Pesado", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, f"Error Filtro Promediador Pesado")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Gaussian Filter
    def apply_gaussian(self):
        print("Attempting to apply Gaussian Filter...")
        img_to_process = self._get_image_for_filter() # Get the correct image based on noisy state
        if img_to_process is None:
             QMessageBox.warning(self, "Advertencia", "No hay imagen para aplicar filtro Gaussiano.")
             print("apply_gaussian: img_to_process is None.")
             return

        kernel_size = (5, 5) # Kernel size (must be odd)
        try:
             print(f"Aplicando cv2.GaussianBlur with kernel {kernel_size} to image of shape {img_to_process.shape} and dtype {img_to_process.dtype}")
             gauss = cv2.GaussianBlur(img_to_process, kernel_size, 0) # Apply Gaussian filter
             if gauss is not None:
                 print(f"cv2.GaussianBlur applied. Result shape: {gauss.shape}, dtype: {gauss.dtype}")
                 # display_processed_image is called, which will apply display adjustments
                 self.display_processed_image(gauss, f"Filtro Gaussiano ({kernel_size[0]}x{kernel_size[1]})")
                 self._cancel_threshold_worker()
                 self._is_noisy = False
             else:
                 print("cv2.GaussianBlur returned None.")
                 self.display_processed_image(None, f"Error Filtro Gaussiano")
                 self._cancel_threshold_worker()
                 self._is_noisy = False


        except cv2.error as e:
             print(f"OpenCV error in GaussianBlur: {e}")
             QMessageBox.critical(self,"Error en Gaussiano", f"Error de OpenCV en GaussianBlur: {e}")
             self.display_processed_image(None, f"Error Filtro Gaussiano")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             print(f"General error in apply_gaussian: {e}")
             traceback.print_exc()
             QMessageBox.critical(self,"Error General en Gaussiano", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, f"Error Filtro Gaussiano")
             self._cancel_threshold_worker()
             self._is_noisy = False

    # Apply Median Filter
    def apply_median(self):
        if not SCIPY_AVAILABLE: # Check if Scipy is available
            QMessageBox.critical(self, "Error de Dependencia", "Se necesita 'scipy'. Instala con: pip install scipy")
            return
        img_to_process = self._get_image_for_filter() # Get the correct image based on noisy state
        if img_to_process is None:
             QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para aplicar filtro Mediana.")
             return

        kernel_size = 5 # Kernel size (must be odd and greater than 1)
        # Median filter only works on 8-bit single-channel or 8-bit 3-channel images.
        # Need to handle different types or channel counts
        if img_to_process.dtype != np.uint8:
             QMessageBox.warning(self, "Advertencia", f"El filtro de mediana requiere imágenes de 8-bit (uint8). Imagen de trabajo es {img_to_process.dtype}. Intentando convertir...")
             try:
                 img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
                 print("Imagen converted to uint8 for median filter.")
             except Exception as e_conv:
                  QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para filtro de mediana: {e_conv}")
                  self._is_noisy = False
                  return


        if img_to_process.ndim == 3 and img_to_process.shape[2] == 4:
             QMessageBox.warning(self, "Advertencia", "El filtro de mediana no soporta imágenes con canal alfa (4 canales).")
             self._is_noisy = False
             return


        try:
             med = cv2.medianBlur(img_to_process, kernel_size) # Apply Median filter
             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(med, f"Filtro Mediana (Kernel {kernel_size})")
             self._cancel_threshold_worker()
             self._is_noisy = False


        except cv2.error as e:
             QMessageBox.critical(self,"Error en Mediana", f"Error de OpenCV en MedianBlur: {e}")
             self.display_processed_image(None, f"Error Filtro Mediana")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Mode Filter
    def apply_mode_filter(self):
        if not SCIPY_AVAILABLE:
            QMessageBox.critical(self, "Error de Dependencia", "Se necesita 'scipy'. Instala con: pip install scipy")
            return

        try:
            from scipy import stats
            from scipy import ndimage
        except ImportError as e:
            QMessageBox.critical(self, "Error de Importación", f"No se pudo importar scipy: {e}")
            return

        img_to_process = self._get_image_for_filter()
        if img_to_process is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para aplicar filtro de Moda.")
            return

        kernel_size = 3

        def calculate_mode(arr):
            try:
                # Convertir a entero para evitar problemas con tipos flotantes
                arr_int = arr.astype(np.int32)
                # Usar stats.mode con keepdims=False para obtener un valor escalar
                mode_result = stats.mode(arr_int, axis=None, keepdims=False)
                # Extraer el valor de la moda
                mode_val = mode_result.mode
                return int(mode_val)  # Asegurar que devolvemos un entero
            except Exception as e:
                print(f"Error calculando moda: {e}")
                # Si falla, usar la mediana como respaldo
                return int(np.median(arr))

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Verificar y convertir el tipo de datos
            if img_to_process.dtype != np.uint8:
                print(f"Convirtiendo imagen de {img_to_process.dtype} a uint8")
                img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)

            mode_filtered = None

            if img_to_process.ndim == 2:  # Escala de grises
                print(f"Aplicando filtro de moda ({kernel_size}x{kernel_size}) a imagen en escala de grises...")
                mode_filtered = ndimage.generic_filter(img_to_process, calculate_mode, size=kernel_size)
            elif img_to_process.ndim == 3:  # Color
                if img_to_process.shape[2] == 4:
                    print("Convirtiendo imagen RGBA a BGR para filtro de moda")
                    img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGRA2BGR)

                if img_to_process.shape[2] == 3:
                    print(f"Aplicando filtro de moda ({kernel_size}x{kernel_size}) a imagen color (BGR)...")
                    mode_filtered = np.zeros_like(img_to_process)
                    for i in range(img_to_process.shape[2]):
                        mode_filtered[:, :, i] = ndimage.generic_filter(img_to_process[:, :, i], calculate_mode, size=kernel_size)
                else:
                    raise ValueError(f"Formato de imagen no soportado: {img_to_process.shape[2]} canales")
            else:
                raise ValueError("La imagen debe ser 2D (escala de grises) o 3D (color)")

            # Asegurar que el resultado es uint8
            mode_filtered = np.clip(mode_filtered, 0, 255).astype(np.uint8)

            # Mostrar el resultado
            self.display_processed_image(mode_filtered, f"Filtro de Moda ({kernel_size}x{kernel_size})")
            self._cancel_threshold_worker()
            self._is_noisy = False

        except Exception as e:
            print(f"Error aplicando filtro de moda: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error en Filtro de Moda", f"Ocurrió un error: {e}")
            self.display_processed_image(None, "Error Filtro Moda")
            self._cancel_threshold_worker()
            self._is_noisy = False
        finally:
            QApplication.restoreOverrideCursor()

    # Apply Maximum Filter (Dilation)
    def apply_max_filter(self):
        img_to_process = self._get_image_for_filter() # Get the correct image based on noisy state
        if img_to_process is None:
             QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para aplicar filtro Máximo.")
             return

        kernel_size = 5 # Kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8) # Rectangular kernel of ones
        # Morphological operations usually work best on binary or grayscale images.
        # If it's a color image, they are applied per channel.
        if img_to_process.dtype != np.uint8:
             QMessageBox.warning(self, "Advertencia", f"La Dilación requiere imágenes de 8-bit (uint8). Imagen de trabajo es {img_to_process.dtype}. Intentando convertir...")
             try:
                 img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
                 print("Imagen converted to uint8 for dilation.")
             except Exception as e_conv:
                  QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para Dilación: {e_conv}")
                  self._is_noisy = False
                  return

        if img_to_process.ndim == 3 and img_to_process.shape[2] == 4:
             QMessageBox.warning(self, "Advertencia", "La Dilación no soporta imágenes con canal alfa (4 canales).")
             self._is_noisy = False
             return


        try:
             max_filtered = cv2.dilate(img_to_process, kernel, iterations=1) # Apply dilation
             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(max_filtered, f"Filtro Máximo (Dilación {kernel_size}x{kernel_size})")
             self._cancel_threshold_worker()
             self._is_noisy = False # After dilation, the image is not noisy


        except cv2.error as e:
             QMessageBox.critical(self,"Error en Dilación", f"Error de OpenCV en dilate: {e}")
             self.display_processed_image(None, f"Error Filtro Máximo")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Minimum Filter (Erosion)
    def apply_min_filter(self):
        img_to_process = self._get_image_for_filter() # Get the correct image based on noisy state
        if img_to_process is None:
             QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para aplicar filtro Mínimo.")
             return

        kernel_size = 5 # Kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8) # Rectangular kernel of ones
         # Morphological operations usually work best on binary or grayscale images.
        # If it's a color image, they are applied per channel.
        if img_to_process.dtype != np.uint8:
             QMessageBox.warning(self, "Advertencia", f"La Erosión requiere imágenes de 8-bit (uint8). Imagen de trabajo es {img_to_process.dtype}. Intentando convertir...")
             try:
                 img_to_process = np.clip(img_to_process, 0, 255).astype(np.uint8)
                 print("Imagen converted to uint8 for erosion.")
             except Exception as e_conv:
                  QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para Erosión: {e_conv}")
                  self._is_noisy = False
                  return

        if img_to_process.ndim == 3 and img_to_process.shape[2] == 4:
             QMessageBox.warning(self, "Advertencia", "La Erosión no soporta imágenes con canal alfa (4 canales).")
             self._is_noisy = False
             return

        try:
            min_filtered = cv2.erode(img_to_process, kernel, iterations=1) # Apply erosion
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(min_filtered, f"Filtro Mínimo (Erosión {kernel_size}x{kernel_size})")
            self._cancel_threshold_worker()
            self._is_noisy = False # After erosion, the image is not noisy


        except cv2.error as e:
             QMessageBox.critical(self,"Error en Erosión", f"Error de OpenCV en erode: {e}")
             self.display_processed_image(None, f"Error Filtro Mínimo")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # --- Edge Detection Filter Functions (Implemented directly) ---

    def _get_grayscale_image_for_edge_filter(self):
        """Helper to get a uint8 grayscale image for edge detection."""
        img_to_process = self._get_image_for_filter() # Get the image based on noise state
        if img_to_process is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen de trabajo para detección de bordes.")
            return None

        # Convert to grayscale if needed
        gray = None
        if img_to_process.ndim == 3:
             if img_to_process.shape[2] == 4: # BGRA
                 gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGRA2GRAY)
             elif img_to_process.shape[2] == 3: # BGR
                 gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
             else:
                 QMessageBox.warning(self, "Advertencia", f"Formato de imagen con {img_to_process.shape[2]} canales no válido para detección de bordes.")
                 # Display the original active image with an info message
                 self.display_processed_image(self.active_image.copy(), f"Formato no compatible con Detección de Bordes")
                 self._is_noisy = False
                 return None
        elif img_to_process.ndim == 2: # Already grayscale
             gray = img_to_process.copy()
        else:
             QMessageBox.warning(self, "Advertencia", "Formato de imagen no válido para detección de bordes.")
             self.display_processed_image(self.active_image.copy(), f"Formato no compatible con Detección de Bordes")
             self._is_noisy = False
             return None

        # Ensure the grayscale image is 8-bit (uint8)
        if gray.dtype != np.uint8:
            QMessageBox.warning(self, "Advertencia", f"La detección de bordes requiere imágenes de 8-bit (uint8). Imagen de trabajo es {gray.dtype}. Intentando convertir...")
            try:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
                print("Imagen converted to uint8 for edge detection.")
            except Exception as e_conv:
                 QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para detección de bordes: {e_conv}")
                 self.display_processed_image(None, f"Error al obtener imagen para Bordes")
                 self._is_noisy = False
                 return None

        return gray


    # Apply Canny Edge Detector
    def apply_canny_filter(self):
        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        # Get thresholds from class attributes (updated by sliders if visible)
        low_threshold = self.canny_low_threshold
        high_threshold = self.canny_high_threshold

        try:
            # Implement Canny logic directly
            edges = cv2.Canny(gray_img, low_threshold, high_threshold)
            # Canny returns a 1-channel grayscale image (uint8)
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(edges, f"Bordes Canny (T1={low_threshold}, T2={high_threshold})")
            self._cancel_threshold_worker()
            self._is_noisy = False # After edge detection, the image is not noisy

        except cv2.error as e:
            QMessageBox.critical(self,"Error en Canny", f"Error de OpenCV al aplicar Canny: {e}")
            self.display_processed_image(None, f"Error Filtro Canny")
            self._cancel_threshold_worker()
            self._is_noisy = False
        except Exception as e:
             QMessageBox.critical(self,"Error General en Canny", f"Ocurrió un error inesperado al aplicar Canny: {e}")
             self.display_processed_image(None, f"Error Filtro Canny")
             self._cancel_threshold_worker()
             self._is_noisy = False

    # Method to toggle visibility of Canny threshold controls
    def toggle_canny_controls_visibility(self):
        """Muestra los sliders de Canny solo cuando el filtro Canny está activo."""
        # Asegurar que el grupo de filtros de bordes esté visible
        self.show_edge_filters_group()
        
        # Obtener el estado actual de visibilidad
        is_visible = self.canny_low_thresh_label.isVisible()
        
        # Ocultar todos los controles de Canny primero
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        
        # Si estaban ocultos, mostrarlos ahora
        if not is_visible:
            self.canny_low_thresh_label.setVisible(True)
            self.canny_low_thresh_slider.setVisible(True)
            self.canny_high_thresh_label.setVisible(True)
            self.canny_high_thresh_slider.setVisible(True)
            
            # Actualizar etiquetas con valores actuales
            self.canny_low_thresh_label.setText(f"Umbral Bajo: {self.canny_low_threshold}")
            self.canny_high_thresh_label.setText(f"Umbral Alto: {self.canny_high_threshold}")
            
            # Aplicar el filtro Canny con los valores actuales
            self.apply_canny_filter()
        
        # Actualizar estado de los botones
        self.set_operation_buttons_enabled(True)

    def update_canny_low_thresh(self, value):
        """Updates the low threshold value and reapplies the Canny filter."""
        self.canny_low_threshold = value
        self.canny_low_thresh_label.setText(f"Umbral Bajo: {value}")
        if self.canny_low_thresh_slider.isVisible():
            self.apply_canny_filter()

    def update_canny_high_thresh(self, value):
        """Updates the high threshold value and reapplies the Canny filter."""
        self.canny_high_threshold = value
        self.canny_high_thresh_label.setText(f"Umbral Alto: {value}")
        if self.canny_high_thresh_slider.isVisible():
            self.apply_canny_filter()

    def apply_canny_filter(self):
        """Applies the Canny edge detection filter with current threshold values."""
        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return  # Warning handled inside helper

        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_img, self.canny_low_threshold, self.canny_high_threshold)
            
            # Display the result
            self.display_processed_image(edges, f"Bordes Canny (T1={self.canny_low_threshold}, T2={self.canny_high_threshold})")
            self._cancel_threshold_worker()
            self._is_noisy = False
            
        except cv2.error as e:
            QMessageBox.critical(self, "Error en Canny", f"Error de OpenCV en Canny: {e}")
            self.display_processed_image(None, "Error Filtro Canny")
            self._cancel_threshold_worker()
            self._is_noisy = False
        except Exception as e:
            QMessageBox.critical(self, "Error General en Canny", f"Ocurrió un error inesperado: {e}")
            self.display_processed_image(None, "Error Filtro Canny")
            self._cancel_threshold_worker()
            self._is_noisy = False

    # Apply Horizontal Edge Filter
    def apply_horizontal_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
             # Implement horizontal edge logic directly
             # Define the horizontal kernel (using the one from your reference)
             kernel_horizontal = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
             # Apply the filter
             # Use -1 for ddepth to match the source image depth (uint8)
             result = cv2.filter2D(gray_img, -1, kernel_horizontal)

             # The result of this specific horizontal kernel might have negative values.
             # To display it correctly as an image, we should normalize it to 0-255.
             # Using cv2.normalize for safe scaling.
             result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(result, "Borde Horizontal (Convolución)")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except cv2.error as e:
             QMessageBox.critical(self,"Error en Borde Horizontal", f"Error de OpenCV al aplicar filtro 2D horizontal: {e}")
             self.display_processed_image(None, "Error Borde Horizontal")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             QMessageBox.critical(self,"Error General en Borde Horizontal", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, "Error Borde Horizontal")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Vertical Edge Filter
    def apply_vertical_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
             # Implement vertical edge logic directly
             # Define the vertical kernel (using the one from your reference)
             kernel_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
             # Apply the filter
             # Use -1 for ddepth to match the source image depth (uint8)
             result = cv2.filter2D(gray_img, -1, kernel_vertical)

             # The result of this specific vertical kernel might have negative values.
             # To display it correctly as an image, we should normalize it to 0-255.
             # Using cv2.normalize for safe scaling.
             result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(result, "Borde Vertical (Convolución)")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except cv2.error as e:
             QMessageBox.critical(self,"Error en Borde Vertical", f"Error de OpenCV al aplicar filtro 2D vertical: {e}")
             self.display_processed_image(None, "Error Borde Vertical")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except Exception as e:
             QMessageBox.critical(self,"Error General en Borde Vertical", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, "Error Borde Vertical")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Sobel Filter (Based on your provided code - magnitude)
    def apply_sobel_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
             # Implement Sobel logic based on your provided code (magnitude)
             # Apply Sobel filters in X and Y direction
             # Use cv2.CV_64F for intermediate calculations to avoid overflow
             sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3) # ksize=3 is standard for basic Sobel
             sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3) # ksize=3 is standard for basic Sobel

             # Compute the magnitude of the gradients using cv2.magnitude
             # This is equivalent to np.sqrt(sobelx**2 + sobely**2) but can be slightly more robust
             result = cv2.magnitude(sobelx, sobely)

             # Normalize the result to 0-255 for display using safe_normalize logic
             # The display_processed_image helper already handles normalization for float images,
             # but explicitly normalizing here ensures it's done as part of the filter logic
             # and matches the behavior in your provided code snippet.
             result = self.safe_normalize(result)


             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(result, "Filtro Sobel (Magnitud)")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except cv2.error as e:
             QMessageBox.critical(self,"Error en Sobel", f"Error de OpenCV al aplicar Sobel: {e}")
             self.display_processed_image(None, "Error Filtro Sobel")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             QMessageBox.critical(self,"Error General en Sobel", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, "Error Filtro Sobel")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Prewitt Filter (Based on your provided code - addWeighted)
    def apply_prewitt_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
             # Implement Prewitt logic based on your provided code (addWeighted)
             # Define Prewitt kernels
             kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
             kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

             # Apply the filters
             # Use cv2.CV_64F for intermediate calculations to avoid overflow
             imgx = cv2.filter2D(gray_img, cv2.CV_64F, kernelx)
             imgy = cv2.filter2D(gray_img, cv2.CV_64F, kernely)

             # Compute the combined result using addWeighted on the absolute values
             # This matches the approach in your provided snippet
             abs_imgx = np.abs(imgx)
             abs_imgy = np.abs(imgy)
             # Ensure the inputs to addWeighted are of the same type (float64 here)
             result = cv2.addWeighted(abs_imgx, 0.5, abs_imgy, 0.5, 0)

             # Normalize the result to 0-255 for display using safe_normalize logic
             result = self.safe_normalize(result)

             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(result, "Filtro Prewitt (Combinado)")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except cv2.error as e:
             QMessageBox.critical(self,"Error en Prewitt", f"Error de OpenCV al aplicar Prewitt: {e}")
             self.display_processed_image(None, "Error Filtro Prewitt")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             QMessageBox.critical(self,"Error General en Prewitt", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, "Error Filtro Prewitt")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Laplacian Filter
    def apply_laplacian_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
             # Implement Laplacian logic directly
             # Apply Laplacian filter
             laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3) # Use 64F for intermediate calculations, ksize=3 is common

             # Compute absolute values and normalize to uint8
             result = self.safe_normalize(np.abs(laplacian))

             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(result, "Filtro Laplaciano")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except cv2.error as e:
             QMessageBox.critical(self,"Error en Laplaciano", f"Error de OpenCV al aplicar Laplaciano: {e}")
             self.display_processed_image(None, "Error Filtro Laplaciano")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             QMessageBox.critical(self,"Error General en Laplaciano", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, "Error Filtro Laplaciano")
             self._cancel_threshold_worker()
             self._is_noisy = False


    # Apply Scharr Filter (Based on your provided code - magnitude)
    def apply_scharr_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
             # Implement Scharr logic based on your provided code (magnitude)
             # Apply Scharr filters in X and Y direction
             # Scharr is a more sensitive version of Sobel for 3x3 kernels
             scharrx = cv2.Scharr(gray_img, cv2.CV_64F, 1, 0) # dx=1, dy=0 (ksize is 3 implicitly for Scharr)
             scharry = cv2.Scharr(gray_img, cv2.CV_64F, 0, 1) # dx=0, dy=1

             # Compute the magnitude of the gradients using np.sqrt
             # This matches the approach in your provided snippet
             result = np.sqrt(scharrx**2 + scharry**2)

             # Normalize the result to 0-255 for display using safe_normalize logic
             result = self.safe_normalize(result)


             # display_processed_image is called, which will apply display adjustments
             self.display_processed_image(result, "Filtro Scharr (Magnitud)")
             self._cancel_threshold_worker()
             self._is_noisy = False

        except cv2.error as e:
             QMessageBox.critical(self,"Error en Scharr", f"Error de OpenCV al aplicar Scharr: {e}")
             self.display_processed_image(None, "Error Filtro Scharr")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             QMessageBox.critical(self,"Error General en Scharr", f"Ocurrió un error inesperado: {e}")
             self.display_processed_image(None, "Error Filtro Scharr")
             self._cancel_threshold_worker()
             self._is_noisy = False

    # Apply Robinson Filter (Using maximum response)
    def apply_robinson_filter(self):
        if self.active_image is None:
            return

        # Ocultar controles de Canny
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)

        # Obtener el botón que disparó la señal
        sender = self.sender()
        
        # Si es el botón principal de Robinson, mostrar/ocultar los botones direccionales
        if sender == self.btn_robinson:
            self.robinson_directions_group.setVisible(not self.robinson_directions_group.isVisible())
            return

        # Mapeo de botones a direcciones
        direction_map = {
            self.btn_robinson_norte: "North",
            self.btn_robinson_noreste: "NorthEast",
            self.btn_robinson_este: "East",
            self.btn_robinson_sureste: "SouthEast",
            self.btn_robinson_sur: "South",
            self.btn_robinson_suroeste: "SouthWest",
            self.btn_robinson_oeste: "West",
            self.btn_robinson_noroeste: "NorthWest",
            self.btn_robinson_completo: "Complete"
        }

        # Obtener la dirección del botón presionado
        direction = direction_map.get(sender)
        if not direction:
            return

        try:
            # Convertir la imagen a escala de grises si es necesario
            if len(self.active_image.shape) == 3:
                gray = cv2.cvtColor(self.active_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.active_image.copy()

            # Aplicar el filtro según la dirección
            if direction == "Complete":
                # Aplicar todas las máscaras y tomar el máximo
                result = np.zeros_like(gray, dtype=np.float32)
                for kernel in self.robinson_kernels.values():
                    filtered = cv2.filter2D(gray, -1, kernel)
                    result = np.maximum(result, filtered)
            else:
                # Aplicar la máscara específica
                kernel = self.robinson_kernels[direction]
                result = cv2.filter2D(gray, -1, kernel)

            # Normalizar el resultado para visualización
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)

            # Mostrar el resultado
            self.display_processed_image(result, f"Filtro Robinson - {direction}")

        except cv2.error as e:
            QMessageBox.critical(self, "Error", f"Error de OpenCV: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al aplicar el filtro Robinson: {str(e)}")

    # Apply Kirsch Filter (Using maximum response)
    def apply_kirsch_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
            # Apply each Kirsch kernel and find the maximum response
            max_response = np.zeros_like(gray_img, dtype=np.float32) # Initialize with zeros
            for direction, kernel in self.kirsch_kernels.items():
                # Apply the kernel
                # Use cv2.CV_32F for intermediate calculations to handle potential negative values
                response = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
                # Update the maximum response at each pixel
                max_response = np.maximum(max_response, np.abs(response)) # Use absolute value of response

            # Normalize the result to 0-255 for display
            result = self.safe_normalize(max_response)

            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(result, "Filtro Kirsch (Máx. Respuesta)")
            self._cancel_threshold_worker()
            self._is_noisy = False

        except cv2.error as e:
            QMessageBox.critical(self,"Error en Kirsch", f"Error de OpenCV al aplicar Kirsch: {e}")
            self.display_processed_image(None, "Error Filtro Kirsch")
            self._cancel_threshold_worker()
            self._is_noisy = False
        except Exception as e:
            QMessageBox.critical(self,"Error General en Kirsch", f"Ocurrió un error inesperado: {e}")
            self.display_processed_image(None, "Error Filtro Kirsch")
            self._cancel_threshold_worker()
            self._is_noisy = False

    # Apply Roberts Filter (Using magnitude)
    def apply_roberts_filter(self):
        # Hide Canny sliders when another edge filter is clicked
        self.canny_low_thresh_label.setVisible(False)
        self.canny_low_thresh_slider.setVisible(False)
        self.canny_high_thresh_label.setVisible(False)
        self.canny_high_thresh_slider.setVisible(False)
        self.set_operation_buttons_enabled(True) # Update enabled state

        gray_img = self._get_grayscale_image_for_edge_filter()
        if gray_img is None:
            return # Warning handled inside helper

        try:
            # Apply the two Roberts kernels
            # Use cv2.CV_32F for intermediate calculations
            response1 = cv2.filter2D(gray_img, cv2.CV_32F, self.roberts_kernels["Diagonal1"])
            response2 = cv2.filter2D(gray_img, cv2.CV_32F, self.roberts_kernels["Diagonal2"])

            # Compute the magnitude of the responses
            result = cv2.magnitude(response1, response2)

            # Normalize the result to 0-255 for display
            result = self.safe_normalize(result)

            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(result, "Filtro Roberts (Magnitud)")
            self._cancel_threshold_worker()
            self._is_noisy = False

        except cv2.error as e:
            QMessageBox.critical(self,"Error en Roberts", f"Error de OpenCV al aplicar Roberts: {e}")
            self.display_processed_image(None, "Error Filtro Roberts")
            self._cancel_threshold_worker()
            self._is_noisy = False
        except Exception as e:
            QMessageBox.critical(self,"Error General en Roberts", f"Ocurrió un error inesperado: {e}")
            self.display_processed_image(None, "Error Filtro Roberts")
            self._cancel_threshold_worker()
            self._is_noisy = False

    # --- Helper function for safe normalization (copied from your provided code) ---
    def safe_normalize(self, img):
        """Safely normalizes to avoid OpenCV errors."""
        # Check if the image is valid and contains non-zero values
        if img is None or not np.any(img):
            # If the image is empty or all zeros, return an array of zeros of the same shape (if shape exists)
            if img is not None:
                 QMessageBox.information(self, "Normalización", "No se detectaron bordes significativos para normalizar.")
                 # Return a zero image of the same shape and dtype uint8
                 return np.zeros_like(img, dtype=np.uint8)
            else:
                 # If img is None, return an empty uint8 array or handle as needed
                 QMessageBox.warning(self, "Normalización", "No se puede normalizar: imagen de entrada es None.")
                 return np.zeros((1, 1), dtype=np.uint8) # Return a small zero image as fallback


        # Ensure the input image is a floating-point type for normalization
        if img.dtype not in [np.float32, np.float64]:
            img = img.astype(np.float32) # Convert to float if not already

        # Normalize the image to the range [0, 255]
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return norm_img
    # --- End Helper function for safe normalization ---


    # --- End Edge Detection Filter Methods ---


    # --- Histogram Methods ---

    # Show/Hide the histogram controls group
    def show_histogram_controls(self):
        # Histograms can be generated for Image 1 or the currently processed image
        if self.image is None and self.processed_image is None:
            QMessageBox.warning(self, "Advertencia", "Carga/procesa una imagen para ver el histograma.")
            return
        try:
            self.hide_all_controls() # Hide other groups
            # Toggle visibility of histogram group
            self.histogram_controls_group.setVisible(not self.histogram_controls_group.isVisible())

            # If the group was just shown, display the currently processed image (or the original if no processed) in the central panel
            if self.histogram_controls_group.isVisible():
                img_to_show = self.processed_image if self.processed_image is not None else self.image
                info_text = "Generar Histograma"
                if self.processed_image is not None:
                    info_text = self.processed_info_label.text().replace('⚙️ ', '') # Clean emoji/symbol
                    info_text = f"Generar Histograma ({info_text})"
                elif self.image is not None:
                    info_text = "Generar Histograma (Imagen Original 1)"

                # Show the current image (copy for safety)
                # display_processed_image is called, which will apply display adjustments
                if img_to_show is not None:
                    self.display_processed_image(img_to_show.copy(), info_text)
                else:
                    self.display_processed_image(None, info_text) # Show empty state if no image
                self._cancel_threshold_worker()
                # No change to _is_noisy here as histogram is only a visualization
        except Exception as e:
            QMessageBox.critical(self, "Error en Histograma", f"Error de OpenCV al generar histograma: {e}")
            self.display_processed_image(None, "Error Histograma")
            self._clear_histogram_window_reference()

    # Generate and display the histogram using the advanced Histogram window
    def generate_and_display_histogram(self, image_source):
        img_to_process = None
        title = "Histograma"

        # Determine which image to use based on the selected source
        if image_source == 'original1': # Use explicitly the original Image 1
            if self.image is not None:
                img_to_process = self.image
                title = "Histograma - Imagen Original 1"
            else:
                # Warning if original image 1 is not loaded
                QMessageBox.warning(self, "Advertencia", "Carga la Imagen 1 primero para ver su histograma.")
                return
        elif image_source == 'processed':
            # Use the image currently displayed in the central panel
            if self.processed_image is not None:
                 img_to_process = self.processed_image
                 # Get the info text from the central panel for the histogram window title
                 info_text = self.processed_info_label.text().replace('⚙️ ', '') # Clean emoji/symbol
                 title = f"Histograma - {info_text}"
            else:
                # Warning if no processed image in the central panel
                QMessageBox.warning(self, "Advertencia", "No hay imagen procesada en el panel principal para generar histograma.")
                return
        else:
            # Debug message if source is unknown (should not happen with current buttons)
            print(f"Error: Unknown source '{image_source}' for histogram.")
            return


        if img_to_process is None:
             # Debug message if image is None (should not happen if previous checks work)
             print("Error: No se pudo obtener la imagen para el histograma.")
             return


        # *** Create and show the advanced Histogram window ***
        # Now we create an instance of the new HistogramWindow class (the advanced one)
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor
            # Pass a copy of the image data to the Histogram window
            # It's important to pass a copy so the window doesn't modify the original or processed image
            self.histogram_window = HistogramWindow(img_to_process.copy(), self)
            # Ensure the window is deleted from memory when closed
            self.histogram_window.setAttribute(Qt.WA_DeleteOnClose)
            self.histogram_window.setWindowTitle(title) # Set window title
            # Connect the 'finished' signal of the histogram window to a slot to clear the reference
            self.histogram_window.finished.connect(self._clear_histogram_window_reference)

            # The new Histogram window handles calculating and showing the histogram in its __init__

            self.histogram_window.show() # Show the window
            self.histogram_window.raise_() # Bring the window to front
            self.histogram_window.activateWindow() # Activate the window

            QApplication.restoreOverrideCursor() # Restore cursor

            self._cancel_threshold_worker() # Cancel any threshold worker if active
            # No change to _is_noisy here as histogram is only a visualization

        except Exception as e:
            QApplication.restoreOverrideCursor() # Restore cursor before showing message
            print(f"Error al crear/mostrar ventana de histograma: {e}")
            traceback.print_exc() # Print full traceback for debugging
            QMessageBox.critical(self, "Error de Histograma", f"Ocurrió un error al crear la ventana del histograma: {e}")
            self._cancel_threshold_worker() # Cancel any worker in case of error
            # No change to _is_noisy here as histogram is only a visualization

    # --- New slot to clear the histogram window reference ---
    def _clear_histogram_window_reference(self):
        """Sets self.histogram_window to None when the histogram window closes."""
        print("Histogram window closed. Clearing reference.")
        self.histogram_window = None

    # *** New methods for Brightness and Contrast adjustment ***
    def update_brightness_label(self, value):
        """Updates the brightness label and stores the value."""
        self._brightness_value = value
        self.brightness_label.setText(f"Brillo: {self._brightness_value}")

    def update_contrast_label(self, value):
        """Updates the contrast label and stores the value."""
        # Convert integer slider value (0-200) to float factor (0.0-2.0)
        self._contrast_factor = value / 100.0
        self.contrast_label.setText(f"Contraste: {self._contrast_factor:.1f}")

    def apply_display_adjustments(self):
        """Applies the current brightness and contrast to the *currently displayed* image."""
        # This method is called when sliders are moved or on image display/resize
        # Get the image that is currently stored in self.processed_image
        # If processed_image is None (e.g., no operations applied yet), use active_image
        img_to_display = self.processed_image if isinstance(self.processed_image, np.ndarray) else (self.active_image if isinstance(self.active_image, np.ndarray) else None)

        if img_to_display is None:
            return # Nothing to adjust

        # The display_processed_image method already calls cv_to_qpixmap which applies
        # the brightness and contrast stored in self._brightness_value and self._contrast_factor.
        # So, simply re-calling display_processed_image with the current image
        # will trigger the re-rendering with the new brightness/contrast values.
        current_info_text = self.processed_info_label.text()
        # Remove the prefix "⚙️ " before passing to display_processed_image
        info_text = current_info_text.replace('⚙️ ', '')
        # Pass a copy to prevent display_processed_image from modifying the original array
        self.display_processed_image(img_to_display.copy(), info_text)


    def reset_brightness_contrast(self):
        """Resets brightness and contrast to default values (0 brightness, 1.0 contrast)."""
        # Block signals temporarily to avoid triggering apply_display_adjustments multiple times
        self.brightness_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)

        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100) # 100 corresponds to a factor of 1.0

        # Update the internal values and labels
        self._brightness_value = 0
        self._contrast_factor = 1.0
        self.brightness_label.setText(f"Brillo: {self._brightness_value}")
        self.contrast_label.setText(f"Contraste: {self._contrast_factor:.1f}")

        # Re-enable signals
        self.brightness_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)

        # Apply the reset adjustments if an image is currently displayed
        self.apply_display_adjustments()

    # *** End of new methods ***


    # Window close event
    def closeEvent(self, event):
        # Ensure all OpenCV windows (if any were opened) are closed
        cv2.destroyAllWindows()
        # Ensure the threshold worker stops when the application closes
        self._cancel_threshold_worker() # Use the cancellation method

        # Close all open histogram windows
        # Iterate over the top-level widgets of the application
        for widget in QApplication.topLevelWidgets():
            # If the widget is an instance of HistogramWindow, close it
            if isinstance(widget, HistogramWindow):
                print(f"Cerrando ventana de histograma: {widget.windowTitle()}")
                widget.close() # Close the window
                # The 'finished' signal of this window will now trigger _clear_histogram_window_reference


        # Allow the close event to continue (close the main window)
        super().closeEvent(event)


    # --- NEW: Methods for Connected Components Analysis and Display (Moved inside ImageProcessor) ---

    # Method to show Connected Components controls and perform analysis
    def show_connected_components_controls(self):
        # This operation should work on the active image
        if self.active_image is None:
            QMessageBox.warning(self, "Advertencia", "Carga una imagen de trabajo (Imagen 1 o Imagen 2) primero para el Etiquetado de Componentes Conexas.")
            return

        self.hide_all_controls() # Hide other groups
        self.connected_components_group.setVisible(True) # Show connected components controls

        img_to_process = self.active_image.copy() # Work on a copy of the active image

        # --- Perform the Connected Components Analysis ---
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor

            # 1. Convert to grayscale if necessary
            gray = None
            if img_to_process.ndim == 3:
                if img_to_process.shape[2] == 4: # BGRA
                    gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGRA2GRAY)
                elif img_to_process.shape[2] == 3: # BGR
                    gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
                else:
                    QMessageBox.warning(self, "Advertencia", f"Formato de imagen con {img_to_process.shape[2]} canales no válido para Componentes Conexas.")
                    self.display_processed_image(self.active_image.copy(), f"Formato no compatible con Componentes Conexas")
                    self._is_noisy = False
                    QApplication.restoreOverrideCursor()
                    return
            elif img_to_process.ndim == 2: # Already grayscale
                gray = img_to_process.copy()
            else:
                QMessageBox.warning(self, "Advertencia", "Formato de imagen no válido para Etiquetado de Componentes Conexas.")
                self.display_processed_image(self.active_image.copy(), f"Formato no compatible con Componentes Conexas")
                self._is_noisy = False
                QApplication.restoreOverrideCursor()
                return

             # Ensure the grayscale image is 8-bit (uint8) for thresholding
            if gray.dtype != np.uint8:
                QMessageBox.warning(self, "Advertencia", f"El etiquetado requiere imágenes de 8-bit (uint8). Imagen de trabajo es {gray.dtype}. Intentando convertir...")
                try:
                    gray = np.clip(gray, 0, 255).astype(np.uint8)
                    print("Image converted to uint8 for connected components.")
                except Exception as e_conv:
                     QMessageBox.critical(self, "Error de Conversión", f"No se pudo convertir imagen a uint8 para Componentes Conexas: {e_conv}")
                     self.display_processed_image(None, f"Error al preparar imagen para Componentes Conexas")
                     self._is_noisy = False
                     QApplication.restoreOverrideCursor()
                     return


            # 2. Binarize the image (using Otsu if possible, fallback to fixed threshold)
            try:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print("Umbral binarización (Otsu):", _)
            except Exception as e:
                print(f"Error usando Otsu para binarización, volviendo a umbral fijo: {e}")
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Store the binary image
            self._binary_image = binary.copy()


            # 3. Perform Connected Components Labeling
            self._num_labels_4, self._labels_4 = cv2.connectedComponents(binary, connectivity=4)
            self._num_labels_8, self._labels_8 = cv2.connectedComponents(binary, connectivity=8)
            print(f"Etiquetas Vecindad-4 encontradas: {self._num_labels_4}")
            print(f"Etiquetas Vecindad-8 encontradas: {self._num_labels_8}")


            # 4. Create image with contours and numbers
            # Start from the binary image and convert to BGR to draw color contours
            contoured = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)

            # Find contours
            # Use cv2.RETR_EXTERNAL to find only the outer contours of the objects
            # Use cv2.CHAIN_APPROX_SIMPLE to compress horizontal, vertical, and diagonal segments
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours and add numbers
            for i, contour in enumerate(contours):
                # Draw contour in green (0, 255, 0)
                cv2.drawContours(contoured, [contour], -1, (0, 255, 0), 2)
                # Find the bounding rectangle for the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Put text (object number) slightly above the bounding box
                cv2.putText(contoured, f'{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Store the contoured image
            self._contoured_image = contoured.copy()


            # Display the binary image as the initial view in the central panel
            # display_processed_image is called, which will apply display adjustments
            self.display_processed_image(self._binary_image, "Componentes Conexas (Imagen Binarizada)")

            self._cancel_threshold_worker() # Cancel any threshold worker if active
            self._is_noisy = False # This process reverts noise effect

            # --- NUEVO: Mostrar automáticamente la ventana binarizada después del análisis --- #
            # Esto proporciona feedback inmediato al usuario de que el análisis se completó
            try:
                self.show_binary_window() # Llamar al método para mostrar la ventana binarizada
            except Exception as e_show_bin:
                print(f"Error al intentar mostrar la ventana binarizada automáticamente: {e_show_bin}")
                # No mostramos un QMessageBox aquí, ya que show_binary_window ya lo hace si hay un error.
            # --- FIN NUEVO ---

        except cv2.error as e:
             QApplication.restoreOverrideCursor() # Restore cursor before showing message
             QMessageBox.critical(self,"Error de OpenCV", f"Error durante el análisis de Componentes Conexas: {e}")
             self.display_processed_image(None, f"Error Componentes Conexas")
             self._cancel_threshold_worker()
             self._is_noisy = False
        except Exception as e:
             QApplication.restoreOverrideCursor() # Restore cursor before showing message
             QMessageBox.critical(self,"Error General", f"Ocurrió un error inesperado durante el análisis de Componentes Conexas: {e}")
             print(f"Error during connected components analysis: {e}")
             traceback.print_exc() # Print full traceback for debugging
             self.display_processed_image(None, f"Error Componentes Conexas")
             self._cancel_threshold_worker()
             self._is_noisy = False
        finally:
            QApplication.restoreOverrideCursor() # Ensure cursor is restored


    # --- Methods to show the specific ImageView windows ---
    def show_binary_window(self):
        # --- MODIFICADO: Simplificado - Solo muestra si la imagen binaria existe ---
        if self._binary_image is None:
            QMessageBox.warning(self, "Advertencia", "Primero realiza el Etiquetado de Componentes Conexas para mostrar la imagen binarizada.")
            return
        # --- FIN MODIFICADO ---

        # Check if the window is already open and visible
        if not hasattr(self, '_binary_window') or self._binary_window is None or not self._binary_window.isVisible():
            try:
                print("Creating and showing Binary ImageViewWindow...")
                # No colorbar for binary image, use 'gray' cmap
                self._binary_window = ImageViewWindow(self._binary_image, "Imagen Binarizada", cmap='gray', parent=None)
                self._binary_window.setAttribute(Qt.WA_DeleteOnClose)
                self._binary_window.finished.connect(lambda: setattr(self, '_binary_window', None))
                self._binary_window.show()
                self._binary_window.raise_()
                self._binary_window.activateWindow()
                print("Binary ImageViewWindow should now be visible.")
                QMessageBox.information(self, "Ventana Creada", "La ventana de Imagen Binarizada se ha creado y está lista para visualizar.")
            except Exception as e:
                print(f"Error creating/showing Binary ImageViewWindow: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error al mostrar Ventana", f"Error al mostrar Imagen Binarizada: {e}")
        else:
            print("Binary ImageViewWindow already open. Activating...")
            self._binary_window.activateWindow()
            QMessageBox.information(self, "Ventana Activa", "La ventana de Imagen Binarizada ya está abierta y ha sido activada.")

    def show_vec4_window(self):
        print("Attempting to show Vecindad-4 ImageViewWindow...")
        if self._labels_4 is None or self._num_labels_4 <= 0:
            QMessageBox.warning(self, "Advertencia", "Primero realiza el Etiquetado de Componentes Conexas o no se encontraron objetos para mostrar Etiquetas Vecindad-4.")
            print("Vecindad-4 labels not available.")
            return

        if not hasattr(self, '_vec4_window') or self._vec4_window is None or not self._vec4_window.isVisible():
            try:
                print("Creating and showing Vecindad-4 ImageViewWindow...")
                self._vec4_window = ImageViewWindow(self._labels_4, "Etiquetas Vecindad-4",
                                                    cmap='jet', object_count=self._num_labels_4 - 1,
                                                    add_colorbar=True, parent=None)
                self._vec4_window.setAttribute(Qt.WA_DeleteOnClose)
                self._vec4_window.finished.connect(lambda: setattr(self, '_vec4_window', None))
                self._vec4_window.show()
                self._vec4_window.raise_()
                self._vec4_window.activateWindow()
                print("Vecindad-4 ImageViewWindow should now be visible.")
                QMessageBox.information(self, "Ventana Creada", "La ventana de Etiquetas Vecindad-4 se ha creado y está lista para visualizar.")
            except Exception as e:
                print(f"Error creating/showing Vecindad-4 ImageViewWindow: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error al mostrar Ventana", f"Error al mostrar Etiquetas Vecindad-4: {e}")
        else:
            print("Vecindad-4 ImageViewWindow already open. Activating...")
            self._vec4_window.activateWindow()
            QMessageBox.information(self, "Ventana Activa", "La ventana de Etiquetas Vecindad-4 ya está abierta y ha sido activada.")

    def show_vec8_window(self):
        print("Attempting to show Vecindad-8 ImageViewWindow...")
        if self._labels_8 is None or self._num_labels_8 <= 0:
            QMessageBox.warning(self, "Advertencia", "Primero realiza el Etiquetado de Componentes Conexas o no se encontraron objetos para mostrar Etiquetas Vecindad-8.")
            print("Vecindad-8 labels not available.")
            return

        if not hasattr(self, '_vec8_window') or self._vec8_window is None or not self._vec8_window.isVisible():
            try:
                print("Creating and showing Vecindad-8 ImageViewWindow...")
                self._vec8_window = ImageViewWindow(self._labels_8, "Etiquetas Vecindad-8",
                                                    cmap='jet', object_count=self._num_labels_8 - 1,
                                                    add_colorbar=True, parent=None)
                self._vec8_window.setAttribute(Qt.WA_DeleteOnClose)
                self._vec8_window.finished.connect(lambda: setattr(self, '_vec8_window', None))
                self._vec8_window.show()
                self._vec8_window.raise_()
                self._vec8_window.activateWindow()
                print("Vecindad-8 ImageViewWindow should now be visible.")
                QMessageBox.information(self, "Ventana Creada", "La ventana de Etiquetas Vecindad-8 se ha creado y está lista para visualizar.")
            except Exception as e:
                print(f"Error creating/showing Vecindad-8 ImageViewWindow: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error al mostrar Ventana", f"Error al mostrar Etiquetas Vecindad-8: {e}")
        else:
            print("Vecindad-8 ImageViewWindow already open. Activating...")
            self._vec8_window.activateWindow()
            QMessageBox.information(self, "Ventana Activa", "La ventana de Etiquetas Vecindad-8 ya está abierta y ha sido activada.")

    def show_contours_window(self):
        print("Attempting to show Contours ImageViewWindow...")
        if self._contoured_image is None:
            QMessageBox.warning(self, "Advertencia", "Primero realiza el Etiquetado de Componentes Conexas para mostrar Contornos y Numeración.")
            print("Contoured image not available.")
            return

        if not hasattr(self, '_contours_window') or self._contours_window is None or not self._contours_window.isVisible():
            try:
                print("Creating and showing Contours ImageViewWindow...")
                img_rgb = cv2.cvtColor(self._contoured_image.copy(), cv2.COLOR_BGR2RGB)
                contour_count = 0
                if self._binary_image is not None:
                    try:
                        binary_gray_for_contours = self._binary_image.copy()
                        if binary_gray_for_contours.ndim == 3:
                            binary_gray_for_contours = cv2.cvtColor(binary_gray_for_contours, cv2.COLOR_BGR2GRAY)
                        contours, _ = cv2.findContours(binary_gray_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contour_count = len(contours)
                        print(f"Recalculated contour count: {contour_count}")
                    except Exception as e_contours:
                        print(f"Error recalculando contornos para conteo: {e_contours}")
                        traceback.print_exc()
                        contour_count = None

                self._contours_window = ImageViewWindow(img_rgb, "Contornos y Numeración",
                                                        object_count=contour_count, parent=None)
                self._contours_window.setAttribute(Qt.WA_DeleteOnClose)
                self._contours_window.finished.connect(lambda: setattr(self, '_contours_window', None))
                self._contours_window.show()
                self._contours_window.raise_()
                self._contours_window.activateWindow()
                print("Contours ImageViewWindow should now be visible.")
                QMessageBox.information(self, "Ventana Creada", "La ventana de Contornos y Numeración se ha creado y está lista para visualizar.")
            except Exception as e:
                print(f"Error creating/showing Contours ImageViewWindow: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error al mostrar Ventana", f"Error al mostrar Contornos y Numeración: {e}")
        else:
            print("Contours ImageViewWindow already open. Activating...")
            self._contours_window.activateWindow()
            QMessageBox.information(self, "Ventana Activa", "La ventana de Contornos y Numeración ya está abierta y ha sido activada.")

    # --- END NEW Methods for Connected Components Analysis and Display ---

    def toggle_robinson_directions(self):
        """Alterna la visibilidad del grupo de botones direccionales de Robinson"""
        self.robinson_directions_group.setVisible(not self.robinson_directions_group.isVisible())

    def show_segmentation_window(self):
        try:
            if not SEGMENTATION_AVAILABLE:
                QMessageBox.warning(self, "Error", 
                    "No se pudo cargar el módulo de segmentación.\n"
                    "Asegúrate de que YolovDetectron.py está en el mismo directorio.")
                return

            # Si la ventana ya existe y está visible, traerla al frente
            if hasattr(self, 'segmentation_window') and self.segmentation_window is not None:
                if self.segmentation_window.isVisible():
                    self.segmentation_window.activateWindow()
                    self.segmentation_window.raise_()
                    return
                else:
                    # Si existe pero está oculta, eliminarla
                    self.segmentation_window.close()
                    self.segmentation_window = None

            # Crear nueva instancia de la ventana de segmentación
            self.segmentation_window = SegmentationProcessor()
            
            # Conectar la señal de cierre de la ventana
            self.segmentation_window.closeEvent = lambda event: self._clear_segmentation_window_reference(event)
            
            # Mostrar la ventana
            self.segmentation_window.show()
            
        except Exception as e:
            error_msg = f"Error al abrir la ventana de segmentación:\n{str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", error_msg)

    def _clear_segmentation_window_reference(self, event=None):
        """Limpia la referencia a la ventana de segmentación cuando se cierra"""
        if hasattr(self, 'segmentation_window'):
            self.segmentation_window = None
        if event:
            event.accept()


# --- NEW: Class for the Image View Window (Moved outside ImageProcessor) ---
class ImageViewWindow(QDialog): # Change base class from QWidget to QDialog
    # Add parameters for cmap, object_count, and add_colorbar
    def __init__(self, image_data, title, cmap=None, object_count=None, add_colorbar=False, parent=None):
        super().__init__(parent) # Pass parent
        self.setWindowTitle(title)
        # Adjust window size slightly to accommodate colorbar if needed
        self.setGeometry(200, 200, 650, 550)

        layout = QVBoxLayout(self)

        # Adjust figure size to leave space for colorbar if needed
        # Figure size in inches, figsize=(width, height)
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        # Create a single sub-plot (axes)
        # Adjust subplot position to make space for colorbar on the right if added
        # [left, bottom, width, height] - normalized coordinates
        self.ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8]) # Default position, adjust if colorbar added

        layout.addWidget(self.canvas)

        # Show the image and save the object returned by imshow
        # Use aspect='equal' to maintain aspect ratio
        im = self.ax.imshow(image_data, cmap=cmap, aspect='equal')

        # Add colorbar if requested and if a colormap was used
        if add_colorbar and cmap is not None:
             # Create and add the colorbar linked to the 'im' object (the result of imshow)
             # Make space for the colorbar
             self.figure.subplots_adjust(right=0.85) # Adjust subplot to make space
             # Add the colorbar to the figure
             cbar_ax = self.figure.add_axes([0.9, 0.1, 0.05, 0.8]) # Position for colorbar [left, bottom, width, height]
             self.figure.colorbar(im, cax=cbar_ax)


        # Add title to the plot
        plot_title = title
        # Only show object count if greater than 0 (excluding background label)
        if object_count is not None and object_count > 0:
             plot_title += f" (Objetos Detectados: {object_count})"

        self.ax.set_title(plot_title)
        self.ax.axis('off') # Hide axes

        self.canvas.draw() # Draw on the canvas

        self.setLayout(layout)
# --- FIN NEW Class for the Image View Window ---


# Main application entry point
if __name__ == "__main__":
    # Settings for high pixel density screens (recommended for PyQt)
    # This helps GUI elements appear correctly on high-resolution displays
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create the application instance
    app = QApplication(sys.argv)
    # Create the instance of the ImageProcessor class
    window = ImageProcessor()
    # Show the window
    window.show()
    # Start the main application event loop
    sys.exit(app.exec_())
