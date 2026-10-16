"""Compact eDASpy instrument-workstation theme and matching vector tab icons."""
from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget, QLineEdit, QComboBox
from scripts.ui.control_scroll import ControlScrollArea


STYLE = """
QMainWindow, QWidget#workspace { background: #edf2f8; }
QWidget { color: #21384f; font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 14px; }
QWidget#appHeader {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #f5fbff,stop:0.5 #ffffff,stop:1 #edf8fb);
    border: 1px solid #ccdee8; border-radius: 8px;
}
QLabel#appTitle { background: transparent; color: #24536b; font-size: 24px; font-weight: 600; }
QLabel#appSubtitle { background: transparent; color: #9cbfd5; font-size: 11px; }
QWidget#filePanel { background: #f8fbff; border-radius: 8px; }
QWidget#controlBar, QGroupBox {
    background: #f8fbff; border: 1px solid #d4e1ec; border-radius: 6px;
}
QGroupBox { margin-top: 10px; padding-top: 5px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #536c85; }
QWidget#wiggleBar { background: #eff6fb; border: 1px solid #d4e1ec; border-radius: 6px; }
QLineEdit, QComboBox {
    background: #ffffff; color: #19354f; border: 1px solid #bdccdc; border-radius: 4px;
    padding: 2px 5px; min-height: 20px; selection-background-color: #bce8f1;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #0998b2; background: #f5fdff; }
QComboBox { padding-right: 17px; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView { background: #ffffff; color: #21384f; selection-background-color: #d9eff5; }
QPushButton, QToolButton {
    background: #ffffff; color: #284d6c; border: 1px solid #bccddd; border-radius: 4px;
    padding: 3px 7px; min-height: 20px; font-weight: 500;
}
QPushButton:hover, QToolButton:hover { background: #e9f6fa; border-color: #51a9bc; }
QPushButton:pressed, QToolButton:pressed { background: #d1eaf2; }
QPushButton[primary="true"] { background: #143d59; color: #f4fcff; border-color: #167c96; }
QPushButton[primary="true"]:hover { background: #15536c; border-color: #19bbd1; }
QPushButton:disabled, QToolButton:disabled { background: #e8eef4; color: #91a0b0; border-color: #d3dce6; }
QToolButton::menu-indicator { subcontrol-position: right center; right: 3px; }
QCheckBox { spacing: 4px; background: transparent; }
QCheckBox::indicator { width: 13px; height: 13px; }
QTabWidget::pane { background: #ffffff; border: 1px solid #d0dce8; border-radius: 6px; top: -1px; }
QTabBar::tab {
    background: #e5edf5; color: #4b6379; border: 1px solid #d0dce8;
    border-bottom: 2px solid transparent; padding: 7px 15px; min-height: 21px; margin-right: 3px;
    border-top-left-radius: 5px; border-top-right-radius: 5px; font-weight: 500;
}
QTabBar::tab:selected { background: #ffffff; color: #123e59; border-bottom: 2px solid #0ca7bd; }
QTabBar::tab:hover:!selected { background: #d9eaf3; }
QListWidget { background: #ffffff; border: 1px solid #d6e1ed; border-radius: 5px; outline: 0; }
QListWidget::item { padding: 4px 3px; }
QListWidget::item:selected { background: #d9edf5; color: #163d57; border-left: 3px solid #0c9db5; }
QMenu { background: #f8fbff; color: #21384f; border: 1px solid #c4d5e5; padding: 8px; }
QToolTip { background: #17354e; color: #f2f9ff; border: 1px solid #2b7189; padding: 6px; }
QStatusBar { background: #e4edf5; color: #4f6a81; border-top: 1px solid #cbdbe8; }
QSplitter::handle { background: #d4e0eb; }
QScrollBar:horizontal { height: 10px; background: #ecf2f7; }
QScrollBar:vertical { width: 10px; background: #ecf2f7; }
QScrollBar::handle { background: #afc4d5; border-radius: 4px; min-width: 24px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QSlider::groove:horizontal { height: 3px; background: #cbdce8; border-radius: 1px; }
QSlider::handle:horizontal { background: #087e99; border: 1px solid #07677d; width: 9px; margin: -5px 0; border-radius: 3px; }
"""


def tab_icon(kind: str) -> QIcon:
    """Five glyphs share a 24px grid, stroke weight and restrained teal hue."""
    pixmap = QPixmap(48, 48)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#1289a4"), 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    if kind == "processing":
        for x, y in ((5,8),(12,16),(19,10)):
            painter.drawLine(QPointF(x,4),QPointF(x,20))
            painter.setBrush(QColor("#f6fbff")); painter.drawEllipse(QPointF(x,y),2.2,2.2)
    elif kind == "waveform":
        painter.drawPolyline(QPolygonF([QPointF(2,12),QPointF(6,12),QPointF(9,5),QPointF(12,19),QPointF(15,9),QPointF(18,12),QPointF(22,12)]))
    elif kind == "fk":
        painter.drawLine(4,20,4,4);painter.drawLine(4,20,21,20)
        for start,end in ((7,17),(11,13),(15,9)):
            painter.drawLine(start,18,20,end-8)
    elif kind == "spectrogram":
        for x, top, bottom in ((4,10,18),(8,5,15),(12,8,20),(16,3,13),(20,7,17)):
            painter.drawLine(x,top,x,bottom)
    elif kind == "traffic":
        painter.drawLine(4,4,4,21);painter.drawLine(20,4,20,21)
        painter.drawPolyline(QPolygonF([QPointF(7,19),QPointF(10,14),QPointF(15,11),QPointF(17,5)]))
        painter.setBrush(QColor("#1289a4"))
        painter.drawEllipse(QPointF(10,14),1.8,1.8)
    else:
        painter.end()
        raise ValueError(kind)
    painter.end()
    return QIcon(pixmap)


def apply_theme(window: QWidget, font_size: int = 18) -> None:
    font = QFont("Microsoft YaHei UI")
    font.setPixelSize(font_size)
    window.setFont(font)
    # Older panels set Times New Roman/SimSun locally; normalize the GUI type
    # while plot axes use the separate shared scientific-plot style.
    for widget in window.findChildren(QWidget):
        widget.setFont(font)
        if isinstance(widget, (QLineEdit, QComboBox, QPushButton)):
            if not hasattr(widget, "_theme_original_width"):
                widget._theme_original_width = widget.minimumWidth() if widget.minimumWidth() == widget.maximumWidth() else None
                widget._theme_original_height = widget.minimumHeight() if widget.minimumHeight() == widget.maximumHeight() else None
            if widget._theme_original_width is not None:
                widget.setFixedWidth(round(widget._theme_original_width * font_size / 14))
            if widget._theme_original_height is not None:
                widget.setFixedHeight(max(font_size + 10, round(widget._theme_original_height * font_size / 14)))
            else:
                widget.setMaximumHeight(16777215)
                widget.setMinimumHeight(font_size + 8)
    title_size = max(24, font_size + 6)
    window.setStyleSheet(STYLE.replace("font-size: 14px", f"font-size: {font_size}px").replace("font-size: 24px", f"font-size: {title_size}px"))
    window.title_label.parentWidget().setFixedHeight(max(56, title_size + 24))
    import pyqtgraph as pg
    for plot in window.findChildren(pg.PlotWidget):
        for entry in plot.getPlotItem().axes.values():
            axis = entry["item"]
            axis.setStyle(tickFont=font)
            axis.label.setFont(font)
            axis.setLabel(axis.labelText, units=axis.labelUnits,
                          unitPrefix=axis.labelUnitPrefix,
                          **dict(axis.labelStyle, **{"font-size": f"{font_size}px"}))
        title = plot.getPlotItem().titleLabel
        title.setText(title.text, size=f"{font_size}px")
    for histogram in window.findChildren(pg.HistogramLUTWidget):
        axis = histogram.item.axis
        axis.setStyle(tickFont=font)
        axis.label.setFont(font)
        axis.setLabel(axis.labelText, units=axis.labelUnits,
                      unitPrefix=axis.labelUnitPrefix,
                      **dict(axis.labelStyle, **{"font-size": f"{font_size}px"}))
    for scroll in window.findChildren(ControlScrollArea):
        scroll.refresh_geometry()
