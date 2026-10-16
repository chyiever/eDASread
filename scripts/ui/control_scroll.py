"""Keep large-font parameter strips compact and reachable on narrow windows."""
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QScrollArea, QSizePolicy, QWidget, QLayout


class ControlScrollArea(QScrollArea):
    def __init__(self, content, parent=None):
        super().__init__(parent)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        self.setWidget(content)

    def refresh_geometry(self):
        content = self.widget()
        if content.layout():
            for widget in reversed(content.findChildren(QWidget)):
                widget.ensurePolished()
                if widget.layout():
                    widget.layout().setSizeConstraint(QLayout.SetMinimumSize)
                    widget.layout().invalidate()
                    widget.layout().activate()
            content.layout().invalidate()
            content.layout().activate()
            minimum = content.layout().minimumSize()
            height = content.layout().sizeHint().height()
            content.setMinimumWidth(minimum.width())
            content.setFixedHeight(height)
            overflow = minimum.width() > self.viewport().width()
            self.setFixedHeight(height + (self.horizontalScrollBar().sizeHint().height() if overflow else 0))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refresh_geometry)
