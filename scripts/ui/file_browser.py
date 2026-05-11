"""File-browser widgets for paginated bin file selection."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)


class FileBrowserPanel(QWidget):
    """Directory chooser plus paginated file list."""

    PAGE_SIZE = 100

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ui_font = QFont("Times New Roman", 10)

        self.path_edit = QLineEdit()
        self.path_edit.setFont(ui_font)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setFont(ui_font)
        self.file_list = QListWidget()
        self.file_list.setFont(ui_font)
        self.previous_button = QPushButton("Previous 100")
        self.next_button = QPushButton("Next 100")
        self.page_jump_edit = QLineEdit()
        self.page_jump_edit.setFont(ui_font)
        self.page_jump_edit.setPlaceholderText("page / home / end")
        self.jump_button = QPushButton("Jump")
        self.jump_button.setFont(ui_font)
        self.page_label = QLabel("Page 0 / 0")
        for widget in [self.previous_button, self.next_button, self.page_label]:
            widget.setFont(ui_font)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_button)

        paging_row = QHBoxLayout()
        paging_row.addWidget(self.previous_button)
        paging_row.addWidget(self.next_button)
        paging_row.addWidget(self.page_jump_edit, 1)
        paging_row.addWidget(self.jump_button)

        layout = QVBoxLayout(self)
        layout.addLayout(path_row)
        layout.addWidget(self.file_list, 1)
        layout.addLayout(paging_row)
        layout.addWidget(self.page_label)

        self.browse_button.clicked.connect(self._choose_directory)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            self.path_edit.text().strip() or str(Path.cwd()),
        )
        if directory:
            self.path_edit.setText(directory)
