"""File-browser widgets for paginated bin file selection."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QAbstractItemView,
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
        self.compress_bz_button = QPushButton("Batch Bin -> bz")
        self.compress_bz_button.setFont(ui_font)
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setToolTip("重新扫描当前目录，更新新增或删除的文件（F5）")
        self.refresh_button.setShortcut("F5")
        self.file_list_status = QLabel("")
        self.sort_combo = QComboBox()
        self.sort_combo.setFont(ui_font)
        self.sort_combo.addItem("Name", "name_asc")
        self.sort_combo.addItem("Modified: Newest", "mtime_desc")
        self.sort_combo.addItem("Modified: Oldest", "mtime_asc")
        self.file_list = QListWidget()
        self.file_list.setFont(ui_font)
        self.file_list.setUniformItemSizes(True)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.previous_button = QPushButton("上一页")
        self.next_button = QPushButton("下一页")
        self.page_jump_edit = QLineEdit()
        self.page_jump_edit.setFont(ui_font)
        self.page_jump_edit.setPlaceholderText("page / home / end")
        self.jump_button = QPushButton("跳转")
        self.jump_button.setFont(ui_font)
        self.page_label = QLabel("Page 0 / 0")
        for widget in [self.previous_button, self.next_button, self.page_label]:
            widget.setFont(ui_font)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_button)
        self.browse_button.setText("目录")
        self.compress_bz_button.setText("批量压缩")
        action_row = QHBoxLayout()
        action_row.addWidget(self.refresh_button)
        action_row.addWidget(self.compress_bz_button)

        sort_row = QHBoxLayout()
        sort_label = QLabel("Sort")
        sort_label.setFont(ui_font)
        sort_row.addWidget(sort_label)
        sort_row.addWidget(self.sort_combo, 1)

        paging_row = QHBoxLayout()
        paging_row.addWidget(self.previous_button)
        paging_row.addWidget(self.next_button)
        paging_row.addWidget(self.page_jump_edit, 1)
        paging_row.addWidget(self.jump_button)

        layout = QVBoxLayout(self)
        layout.addLayout(path_row)
        layout.addLayout(action_row)
        layout.addLayout(sort_row)
        layout.addWidget(self.file_list_status)
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
