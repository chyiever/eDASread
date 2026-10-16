"""User-facing appearance settings."""
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QSpinBox, QVBoxLayout


class SettingsDialog(QDialog):
    def __init__(self, parent, font_size: int, apply_settings):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(12, 28)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.setValue(font_size)
        self.font_size_spin.setObjectName("interfaceFontSize")
        form = QFormLayout()
        form.addRow("界面文字大小", self.font_size_spin)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("默认 18px，可设置 12～28px。\n应用后立即生效并保存，下次启动自动恢复。\n参数区过宽时可横向滚动查看。"))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        for role, text in ((QDialogButtonBox.Ok,"确定"),(QDialogButtonBox.Apply,"应用"),(QDialogButtonBox.Cancel,"取消"),(QDialogButtonBox.RestoreDefaults,"恢复默认")):
            buttons.button(role).setText(text)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(lambda: apply_settings(self.font_size_spin.value()))
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(lambda: self.font_size_spin.setValue(18))
        buttons.accepted.connect(lambda: (apply_settings(self.font_size_spin.value()), self.accept()))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
