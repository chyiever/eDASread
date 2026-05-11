"""Background worker wrappers for long-running tasks."""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


class WorkerSignals(QObject):
    """Qt signals emitted by background workers."""

    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)


class FunctionWorker(QRunnable):
    """Execute one Python callable on a thread-pool worker thread."""

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
