"""Regression: Wiggle reloads must cover the current data and time span.

Run: python scripts/tests/smoke_test_wiggle_reload.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PyQt5.QtWidgets import QApplication
from scripts.core.models import FileMetadata
from scripts.render.space_time_view import SpaceTimeView


def main() -> None:
    app = QApplication.instance() or QApplication([])
    view = SpaceTimeView()
    view.set_plot_mode("多道波形(Wiggle)")
    view.set_wiggle_params({"fill": True, "spacing_mode": "道数", "spacing_value": 1})
    chunks = [np.full((3001, 4), value, dtype=np.float32) for value in (0.1, 0.4, -0.8)]

    def check(data: np.ndarray) -> None:
        metadata = FileMetadata(Path("synthetic.bin"), 1000.0, 4, len(data), len(data) / 1000.0, data.nbytes)
        view.set_data(metadata, data, 0, 3, 0)
        app.processEvents()
        expected_end = (len(data) - 1) / metadata.sample_rate_hz
        assert view._wiggle_items
        for index, item in enumerate(view._wiggle_items):
            x, y = item.getData()
            assert np.isclose(x[-1], expected_end), (x[-1], expected_end)
            cached_x, envelope, baseline = view._wiggle_display_cache[index]
            assert np.isclose(envelope[-1], data[-1, index])
            assert np.isclose(view._wiggle_fills[index].getData()[0][-1], expected_end)
            assert np.isclose(view._wiggle_baselines[index].getData()[0][-1], metadata.duration_s)
        assert np.isclose(view.current_ranges()[0][1], metadata.duration_s)
        assert view._w_stats["matrix"].shape == data.shape

    check(chunks[0])
    check(np.concatenate(chunks))
    check(chunks[1])
    # Replacing data with the same shape must also invalidate amplitudes.
    check(chunks[2])
    items = list(view._wiggle_items)
    view.set_wiggle_params({"gain_db": 6.0})
    view._refresh_wiggle()
    assert view._wiggle_items == items, "Gain updates should retain the fast in-place path"
    # Repeated mode switches must restore reused traces, fills and baselines.
    # Exercise both filled and unfilled profiles, including the deferred refresh.
    for fill in (True, False):
        view.set_wiggle_params({"fill": fill})
        view._refresh_wiggle()
        for _ in range(3):
            items = view._wiggle_items + view._wiggle_fills + view._wiggle_baselines
            assert len(view._wiggle_items) == 4
            assert len(view._wiggle_fills) == (4 if fill else 0)
            before = [item.getData()[1].copy() for item in items]
            view.set_plot_mode("时空颜色图")
            assert view.image_item.isVisible()
            assert all(not item.isVisible() for item in items)
            view.set_plot_mode("多道波形(Wiggle)")
            view.set_wiggle_params({"gain_db": 6.0})
            view._wiggle_timer.stop()
            view._refresh_wiggle()
            app.processEvents()
            assert not view.image_item.isVisible()
            assert items == view._wiggle_items + view._wiggle_fills + view._wiggle_baselines
            assert all(item.isVisible() for item in items)
            for item, expected in zip(items, before):
                np.testing.assert_allclose(item.getData()[1], expected)
    view.close()
    print("PASS: single/multiple/single reload, same-shape replacement, gain update, mode switch")


if __name__ == "__main__":
    main()
