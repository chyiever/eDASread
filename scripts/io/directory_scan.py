"""Cancelable directory snapshots; no data headers are read while browsing."""
import os
from pathlib import Path
from scripts.io.bin_reader import DATA_SUFFIXES


def scan_directory(directory, mode, cancelled):
    records = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if cancelled.is_set():
                return None
            if os.path.splitext(entry.name)[1].lower() not in DATA_SUFFIXES:
                continue
            try:
                if not entry.is_file():
                    continue
                modified = entry.stat().st_mtime_ns if mode != "name_asc" else 0
            except FileNotFoundError:
                continue  # File removed while a live acquisition directory is scanned.
            records.append((entry.name, modified))
    if cancelled.is_set():
        return None
    if mode == "mtime_desc":
        records.sort(key=lambda row: (-row[1], row[0].casefold(), row[0]))
    elif mode == "mtime_asc":
        records.sort(key=lambda row: (row[1], row[0].casefold(), row[0]))
    else:
        records.sort(key=lambda row: (row[0].casefold(), row[0]))
    folder = Path(directory)
    return [folder / name for name, _ in records] if not cancelled.is_set() else None
