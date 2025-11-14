# /// script
# requires-python = ">=3.10"
# dependencies = ["PyQt6"]
# ///
"""
yolo_cls_edit.py - YOLO classification dataset editor

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime
from PyQt6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QUrl, QStringListModel
from PyQt6.QtGui import QAction, QDesktopServices, QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(slots=True)
class ImageRecord:
    source_path: Path
    original_class: str
    assigned_class: Optional[str] = None

    def current_class(self) -> str:
        return self.assigned_class or self.original_class


class ClassAssignDialog(QDialog):
    """Collect a class display name with autocompletion."""

    def __init__(self, suggestions: List[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign Class")
        self.setModal(True)

        self.base_suggestions: List[str] = suggestions
        self.model = QStringListModel(self.base_suggestions)
        self.completer = QCompleter(self.model, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._suppress_updates = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Type or choose a class display name:"))

        self.input = QLineEdit()
        self.input.setCompleter(self.completer)
        self.input.installEventFilter(self)
        self.input.textChanged.connect(self._on_text_changed)
        self.input.returnPressed.connect(self.accept)
        if self.base_suggestions:
            self.input.setText(self.base_suggestions[0])
            self.input.selectAll()
        layout.addWidget(self.input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_text_changed(self.input.text())

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if obj is self.input and isinstance(event, QKeyEvent):
            if event.type() == QEvent.Type.ShortcutOverride and event.key() in (
                Qt.Key.Key_Tab,
                # Qt.Key.Key_Backtab,
            ):
                event.accept()
                return True

            if event.type() == QEvent.Type.KeyRelease and event.key() in (
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
            ):
                self._suppress_updates = True
                try:
                    idx = self.base_suggestions.index(self.input.text().strip())
                    if event.key() == Qt.Key.Key_Up and idx < len(
                        self.base_suggestions
                    ):
                        idx = idx + 1
                    elif event.key() == Qt.Key.Key_Down and idx > 0:
                        idx = idx - 1

                    self.input.setText(self.base_suggestions[idx])
                except (ValueError, IndexError):
                    if event.key() == Qt.Key.Key_Up:
                        self.input.setText(self.base_suggestions[0])  # last item
                    else:
                        self.input.setText("")
                finally:
                    self._suppress_updates = False

            if event.type() == QEvent.Type.KeyRelease and event.key() in (
                Qt.Key.Key_Tab,
                # Qt.Key.Key_Backtab,
            ):
                direction = (
                    -1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                )
                if self._cycle_completion(direction):
                    event.accept()
                    return True

        return super().eventFilter(obj, event)

    def _on_text_changed(self, text: str) -> None:
        if self._suppress_updates:
            return

        if self.base_suggestions:
            if self.model.stringList() != self.base_suggestions:
                self.model.setStringList(self.base_suggestions)
            if self.completer.currentRow() == -1:
                self.completer.setCurrentRow(0)
            self.completer.complete()
            popup = self.completer.popup()
            model = self.completer.completionModel()
            if popup is not None and model is not None:
                first_index = model.index(0, 0)
                if first_index.isValid():
                    popup.setCurrentIndex(first_index)

    def _cycle_completion(self, step: int) -> bool:
        if self.model.rowCount() == 0:
            return False

        current_index = self.completer.currentIndex()
        row = current_index.row()
        n_items = self.completer.completionCount()
        if row < 0:
            row = 0 if step >= 0 else n_items - 1
        else:
            row = (row + step) % n_items

        self.completer.setCurrentRow(row)
        completion = self.completer.currentCompletion()
        if completion:
            self._suppress_updates = True
            try:
                self.input.setText(completion)
                self.input.selectAll()
            finally:
                self._suppress_updates = False

        self.completer.complete()
        popup = self.completer.popup()
        model = self.completer.completionModel()
        if popup is not None and model is not None:
            model_index = model.index(row, 0)
            if model_index.isValid():
                popup.setCurrentIndex(model_index)
        return True

    def selected_text(self) -> str:
        return self.input.text().strip()


class MainWindow(QMainWindow):
    def __init__(self, class_mapping: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Class Editor")
        self.resize(1280, 840)

        self.records: List[ImageRecord] = []
        self.class_codes: List[str] = []
        self.class_shortcuts: Dict[int, str] = {}
        self.recent_class_codes: List[str] = []
        self.code_to_display: Dict[str, str] = dict(class_mapping or {})
        self.display_to_code: Dict[str, str] = {}
        self.display_lower_to_code: Dict[str, str] = {}

        if self.code_to_display:
            self._ensure_mapping_for_codes(list(self.code_to_display.keys()))
            self.class_codes = sorted(
                self.code_to_display.keys(),
                key=lambda code: self.code_to_display[code].lower(),
            )

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setSpacing(10)
        self.gallery.setSelectionRectVisible(True)
        self.gallery.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self.gallery.setMovement(QListWidget.Movement.Static)
        self.gallery.setDragEnabled(False)
        self.gallery.setUniformItemSizes(True)
        self.gallery.setIconSize(QSize(180, 180))
        self.gallery.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.gallery.customContextMenuRequested.connect(self._show_gallery_menu)
        self.gallery.itemDoubleClicked.connect(self._open_image_item)
        self.gallery.setStyleSheet(
            """
            QListWidget::item {
                border: 1px solid #6c6c6c;
                border-radius: 4px;
                margin: 4px;
                padding: 6px;
                background: palette(base);
            }
            QListWidget::item:selected {
                border: 2px solid #377ef3;
                background: palette(highlight);
                color: palette(highlighted-text);
            }
            """
        )

        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.class_list.itemDoubleClicked.connect(self._assign_from_class_list)

        class_panel = QWidget()
        class_layout = QVBoxLayout(class_panel)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.addWidget(QLabel("All classes"))
        class_layout.addWidget(self.class_list)

        self.help_box = QTextEdit()
        self.help_box.setReadOnly(True)
        self.help_box.setMinimumHeight(160)
        self.help_box.setHtml(
            """
            <b>Usage tips</b><br/>
            • Use <i>File → Open folder</i> to select a YOLO classification dataset.<br/>
            • Drag-select or Shift-click to select multiple images at any time.<br/>
            • Resize thumbnails with the toolbar slider to suit your screen.<br/>
            • Right-click the gallery, press number keys, or double-click a class to reassign.<br/>
            • Double-click an image to open it in your default viewer.<br/>
            • Press <b>Enter</b> on a selection to type a class name with autocompletion.<br/>
            • Use <b>Tab</b> in the prompt to cycle through matching classes.<br/>
            • Save with <i>File → Save result</i> to create a new dataset folder.
            """
        )
        class_layout.addWidget(self.help_box)

        splitter = QSplitter()
        splitter.addWidget(class_panel)
        splitter.addWidget(self.gallery)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.addWidget(splitter)
        self.setCentralWidget(container)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._create_actions()
        self._create_toolbar()
        self._rebuild_display_lookup()
        if self.class_codes:
            self._populate_class_list()
            self._update_filter_class_options()

    def _create_actions(self) -> None:
        self.open_action = QAction("Open folder", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._on_open_folder)

        self.save_action = QAction("Save result", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._on_save_result)
        self.save_action.setEnabled(False)

        menu_bar = self.menuBar()
        if menu_bar:
            file_menu = menu_bar.addMenu("File")
            assert file_menu is not None
            file_menu.addAction(self.open_action)
            file_menu.addSeparator()
            file_menu.addAction(self.save_action)
            file_menu.addSeparator()
            exit_action = QAction("Exit", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)

        size_container = QWidget()
        size_layout = QHBoxLayout(size_container)
        size_layout.setContentsMargins(4, 0, 4, 0)
        size_layout.setSpacing(6)
        size_layout.addWidget(QLabel("Thumb"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(80, 320)
        self.size_slider.setSingleStep(10)
        self.size_slider.setValue(180)
        self.size_slider.valueChanged.connect(self._update_icon_size)
        size_layout.addWidget(self.size_slider)
        size_action = QWidgetAction(self)
        size_action.setDefaultWidget(size_container)
        toolbar.addAction(size_action)

        filter_container = QWidget()
        filter_layout = QHBoxLayout(filter_container)
        filter_layout.setContentsMargins(4, 0, 4, 0)
        filter_layout.setSpacing(6)
        filter_layout.addWidget(QLabel("Filter"))

        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem("All images", None)
        self.filter_mode_combo.addItem("Original class", "original")
        self.filter_mode_combo.addItem("New class", "new")
        self.filter_mode_combo.addItem("Current class", "current")
        self.filter_mode_combo.currentIndexChanged.connect(self._on_filter_mode_changed)
        filter_layout.addWidget(self.filter_mode_combo)

        self.filter_class_combo = QComboBox()
        self.filter_class_combo.setEnabled(False)
        self.filter_class_combo.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_class_combo)

        filter_action = QWidgetAction(self)
        filter_action.setDefaultWidget(filter_container)
        toolbar.addAction(filter_action)

        self.addToolBar(toolbar)
        self._update_icon_size(self.size_slider.value())

    def _on_open_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select YOLO classification folder"
        )
        if directory:
            self.open_dataset(Path(directory))

    def open_dataset(self, root: Path) -> None:
        self._load_dataset(root)

    def _load_dataset(self, root: Path) -> None:
        if not root.exists() or not root.is_dir():
            QMessageBox.warning(
                self, "Invalid folder", "Please select a valid directory."
            )
            return

        records: List[ImageRecord] = []
        class_codes: List[str] = []

        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            class_codes.append(class_dir.name)
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(ImageRecord(image_path, class_dir.name))

        if not records:
            QMessageBox.information(
                self,
                "No images found",
                "The selected folder does not contain images in YOLO classification format.",
            )
            return

        self.records = records
        self.recent_class_codes.clear()
        unique_codes = sorted(set(class_codes))
        self._ensure_mapping_for_codes(unique_codes)
        combined_codes = set(unique_codes) | set(self.code_to_display.keys())
        self.class_codes = sorted(
            combined_codes, key=lambda code: self._display_name(code).lower()
        )
        self.save_action.setEnabled(True)

        self._populate_class_list()
        self._update_filter_class_options()
        self._apply_filter()
        self.status_bar.showMessage(
            f"Loaded {len(records)} images across {len(self.class_codes)} classes."
        )

    def _ensure_mapping_for_codes(self, codes: List[str]) -> None:
        for code in codes:
            self.code_to_display.setdefault(code, code)
        self._rebuild_display_lookup()

    def _rebuild_display_lookup(self) -> None:
        self.display_to_code.clear()
        self.display_lower_to_code.clear()
        for code, display in self.code_to_display.items():
            self.display_to_code[display] = code
            self.display_lower_to_code[display.lower()] = code

    def _ordered_class_codes(self) -> List[str]:
        ordered = [
            code for code in self.recent_class_codes if code in self.code_to_display
        ]
        for code in self.class_codes:
            if code not in ordered:
                ordered.append(code)
        return ordered

    def _class_display_sequence(self) -> List[Tuple[str, str]]:
        return [
            (code, self._display_name(code)) for code in self._ordered_class_codes()
        ]

    def _display_name(self, class_code: str) -> str:
        return self.code_to_display.get(class_code, class_code)

    def _populate_class_list(self) -> None:
        self.class_list.clear()
        for code, display in self._class_display_sequence():
            item = QListWidgetItem(
                display if code not in self.recent_class_codes else f"★ {display}"
            )
            item.setData(Qt.ItemDataRole.UserRole, code)
            item.setToolTip(f"{display} ({code})")
            self.class_list.addItem(item)
        self._refresh_shortcuts()

    def _refresh_shortcuts(self) -> None:
        self.class_shortcuts.clear()
        for idx, class_code in enumerate(self._ordered_class_codes()[:9], start=1):
            self.class_shortcuts[idx] = class_code
        if self.class_shortcuts:
            hint = ", ".join(
                f"{digit}→{self._display_name(code)}"
                for digit, code in self.class_shortcuts.items()
            )
            self.status_bar.showMessage(
                f"Shortcut mapping: {hint} (press number keys while images are selected)",
                5000,
            )

    def _update_filter_class_options(self) -> None:
        previous_code = self.filter_class_combo.currentData()
        self.filter_class_combo.blockSignals(True)
        self.filter_class_combo.clear()
        self.filter_class_combo.addItem("All classes", None)
        for code, display in self._class_display_sequence():
            self.filter_class_combo.addItem(f"{display} ({code})", code)
        if previous_code:
            for index in range(self.filter_class_combo.count()):
                if self.filter_class_combo.itemData(index) == previous_code:
                    self.filter_class_combo.setCurrentIndex(index)
                    break
        self.filter_class_combo.blockSignals(False)

    def _apply_filter(self) -> None:
        self._populate_gallery()

    def _populate_gallery(self) -> None:
        self.gallery.clear()
        for record_index in self._filtered_record_indexes():
            record = self.records[record_index]
            item = QListWidgetItem()
            item.setText(self._item_label(record))
            item.setToolTip(
                "\n".join(
                    [
                        f"File: {record.source_path.name}",
                        f"Original: {self._display_name(record.original_class)} ({record.original_class})",
                        f"Current: {self._display_name(record.current_class())} ({record.current_class()})",
                        f"Path: {record.source_path}",
                    ]
                )
            )
            item.setData(Qt.ItemDataRole.UserRole, record_index)
            icon = self._load_icon(record.source_path)
            if icon:
                item.setIcon(icon)
            self.gallery.addItem(item)

    def _filtered_record_indexes(self) -> List[int]:
        if not self.records:
            return []

        mode = self.filter_mode_combo.currentData()
        class_code = (
            self.filter_class_combo.currentData()
            if self.filter_class_combo.isEnabled()
            else None
        )

        if mode is None:
            return list(range(len(self.records)))

        if class_code is None:
            if mode == "new":
                return [
                    idx
                    for idx, record in enumerate(self.records)
                    if record.assigned_class
                ]
            return list(range(len(self.records)))

        matched: List[int] = []
        for idx, record in enumerate(self.records):
            if mode == "original" and record.original_class == class_code:
                matched.append(idx)
            elif mode == "new" and record.assigned_class == class_code:
                matched.append(idx)
            elif mode == "current" and record.current_class() == class_code:
                matched.append(idx)
        return matched

    def _load_icon(self, path: Path) -> Optional[QIcon]:
        return QIcon(str(path))

    def _item_label(self, record: ImageRecord) -> str:
        original = self._display_name(record.original_class)
        if record.assigned_class is None:
            return f"{record.source_path.name}\n[{original}]"
        updated = self._display_name(record.assigned_class)
        return f"{record.source_path.name}\n[{original} → {updated}]"

    def _assign_from_class_list(self, item: QListWidgetItem) -> None:
        class_code = item.data(Qt.ItemDataRole.UserRole)
        if class_code:
            self._assign_selection(str(class_code))

    def _open_image_item(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        record = self.records[int(index)]
        if not record.source_path.exists():
            self.status_bar.showMessage(
                f"Image not found: {record.source_path}",
                4000,
            )
            return
        url = QUrl.fromLocalFile(str(record.source_path.resolve()))
        if not QDesktopServices.openUrl(url):
            self.status_bar.showMessage(
                f"Unable to open image: {record.source_path}",
                4000,
            )

    def _assign_selection(self, class_code: str) -> None:
        selected_items = self.gallery.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "No selection", "Select one or more images first."
            )
            return

        record_indexes: List[int] = []
        for item in selected_items:
            index = item.data(Qt.ItemDataRole.UserRole)
            if index is None:
                continue
            record_indexes.append(int(index))
            record = self.records[int(index)]
            record.assigned_class = class_code
            item.setText(self._item_label(record))
            item.setToolTip(
                "\n".join(
                    [
                        f"File: {record.source_path.name}",
                        f"Original: {self._display_name(record.original_class)} ({record.original_class})",
                        f"Current: {self._display_name(record.current_class())} ({record.current_class()})",
                        f"Path: {record.source_path}",
                    ]
                )
            )

        if not record_indexes:
            return

        self._touch_recent_class(class_code)
        if self.filter_mode_combo.currentData() in {"new", "current"}:
            self._apply_filter()

        display = self._display_name(class_code)
        self.status_bar.showMessage(
            f"Assigned {len(record_indexes)} image(s) to {display} ({class_code}).",
            3000,
        )

    def _touch_recent_class(self, class_code: str) -> None:
        if class_code in self.recent_class_codes:
            self.recent_class_codes.remove(class_code)
        self.recent_class_codes.insert(0, class_code)
        del self.recent_class_codes[10:]
        self._populate_class_list()
        self._update_filter_class_options()

    def _show_gallery_menu(self, position: QPoint) -> None:
        if not self.class_codes:
            return

        menu = QMenu(self)
        for class_code, display in self._class_display_sequence():
            action = menu.addAction(f"{display} ({class_code})")
            if action is not None:
                action.triggered.connect(partial(self._assign_selection, class_code))
        menu.exec(self.gallery.mapToGlobal(position))

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # type: ignore[override]
        if event and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            key_value = event.key()
            if key_value in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                if self.gallery.selectedItems():
                    self._prompt_class_assignment()
                    return
            if Qt.Key.Key_1 <= key_value <= Qt.Key.Key_9:
                digit = key_value - Qt.Key.Key_0
                class_code = self.class_shortcuts.get(digit)
                if class_code:
                    self._assign_selection(class_code)
                    return
        super().keyPressEvent(event)

    def _prompt_class_assignment(self) -> None:
        suggestions = [display for _, display in self._class_display_sequence()]
        dialog = ClassAssignDialog(suggestions, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        display_name = dialog.selected_text()
        if not display_name:
            return

        class_code = self._lookup_class_code(display_name)
        if class_code is None:
            QMessageBox.warning(
                self,
                "Unknown class",
                f"No class matches '{display_name}'. Choose one from the suggestions.",
            )
            return

        self._assign_selection(class_code)

    def _lookup_class_code(self, display_name: str) -> Optional[str]:
        if not display_name:
            return None
        if display_name in self.display_to_code:
            return self.display_to_code[display_name]
        return self.display_lower_to_code.get(display_name.lower())

    def _on_filter_mode_changed(self, index: int) -> None:
        mode = self.filter_mode_combo.itemData(index)
        self.filter_class_combo.setEnabled(mode is not None)
        self._apply_filter()

    def _update_icon_size(self, value: int) -> None:
        size = QSize(value, value)
        self.gallery.setIconSize(size)
        self.gallery.setGridSize(QSize(value + 40, value + 72))

    def _on_save_result(self) -> None:
        base_dir = QFileDialog.getExistingDirectory(self, "Select export parent folder")
        if not base_dir:
            return

        base_path = Path(base_dir)
        if not base_path.exists() or not base_path.is_dir():
            QMessageBox.warning(
                self,
                "Invalid folder",
                "Select an existing directory where the export folder can be created.",
            )
            return

        export_path = self._build_export_path(base_path)
        try:
            export_path.mkdir(parents=True, exist_ok=False)
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not create export folder '{export_path.name}': {exc}",
            )
            return

        try:
            self._write_dataset(export_path)
        except Exception as exc:  # pylint: disable=broad-except
            shutil.rmtree(export_path, ignore_errors=True)
            QMessageBox.critical(self, "Save failed", f"Could not save dataset: {exc}")
            return

        QMessageBox.information(
            self,
            "Save complete",
            f"Dataset saved to new folder '{export_path.name}'.",
        )
        self.status_bar.showMessage(f"Saved dataset to {export_path}", 5000)

    def _write_dataset(self, target_root: Path) -> None:
        target_root.mkdir(parents=True, exist_ok=True)
        for record in self.records:
            class_dir = target_root / record.current_class()
            class_dir.mkdir(parents=True, exist_ok=True)
            destination = class_dir / record.source_path.name
            shutil.copy2(record.source_path, destination)

    def _build_export_path(self, base_dir: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"yolo_cls_export_{timestamp}"
        candidate = base_dir / folder_name
        suffix = 1
        while candidate.exists():
            candidate = base_dir / f"{folder_name}_{suffix}"
            suffix += 1
        return candidate


def load_mapping_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file '{path}' does not exist.")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(
                "JSON mapping must be an object of code → display name pairs."
            )
        return {str(code): str(name) for code, name in data.items()}

    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        mapping: Dict[str, str] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"Row {row_number} in '{path}' must have at least two columns."
                    )
                code, display = row[0].strip(), row[1].strip()
                if not code:
                    raise ValueError(
                        f"Row {row_number} in '{path}' has an empty class code."
                    )
                mapping[code] = display or code
        return mapping

    raise ValueError("Supported mapping formats: .json, .csv, .tsv, .txt")


def parse_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="YOLO classification dataset editor")
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Optional path to a class mapping file (.json, .csv, .tsv, .txt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional dataset folder to open on launch.",
    )
    args, remaining = parser.parse_known_args(argv[1:])
    return args, remaining


def main() -> None:
    args, qt_args = parse_args(sys.argv)

    class_mapping: Dict[str, str] = {}
    if args.mapping:
        try:
            class_mapping = load_mapping_file(args.mapping)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to load mapping file: {exc}", file=sys.stderr)
            return

    app = QApplication([sys.argv[0]] + qt_args)
    window = MainWindow(class_mapping)
    window.show()

    if args.dataset:
        window.open_dataset(args.dataset)

    app.exec()


if __name__ == "__main__":
    main()
