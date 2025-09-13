import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QPushButton, QToolButton, QMenu
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from features import EyeDistance, BlinkDetection, AdaptiveBrightness, NightLight
from pynput import keyboard


class OverlayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Health Overlay")
        self.setGeometry(100, 100, 300, 180)

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.offset = None

        self.setStyleSheet("""
            QWidget#container {
                background-color: #1f1f2f;
                border-radius: 14px;
                color: #f0f0f0;
            }
            QLabel#title {
                font-weight: 600;
                font-size: 15px;
                color: #f5f5f5;
            }
            QPushButton#closeBtn {
                background-color: transparent;
                color: #f0f0f0;
                font-weight: bold;
                border: none;
                font-size: 16px;
            }
            QPushButton#closeBtn:hover {
                color: #ff5f56;
            }
            QToolButton#toggleBtn {
                background-color: #2c2c3c;
                color: #f0f0f0;
                border-radius: 14px;
                padding: 8px 14px;
                font-weight: 500;
            }
            QToolButton#toggleBtn:checked {
                background-color: #4caf50;
                color: white;
            }
            QToolButton#dropdownBtn {
                background-color: #2c2c3c;
                border-radius: 8px;
                padding: 6px ;
                color: #f0f0f0;
                font-weight: 600;
                text-align: left;
                font-size: 14px;
            }
            QToolButton#dropdownBtn:hover {
                background-color: #3a3a4a;
            }
            QToolButton#dropdownBtn::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                right: 10px;
            }
            QMenu {
                background-color: #2c2c3c;
                color: #f0f0f0;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item:selected {
                background-color: #3a3a4a;
            }
        """)

        container = QWidget()
        container.setObjectName("container")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(12)

        # Top bar
        top_layout = QHBoxLayout()
        self.label = QLabel("⚡ Overlay")
        self.label.setObjectName("title")
        top_layout.addWidget(self.label)

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("closeBtn")
        self.close_btn.clicked.connect(self.exit_overlay)
        top_layout.addWidget(self.close_btn, alignment=Qt.AlignRight)
        container_layout.addLayout(top_layout)

        # Turn On toggle
        self.toggle_all_btn = QToolButton()
        self.toggle_all_btn.setText("Turn On Overlay")
        self.toggle_all_btn.setCheckable(True)
        self.toggle_all_btn.setObjectName("toggleBtn")
        self.toggle_all_btn.toggled.connect(self.toggle_all_features)
        container_layout.addWidget(self.toggle_all_btn)

        # Initialize features
        self.features = {
            "Eye Distance": EyeDistance(),
            "Blink Detection": BlinkDetection(),
            "Adaptive Brightness": AdaptiveBrightness(),
            "Night Light": NightLight()
        }

        # Dropdown for features
        self.dropdown_btn = QToolButton()
        self.dropdown_btn.setText("All Features")
        self.dropdown_btn.setPopupMode(QToolButton.InstantPopup)
        self.dropdown_btn.setObjectName("dropdownBtn")  # ✅ Fixed name
        self.dropdown_btn.setMinimumWidth(200)

        self.feature_menu = QMenu()
        self.feature_menu.setMinimumWidth(200)
        self.feature_actions = {}
        for name in self.features.keys():
            action = QAction(name, self, checkable=True)
            action.setChecked(True)
            action.toggled.connect(lambda checked, n=name: self.toggle_feature_dropdown(n, checked))
            self.feature_menu.addAction(action)
            self.feature_actions[name] = action

        self.dropdown_btn.setMenu(self.feature_menu)
        container_layout.addWidget(self.dropdown_btn, alignment=Qt.AlignHCenter)


        main_layout = QVBoxLayout()
        main_layout.addWidget(container)
        self.setLayout(main_layout)

        self.toggle_all_btn.setChecked(True)

    # Draggable
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.offset is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)

    def mouseReleaseEvent(self, event):
        self.offset = None

    # Feature toggle
    def toggle_feature_dropdown(self, feature_name, state):
        feature = self.features[feature_name]
        if state:
            feature.start()
        else:
            feature.stop()

        # Sync main toggle
        all_checked = all(action.isChecked() for action in self.feature_actions.values())
        if self.toggle_all_btn.isChecked() != all_checked:
            self.toggle_all_btn.blockSignals(True)
            self.toggle_all_btn.setChecked(all_checked)
            self.toggle_all_btn.blockSignals(False)

    def toggle_all_features(self, state):
        for action in self.feature_actions.values():
            action.blockSignals(True)
            action.setChecked(state)
            action.blockSignals(False)

        for name, feature in self.features.items():
            if state:
                feature.start()
            else:
                feature.stop()

    def exit_overlay(self):
        QApplication.quit()


class HotkeyListener:
    def __init__(self, toggle_callback):
        self.toggle_callback = toggle_callback
        self.listener = keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+o': self.on_toggle
        })

    def on_toggle(self):
        if callable(self.toggle_callback):
            self.toggle_callback()

    def start(self):
        self.listener.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = OverlayApp()
    overlay.show()

    def toggle_overlay():
        if overlay.isVisible():
            overlay.hide()
        else:
            overlay.show()

    hotkey = HotkeyListener(toggle_overlay)
    hotkey.start()
    sys.exit(app.exec())
