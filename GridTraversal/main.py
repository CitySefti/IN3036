import traverse
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QPushButton

class UI(QMainWindow):
    def init(self):
        super().init()
        self.initUI()

    def initUI(self):
        # create a combo box for selecting the game mode
        self.mode_combo_box = QComboBox(self)
        self.mode_combo_box.addItem("Traverse")
        self.mode_combo_box.move(50, 50)

        # create a button for starting the game
        self.start_button = QPushButton(traverse, self)
        self.start_button.move(50, 150)

app = QApplication([])
window = UI();
window.setGeometry(300, 300, 300, 300)
window.setWindowTitle("UI")
window.show()
app.exec_()

