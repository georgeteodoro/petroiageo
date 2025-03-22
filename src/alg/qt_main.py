import sys
from PyQt6 import QtWidgets, uic

from qt.ConfigWindow import ConfigWindow
from qt.TrainWindow import TrainWindow

# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         uic.loadUi('qt/init.ui', self)

#         self.trainButton = self.findChild(QtWidgets.QPushButton, 'trainButton')
#         self.trainButton.clicked.connect(self.train)

#         self.relButton = self.findChild(QtWidgets.QPushButton, 'relButton')
#         self.relButton.clicked.connect(rel)

#         self.show()

#     def train(self):
#         self.trainWindow = TrainWindow()
#         self.trainWindow.show()
#         self.close()


def main(args_str=None):

    app = QtWidgets.QApplication(sys.argv)
    # window = MainWindow()
    window = TrainWindow()
    # window = ConfigWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
