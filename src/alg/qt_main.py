# import argparse
# from mpi4py import MPI
import sys

import config_parser
# import mpi_module

# import manager
# import worker

import sys
from PyQt6 import QtWidgets, uic

from enum import Enum

from qt.ConfigWindow import ConfigWindow

# # Used only for retrieving the shape of a feature
# from feature_data.backends.FeatureDataBase import FeatureDataBase

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

class TrainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(TrainWindow, self).__init__()

        self.config = config_parser.YAMLConfig()

        uic.loadUi('qt/train.ui', self)

        self.configButton = self.findChildren(QtWidgets.QPushButton, 
            'configButton')[0]
        self.configButton.clicked.connect(self.configure)

        self.show()

    def configure(self):
        self.configWindow = ConfigWindow(self.config, self)
        self.configWindow.show()
        # self.close()

    def update_config(self, config):
        self.config = config


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
