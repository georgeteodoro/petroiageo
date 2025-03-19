import argparse
from mpi4py import MPI
import sys
from time import sleep

import config_parser
import mpi_module

import manager
import worker

import sys
from PyQt6 import QtWidgets, uic

# Used only for retrieving the shape of a feature
from feature_data.backends.FeatureDataBase import FeatureDataBase

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def rel():
    print('rel..ing?')

class TrainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(TrainWindow, self).__init__()
        uic.loadUi('qt/train.ui', self)

        self.show()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('qt/init.ui', self)

        self.trainButton = self.findChild(QtWidgets.QPushButton, 'trainButton')
        self.trainButton.clicked.connect(self.train)

        self.relButton = self.findChild(QtWidgets.QPushButton, 'relButton')
        self.relButton.clicked.connect(rel)

        self.show()

    def train(self):
        self.trainWindow = TrainWindow()
        self.trainWindow.show()
        self.close()


def main(args_str=None):

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
