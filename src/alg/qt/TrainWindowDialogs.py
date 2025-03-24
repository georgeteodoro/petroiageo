from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import pyqtSignal, pyqtSlot

import threading
import subprocess
from time import sleep

from qt.ConfigWindow import ConfigWindow, FeaturePlace


class LoadInfoWindow(QtWidgets.QDialog):
    def __init__(self, config, parent):
        super(LoadInfoWindow, self).__init__()

        self.config = config
        self.parent = parent

        uic.loadUi('qt/dsetinfo.ui', self)
    
        self.okPB = self.findChildren(QtWidgets.QPushButton,
                                              'okPB')[0]
        self.okPB.setEnabled(False)
        self.okPB.clicked.connect(self.done)

        self.logText = self.findChildren(QtWidgets.QPlainTextEdit,
                                 'logText')[0]

        poros_file_path = self.config.starting_porosity_cube_path

        run_str = [
            'python3', '-u', 'hdf5_util_qt.py', poros_file_path, '-i', '10000'
        ]

        self.new_log_line.connect(self.add_to_log)

        self.info_process = subprocess.Popen(run_str, stdout=subprocess.PIPE)

        self.log_thread = threading.Thread(target=self.add_to_log_buffer)
        self.log_thread.start()


    # Signal that a new line was outputted from the subprocess
    new_log_line = pyqtSignal(str)
    # Append a new line to the correct place
    @pyqtSlot(str)
    def add_to_log(self, line):
        self.logText.appendPlainText(line)

    def add_to_log_buffer(self):
        # While process is alive
        info = []
        while self.info_process.poll() is None:
            l = self.info_process.stdout.readline()[:-1].decode('ascii')
            if len(l) > 0:
                # There is a new line, send an update
                self.new_log_line.emit(l)
                info.append(l)
            else:
                # Don't busy wait...
                sleep(1)
        self.parent.update_dset_info_signal.emit(info)
        self.okPB.setEnabled(True)



class ClearItsWindow(QtWidgets.QDialog):
    def __init__(self, config, parent):
        super(ClearItsWindow, self).__init__()

        self.config = config
        self.parent = parent

        uic.loadUi('qt/clear.ui', self)
    
        self.removeButton = self.findChildren(QtWidgets.QPushButton,
                                              'removeButton')[0]
        self.removeButton.clicked.connect(self.perform_clear)

        self.poros_file_path = self.config.starting_porosity_cube_path

        self.itMaxLE = self.findChildren(QtWidgets.QLineEdit,
                                 'itMaxLE')[0]

        self.logText = self.findChildren(QtWidgets.QPlainTextEdit,
                                 'logText')[0]

        self.new_log_line.connect(self.add_to_log)


    # Signal that a new line was outputted from the subprocess
    new_log_line = pyqtSignal(str)
    # Append a new line to the correct place
    @pyqtSlot(str)
    def add_to_log(self, line):
        self.logText.appendPlainText(line)

    def add_to_log_buffer(self):
        # While process is alive
        info = []
        while self.info_process.poll() is None:
            l = self.info_process.stdout.readline()[:-1].decode('ascii')
            if len(l) > 0:
                # There is a new line, send an update
                self.new_log_line.emit(l)
                info.append(l)
            else:
                # Don't busy wait...
                sleep(1)
        self.parent.update_dset_info_signal.emit(info)
        self.removeButton.setText("Continuar")
        self.removeButton.setEnabled(True)
        self.removeButton.clicked.disconnect(self.perform_clear)
        self.removeButton.clicked.connect(self.done)

    def perform_clear(self):
        self.removeButton.setEnabled(False)

        if len(self.itMaxLE.text()) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setText('Valor inválido.')
            msg.setWindowTitle('Erro')
            msg.exec()
            self.removeButton.setEnabled(True)
            return
        
        run_str = [
            'python3', '-u', 'hdf5_util_qt.py', self.poros_file_path, '-i', 
            '10000', '-c', self.itMaxLE.text()
        ]

        self.info_process = subprocess.Popen(run_str, stdout=subprocess.PIPE)

        self.log_thread = threading.Thread(target=self.add_to_log_buffer)
        self.log_thread.start()
