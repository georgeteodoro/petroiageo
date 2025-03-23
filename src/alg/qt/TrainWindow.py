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

        poros_file_path = self.config.starting_porosity_cube_path

        run_str = [
            'python3', 'hdf5_util_qt.py', poros_file_path, '-i', '10000'
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




class TrainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(TrainWindow, self).__init__()

        self.config = None
        self.prop_process = None
        self.is_propagating = False

        uic.loadUi('qt/train.ui', self)

        self.logText = self.findChildren(QtWidgets.QPlainTextEdit,
                                         'logText')[0]

        # Buttons -------------------------------------------------------------
        self.dimLE = self.findChildren(QtWidgets.QLineEdit,
                                         'dimLE')[0]

        self.nPointsLE = self.findChildren(QtWidgets.QLineEdit,
                                         'nPointsLE')[0]

        self.itCurLE = self.findChildren(QtWidgets.QLineEdit,
                                         'itCurLE')[0]

        self.nPropLE = self.findChildren(QtWidgets.QLineEdit,
                                         'nPropLE')[0]

        self.nRealLE = self.findChildren(QtWidgets.QLineEdit,
                                         'nRealLE')[0]

        self.nEmptyLE = self.findChildren(QtWidgets.QLineEdit,
                                         'nEmptyLE')[0]


        # Buttons -------------------------------------------------------------
        self.configButton = self.findChildren(QtWidgets.QPushButton,
                                              'configButton')[0]
        self.configButton.clicked.connect(self.configure)

        self.propagateButton = self.findChildren(QtWidgets.QPushButton,
                                                 'propagateButton')[0]
        self.propagateButton.clicked.connect(self.propagate)

        self.dsetLoadButton = self.findChildren(QtWidgets.QPushButton,
                                                'dsetLoadButton')[0]
        self.dsetLoadButton.clicked.connect(self.load_dset_info)

        self.dsetClearButton = self.findChildren(QtWidgets.QPushButton,
                                                 'dsetClearButton')[0]
        self.dsetClearButton.clicked.connect(self.clear_dset)

        self.totalPB = self.findChildren(QtWidgets.QProgressBar, 'totalPB')[0]
        self.itPB = self.findChildren(QtWidgets.QProgressBar, 'itPB')[0]
        self.fItPB = self.findChildren(QtWidgets.QProgressBar, 'fItPB')[0]

        self.update_dset_info_signal.connect(self.dset_info)


        # Signals for propagation

        # Signal/slot: new line from proc to add to log field
        self.new_log_line.connect(self.add_to_log)

        # Signal/slot: prop proc is done, reset view
        self.prop_is_done.connect(self.stop_propagate)

        self.show()

    def __del__(self):
        self.end_prop()

    def end_prop(self):
        if self.prop_process is not None:
            # Repetition to ensure the process actually ended...
            self.prop_process.terminate()
            self.prop_process.terminate()
            self.prop_process.terminate()
            self.log_thread1.join()
            self.log_thread1 = None
            self.prop_process = None

    def start_prop(self):
        if self.prop_process is not None:
            print('already running.... this is bad')
            return False

        # Reset progress bars
        self.totalPB.setValue(0)
        self.itPB.setValue(0)
        self.fItPB.setValue(0)

        msg = QtWidgets.QMessageBox()
        msg.setText('Erro de configuração')
        msg.setInformativeText('Verifique a configuração de execução '
                               'foi feita.')
        msg.setWindowTitle('Erro')

        # Check if config was filled
        if self.config is None:
            msg.exec()
            return False

        n_workers = self.config.get_param('n_workers')
        config_path = self.config.get_param('config_path')
        poros_file_path = self.config.starting_porosity_cube_path
        it_ini = self.config.get_param('it_ini')
        n_its = self.config.get_param('nits')
        n_feats = self.config.get_param('num_features')
        n_feat_sel = self.config.get_param('max_num_features')
        n_trials = self.config.get_param('max_feats_for_trial')
        window = str(self.config.alg['window'])
        n_chunks = self.config.get_param('n_training_chunks')

        f_sched = ''
        if self.config.get_param('fsched_loc'):
            f_sched = '--fsched-loc'

        f_place = self.config.get_param('feature_place')
        if f_place is not None:
            f_place = FeaturePlace[int(f_place)][0]
        else:
            f_place = ''

        poros_dfs = ''
        if self.config.get_param('is_porosity_dfs'):
            poros_dfs = '--p-dfs'

        shared_td = ''
        if self.config.get_param('is_shared_trial_data'):
            shared_td = '--t-shd'

        chunks = ''
        if self.config.get_param('fsched_loc'):
            chunks = f'--tr-chunk {n_chunks}'

        run_str = [
            'mpirun', '-np',
            str(n_workers + 1), '--bind-to', 'core', 'python3', 'main.py',
            '--no-abort', '--config', config_path, '--poros-file',
            poros_file_path, '--it',
            str(it_ini), '--nits',
            str(n_its), '--nf',
            str(n_feats), '--nsf',
            str(n_feat_sel), '--ntf',
            str(n_trials), '-w',
            str(window), f_sched, f_place, poros_dfs, shared_td, chunks
        ]

        # Remove the last empty config
        while '' in run_str:
            run_str.remove('')

        # Check if config was filled
        for s in run_str:
            if s is None:
                msg.exec()
                return False

        self.logText.clear()
        self.logText.appendPlainText(' '.join(run_str))

        self.prop_process = subprocess.Popen(run_str, stdout=subprocess.PIPE)

        self.log_thread1 = threading.Thread(target=self.add_to_log_buffer)
        self.log_thread1.start()

        return True

    # Signal that a new line was outputted from the subprocess
    new_log_line = pyqtSignal(str)
    # Append a new line to the correct place
    @pyqtSlot(str)
    def add_to_log(self, line):
        self.logText.appendPlainText(line)

        # Only check manager info
        if line[0:9] != '[manager]':
            return

        sline = line.split(' ')

        # Check prep progress bar total
        if 'RunPlanIt' in sline:
            self.totalPB.setMinimum(0)
            self.totalPB.setMaximum(int(sline[4]))
        elif 'RunPlanFSel' in sline:
            self.itPB.setMinimum(0)
            self.itPB.setMaximum(int(sline[2]))
        elif 'RunPlanTrials' in sline:
            self.fItPB.setMinimum(0)
            self.fItPB.setMaximum(int(sline[2]))

        elif 'TrialDone' in sline:
            # Single trial done
            self.fItPB.setValue(self.fItPB.value() + 1)
        elif 'fItDone' in sline:
            # Single feature selected
            self.itPB.setValue(self.itPB.value() + 1)
            self.fItPB.setValue(0)
        elif 'DoneIt' in sline:
            # All features from a single IT are done
            self.totalPB.setValue(self.totalPB.value() + 1)
            self.itPB.setValue(0)

            # If all is done, keep all bars at 100%
            if self.totalPB.value() == self.totalPB.maximum():
                self.itPB.setValue(self.itPB.maximum())
                self.fItPB.setValue(self.fItPB.maximum())

        # Later... parse line to update other fields...

    # Signal that the subprocess has ended
    prop_is_done = pyqtSignal(int)

    def add_to_log_buffer(self):
        # While process is alive
        while self.prop_process.poll() is None:
            l = self.prop_process.stdout.readline()[:-1].decode('ascii')
            if len(l) > 0:
                # There is a new line, send an update
                self.new_log_line.emit(l)
            else:
                # Don't busy wait...
                sleep(1)

        self.prop_is_done.emit(0)

    def configure(self):
        self.configWindow = ConfigWindow(self.config, self)
        self.configWindow.show()

    def update_config(self, config):
        self.config = config

    def propagate(self):
        self.propagateButton.setEnabled(False)
        self.is_propagating = True
        if self.start_prop():
            self.configButton.setEnabled(False)
            self.propagateButton.setText('Stop')
            self.propagateButton.clicked.connect(self.stop_propagate)
            self.propagateButton.clicked.disconnect(self.propagate)
            self.propagateButton.setEnabled(True)
        else:
            self.propagateButton.setEnabled(True)

    def stop_propagate(self):
        if self.is_propagating:
            self.propagateButton.setEnabled(False)
            self.is_propagating = False
            self.end_prop()
            self.configButton.setEnabled(True)
            self.propagateButton.setText('Propagar')
            self.propagateButton.clicked.connect(self.propagate)
            self.propagateButton.clicked.disconnect(self.stop_propagate)
            self.propagateButton.setEnabled(True)

    def load_dset_info(self):
        if self.config is None:
            msg = QtWidgets.QMessageBox()
            msg.setText('Erro de configuração')
            msg.setInformativeText('Não foi configurado um path de porosidade.')
            msg.setWindowTitle('Erro')
            msg.exec()
            return

        msg = LoadInfoWindow(self.config, self)
        msg.exec()

        # self.log_thread1 = threading.Thread(target=self.add_to_log_buffer)
        # self.log_thread1.start()

    update_dset_info_signal = pyqtSignal(list)
    @pyqtSlot(list)
    def dset_info(self, info):
        for l in info:
            l = l.replace(', ', ',')
            # l = l.replace('(', '')
            # l = l.replace(')', '')
            l = l.split(' ')

            if len(l) > 1 and l[1] == 'shape:':
                self.dimLE.setText(l[2])
            if l[0] == 'Total':
                self.nPointsLE.setText(l[2])
            if l[0] == 'MaxRing':
                self.itCurLE.setText(l[1])
            if l[0] == '\tPropagated:':
                self.nPropLE.setText(f'{l[1]}/{l[2]}')
            if l[0] == '\tReal:':
                self.nRealLE.setText(f'{l[1]}/{l[2]}')
            if l[0] == '\tEmpty:':
                self.nEmptyLE.setText(f'{l[1]}/{l[2]}')

    def clear_dset(self):
        pass
