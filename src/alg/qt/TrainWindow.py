from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import pyqtSignal, pyqtSlot

import threading
import subprocess
from time import sleep

from qt.ConfigWindow import ConfigWindow, FeaturePlace

class TrainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(TrainWindow, self).__init__()

        self.config = None

        self.log_buffer = []

        uic.loadUi('qt/train.ui', self)

        self.logText = self.findChildren(QtWidgets.QPlainTextEdit, 
            'logText')[0]

        self.configButton = self.findChildren(QtWidgets.QPushButton, 
            'configButton')[0]
        self.configButton.clicked.connect(self.configure)

        self.propagateButton = self.findChildren(QtWidgets.QPushButton, 
            'propagateButton')[0]
        self.propagateButton.clicked.connect(self.propagate)

        self.prop_process = None

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

        run_str = ['mpirun', '-np', str(n_workers + 1), '--bind-to', 'core', 
                   'python3', 'main.py', '--no-abort', '--config', config_path,
                   '--poros-file', poros_file_path, '--it', str(it_ini), 
                   '--nits', str(n_its), '--nf', str(n_feats), '--nsf', 
                   str(n_feat_sel), '--ntf', str(n_trials), '-w', str(window), 
                   f_sched, f_place, poros_dfs, shared_td, chunks]

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
        
        self.prop_process = subprocess.Popen(
            run_str, stdout=subprocess.PIPE)

        self.log_thread1 = threading.Thread(target=self.add_to_log_buffer)
        self.log_thread1.start()

        return True

    # Signal that a new line was outputted from the subprocess
    new_log_line = pyqtSignal(str)
    # Append a new line to the correct place
    @pyqtSlot(str)
    def add_to_log(self, line):
        self.logText.appendPlainText(line)  

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
        if self.start_prop():
            self.configButton.setEnabled(False)
            self.propagateButton.setText('Stop')
            self.propagateButton.clicked.connect(self.stop_propagate)
            self.propagateButton.clicked.disconnect(self.propagate)
            self.propagateButton.setEnabled(True)
        else:
            self.propagateButton.setEnabled(True)


    def stop_propagate(self):
        self.end_prop()
        self.configButton.setEnabled(True)
        self.propagateButton.setText('Propagar')
        self.propagateButton.clicked.connect(self.propagate)
        self.propagateButton.clicked.disconnect(self.stop_propagate)


