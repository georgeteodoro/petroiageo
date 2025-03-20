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

# # Used only for retrieving the shape of a feature
# from feature_data.backends.FeatureDataBase import FeatureDataBase

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()


def rel():
    print('rel..ing?')

class FeaturePlace(Enum):
    MMAP = '', 'Mem Maping'                    # simple
    CACHING = 'is_feature_cache', 'Cached'     # mmap cache
    INMEM = 'is_feature_in_mem', 'Full in-mem' # in mem all

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


class ConfigWindow(QtWidgets.QMainWindow):
    def __init__(self, config, parent):
        super(ConfigWindow, self).__init__()
        uic.loadUi('qt/config.ui', self)

        self.config = config
        self.parent = parent

        # LineEdits
        self.pathConfigLE = self.findChildren(QtWidgets.QLineEdit,
            'pathConfigLE')[0]

        self.pathPorosityLE = self.findChildren(QtWidgets.QLineEdit,
            'pathPorosityLE')[0]

        self.itIniLE = self.findChildren(QtWidgets.QLineEdit,
            'itIniLE')[0]

        self.nItsLE = self.findChildren(QtWidgets.QLineEdit,
            'nItsLE')[0]

        self.featuresPathLE = self.findChildren(QtWidgets.QLineEdit,
            'featuresPathLE')[0]

        self.nFeatLE = self.findChildren(QtWidgets.QLineEdit,
            'nFeatLE')[0]

        self.nFeatSelLE = self.findChildren(QtWidgets.QLineEdit,
            'nFeatSelLE')[0]

        self.nTrialsLE = self.findChildren(QtWidgets.QLineEdit,
            'nTrialsLE')[0]

        self.windowLE = self.findChildren(QtWidgets.QLineEdit,
            'windowLE')[0]

        self.nWorkersLE = self.findChildren(QtWidgets.QLineEdit,
            'nWorkersLE')[0]

        self.nTrainChunksLE = self.findChildren(QtWidgets.QLineEdit,
            'nTrainChunksLE')[0]

        # CheckBoxes
        self.fSchedCheck = self.findChildren(QtWidgets.QCheckBox,
            'fSchedCheck')[0]

        self.dfsPorosityCheck = self.findChildren(QtWidgets.QCheckBox,
            'dfsPorosityCheck')[0]

        self.sharedTdCheck = self.findChildren(QtWidgets.QCheckBox,
            'sharedTdCheck')[0]

        # ComboBoxes
        self.featurePlaceCB = self.findChildren(QtWidgets.QComboBox,
            'featurePlaceCB')[0]
        for p in FeaturePlace:
            self.featurePlaceCB.addItem(p.value[1], userData=p.value[0])

        # Buttons
        self.loadConfigFileButton = self.findChildren(QtWidgets.QPushButton, 
            'loadConfigFileButton')[0]
        self.loadConfigFileButton.clicked.connect(self.load_conf_file)

        self.saveConfigButton = self.findChildren(QtWidgets.QPushButton, 
            'saveConfigButton')[0]
        self.saveConfigButton.clicked.connect(self.save_config)

        self.load_config(self.config)

        self.show()

    def load_conf_file(self):
        config_path, ok = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecione um arquivo de configuração", 
            "", 
            "Config (*.yaml)"
        )
        if not config_path:
            return

        # Parse config file
        self.config = config_parser.YAMLConfig(config_path)
        self.pathConfigLE.setText(config_path)
        self.load_config(self.config)

    def load_config(self, config):

        # Update shown values
        self.pathPorosityLE.setText(config.starting_porosity_cube_path)
        self.itIniLE.setText(str(config.get_param('it_ini')))
        self.nItsLE.setText(str(config.get_param('nits')))
        self.featuresPathLE.setText(config.features_folder)
        # self.nFeatLE.setText( XXX)
        self.nFeatSelLE.setText(str(config.alg['max_num_features']))
        # self.nTrialsLE.setText( XXX)
        self.windowLE.setText(str(config.alg['window']))
        # self.nWorkersLE.setText( XXX)
        # self.nTrainChunksLE.setText( XXX)
        # self.fSchedCheck.setText( XXX)
        # self.dfsPorosityCheck.setText( XXX)
        # self.sharedTdCheck.setText( XXX)
        # self.featurePlaceCB.setText( XXX)


    def save_config(self):
        self.config.starting_porosity_cube_path = self.pathPorosityLE.text()
        self.config.add_param('it_ini', int(self.itIniLE.text()))
        self.config.add_param('nits', int(self.nItsLE.text()))
        self.config.features_folder = self.featuresPathLE.text()
        self.config.add_param('num_features', self.nFeatLE.text())
        self.config.add_param('max_num_features', self.nFeatSelLE.text())
        self.config.add_param('max_feats_for_trial', self.nTrialsLE.text())
        self.config.alg['window'] = self.windowLE.text()
        self.config.add_param('n_workers', self.nWorkersLE.text())
        self.config.add_param('n_training_chunks', self.nTrainChunksLE.text())
        self.config.add_param('fsched_loc', self.fSchedCheck.isChecked())
        self.config.add_param('is_porosity_dfs', 
            self.dfsPorosityCheck.isChecked())
        self.config.add_param('is_shared_trial_data', 
            self.sharedTdCheck.isChecked())
        if len(self.featurePlaceCB.currentData()) > 0:
            self.config.add_param(self.featurePlaceCB.currentData(), True)

        self.parent.update_config(self.config)

        self.close()



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
    # window = MainWindow()
    window = TrainWindow()
    # window = ConfigWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
