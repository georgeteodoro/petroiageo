from PyQt6 import QtWidgets, uic

import config_parser


# key, config option, text for ComboBox
FeaturePlace = {0: ('',          'Mem Maping' ), # simple
                1: ('--f-cache', 'Cached',    ), # mmap cache
                2: ('--f-inmem', 'Full in-mem'), # in mem all
               }

class ConfigWindow(QtWidgets.QMainWindow):
    def __init__(self, config, parent):
        super(ConfigWindow, self).__init__()
        uic.loadUi('qt/config.ui', self)

        self.config = config
        self.parent = parent

        # LineEdits
        self.LEs = []
        self.pathConfigLE = self.findChildren(QtWidgets.QLineEdit,
            'pathConfigLE')[0]
        # self.LEs.append(self.pathConfigLE)

        self.pathPorosityLE = self.findChildren(QtWidgets.QLineEdit,
            'pathPorosityLE')[0]
        self.LEs.append(self.pathPorosityLE)

        self.itIniLE = self.findChildren(QtWidgets.QLineEdit,
            'itIniLE')[0]
        self.LEs.append(self.itIniLE)

        self.nItsLE = self.findChildren(QtWidgets.QLineEdit,
            'nItsLE')[0]
        self.LEs.append(self.nItsLE)

        self.featuresPathLE = self.findChildren(QtWidgets.QLineEdit,
            'featuresPathLE')[0]
        self.LEs.append(self.featuresPathLE)

        self.nFeatLE = self.findChildren(QtWidgets.QLineEdit,
            'nFeatLE')[0]
        self.LEs.append(self.nFeatLE)

        self.nFeatSelLE = self.findChildren(QtWidgets.QLineEdit,
            'nFeatSelLE')[0]
        self.LEs.append(self.nFeatSelLE)

        self.nTrialsLE = self.findChildren(QtWidgets.QLineEdit,
            'nTrialsLE')[0]
        self.LEs.append(self.nTrialsLE)

        self.windowLE = self.findChildren(QtWidgets.QLineEdit,
            'windowLE')[0]
        self.LEs.append(self.windowLE)

        self.nWorkersLE = self.findChildren(QtWidgets.QLineEdit,
            'nWorkersLE')[0]
        self.LEs.append(self.nWorkersLE)

        self.nTrainChunksLE = self.findChildren(QtWidgets.QLineEdit,
            'nTrainChunksLE')[0]
        self.LEs.append(self.nTrainChunksLE)

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
        for key, (_, txt) in FeaturePlace.items():
            self.featurePlaceCB.addItem(txt, userData=key)

        # Buttons
        self.loadConfigFileButton = self.findChildren(QtWidgets.QPushButton, 
            'loadConfigFileButton')[0]
        self.loadConfigFileButton.clicked.connect(self.load_conf_file)

        self.saveConfigButton = self.findChildren(QtWidgets.QPushButton, 
            'saveConfigButton')[0]
        self.saveConfigButton.clicked.connect(self.save_config)

        if self.config == None:
            self.config = config_parser.YAMLConfig()
        else:
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
        self.config.add_param('config_path', config_path)
        self.load_config(self.config)

    def load_config(self, config):

        # Update shown values
        self.pathConfigLE.setText(config.get_param('config_path'))
        self.pathPorosityLE.setText(config.starting_porosity_cube_path)
        self.itIniLE.setText(str(config.get_param('it_ini')))
        self.nItsLE.setText(str(config.get_param('nits')))
        self.featuresPathLE.setText(config.features_folder)
        self.nFeatLE.setText(str(
            config.get_param('num_features'))) # Not in config file
        self.nFeatSelLE.setText(str(
            config.alg['max_num_features']))
        self.nTrialsLE.setText(str(
            config.get_param('max_feats_for_trial'))) # Not in config file
        self.windowLE.setText(str(config.alg['window']))
        self.nWorkersLE.setText(str(
            config.get_param('n_workers'))) # Not in config file
        self.nTrainChunksLE.setText(str(
            config.get_param('n_training_chunks'))) # Not in config file

        if config.get_param('fsched_loc') is None:
            self.fSchedCheck.setChecked(False) # Not in config file
        else:
            self.fSchedCheck.setChecked(bool(config.get_param(
            'fsched_loc'))) # Not in config file

        if config.get_param('is_porosity_dfs') is None:
            self.dfsPorosityCheck.setChecked(False) # Not in config file
        else:
            self.dfsPorosityCheck.setChecked(bool(config.get_param(
                'is_porosity_dfs'))) # Not in config file

        if config.get_param('is_shared_trial_data') is None:
            self.sharedTdCheck.setChecked(False) # Not in config file
        else:
            self.sharedTdCheck.setChecked(bool(
                config.get_param('is_shared_trial_data'))) # Not in config file

        if config.get_param('feature_place') is not None:
            self.featurePlaceCB.setCurrentIndex(
                config.get_param('feature_place')) # Not in config file


    def save_config(self):
        msg = QtWidgets.QMessageBox()
        msg.setText('Erro de variáveis')
        msg.setInformativeText('Verifique se os tipos estão corretos '
                               'e se todos os campos foram preenchidos.')
        msg.setWindowTitle('Erro')
        # msg.exec()

        # Check if all fields are filled
        all_filled = True
        for LE in self.LEs:
            if len(LE.text()) == 0:
                all_filled = False
                break
        if not all_filled:
            msg.exec()
            return

        # Update config
        self.config.starting_porosity_cube_path = self.pathPorosityLE.text()
        self.config.add_param('it_ini', int(self.itIniLE.text()))
        self.config.add_param('nits', int(self.nItsLE.text()))
        self.config.features_folder = self.featuresPathLE.text()
        self.config.add_param('num_features', int(self.nFeatLE.text()))
        self.config.add_param('max_num_features', int(self.nFeatSelLE.text()))
        self.config.add_param('max_feats_for_trial', int(self.nTrialsLE.text()))
        self.config.alg['window'] = int(self.windowLE.text())
        self.config.add_param('n_workers', int(self.nWorkersLE.text()))
        self.config.add_param('n_training_chunks', 
                              int(self.nTrainChunksLE.text()))
        self.config.add_param('fsched_loc', self.fSchedCheck.isChecked())
        self.config.add_param('is_porosity_dfs', 
            self.dfsPorosityCheck.isChecked())
        self.config.add_param('is_shared_trial_data', 
            self.sharedTdCheck.isChecked())

        self.config.add_param('feature_place', 
                              self.featurePlaceCB.currentData())

        # Push config object to parent
        self.parent.update_config(self.config)

        self.close()
