import yaml

try:
    from yaml import CBaseLoader as Loader
except ImportError:
    from yaml import BaseLoader as Loader

import pathlib
import enum

class FeatureSelection(enum.Enum):
    FORWARD = 0
    NONE = 1

    @staticmethod
    def new_from_key(key):
        try:
            new_feat_selection = FeatureSelection[key] 
            return new_feat_selection
        except:
            error_msg = "Key error for FeatureSelectionType. "
            error_msg += f"Got {key} but should be one of {[key for key in FeatureSelection.__members__]}"
            raise KeyError(error_msg) 

class SaveModelTypes():
    ALL = 'all'
    LAST = 'last'

    @staticmethod
    def _has_positives_only(my_list:list) -> bool:
        return all([element > 0 for element in my_list])

    @staticmethod
    def raise_if_not_valid(save_model_type: str|list):
        if isinstance(save_model_type, list):
            if not self._has_positives_only(save_model_type):
                raise ValueError("alg.save_models_on: The list should have positive integers only!")
        elif isinstance(save_model_type, str):
            valid_strs = [SaveModelTypes.ALL, SaveModelTypes.LAST]
            if save_model_type[:6] == 'every_':
                try:
                    n = int(save_model_type[6:])
                    if n == 0:
                        raise
                except:
                    raise ValueError("alg.save_models_on: 'n' should be a positive integer!")
            elif save_model_type not in valid_strs:
                raise ValueError(f"alg.save_models_on: This str is not valid. Should be one of {valid_strs} but '{save_models_on}' was given!")



class Config():
    """
    This is a config parser.
    If config_path is provided, all other params are ignored and the config will be read from the file.
    If config_path is not provided, it will try to read config_dict. If it is not None, config_str will be ignored.
    If config_path and config_dict are both None, it will try to parse the config_str.
    If all parameters are None, then it will have the default configuration.
    """
    
    def __init__(self, config_path: str|pathlib.Path = None,
                 config_dict: dict = None, config_str: str = None):

        input_config = self._get_input_config(config_path, config_dict, config_str)

        input_config = self._treat_input_config(input_config)

        self.config = self._base_config()

        self._update_config_with_input_config(input_config)
        
        self._raise_if_not_valid_config(self.config)
    
    def _update_config_with_input_config(self, input_config:dict):
        if 'wells' in input_config:
            self.config['wells'].update(input_config['wells'])
        
        if 'alg' in input_config:
            self.config['alg'].update(input_config['alg'])
        
        if 'features_folder' in input_config:
            self.config['features_folder'] = input_config['features_folder']
        
        if 'porosity_cube_output_path' in input_config:
            self.config['porosity_cube_output_path'] = input_config['porosity_cube_output_path']
    
    def _get_input_config(self, config_path: str|pathlib.Path = None,
                          config_dict: dict = None, config_str: str = None):
        """
        Tries to choose where are the input configs. If every param is none, return a empty dict.
        """
        
        if config_path is not None:
            if isinstance(config_path, str) or isinstance(config_path, pathlib.Path):
            
                self._raise_if_path_doesnt_exists_or_isnt_file(config_path)
                return self._parse_config_file(config_path)
            
            else:
                raise TypeError(f'config_path should be a str|pathlib.Path but {type(config_path)} was given!')
        
        if config_dict is not None:
            if isinstance(config_dict, dict):
                return config_dict
            else:
                raise TypeError(f'config_dict should be a dict but {type(config_dict)} was given!')
        
        if config_str is not None:
            if isinstance(config_str, str):
                return self._parse_config_str(config_str)
            else:
                raise TypeError(f'config_str should be a str but {type(config_str)} was given!')

        #No input config
        return dict()

    def _raise_if_path_doesnt_exists_or_isnt_file(self, file_path: str):
        file_pathlib = pathlib.Path(file_path)
        if not (file_pathlib.exists() and file_pathlib.is_file()):
            raise FileNotFoundError(f"File at {file_path} doesn't exists!")
    
    def _parse_config_file(self, file_path:str) -> dict:
        raise NotImplementedError("_parse_config_file is not implemented!")

    def _parse_config_str(self, config_str:str) -> dict:
        raise NotImplementedError("_parse_config_str is not implemented!")

    def _treat_input_config(self, config_dict:dict) -> dict:
        """
        Does type casting on configs.
        """
        raise NotImplementedError("_treat_input_config is not implemented!")

    def _raise_if_not_valid_config(self, config_dict:dict):
        self._raise_if_alg_config_not_valid(config_dict)
        self._raise_if_wells_config_not_valid(config_dict)

    def _raise_if_alg_config_not_valid(self, config_dict:dict):
        alg_configs = config_dict['alg']
        
        starting_it = alg_configs['starting_it']
        if starting_it < 0:
            raise ValueError(f'alg.starting_it should be a positive integer but {starting_it} was given!')
        
        if alg_configs['num_its'] < 1:
            raise ValueError(f"alg.num_its should be at least 1 but {alg_configs['num_its']} was given!")
        
        if not self._has_positives_only(alg_configs['validation_only_wells']):
            raise ValueError("alg.validation_only_wells: well index can't be negative!")
        
        num_wells = len(config_dict['wells']['coords'])
        if not self._under_max_value_only(alg_configs['validation_only_wells'], num_wells):
            raise ValueError(f'alg.validation_only_wells: Invalid well index. Max: {num_wells-1}')
        
        if not self._has_positives_only(alg_configs['test_only_wells']):
            raise ValueError("alg.test_only_wells: well index can't be negative!")

        if not self._under_max_value_only(alg_configs['test_only_wells'], num_wells):
            raise ValueError(f'alg.test_only_wells: Invalid well index. Max: {num_wells-1}')

        try:
            #Just try to access a feature selection type
            FeatureSelection[alg_configs['feature_selection_type'].name]
        except:
            raise ValueError(f"alg.feature_selection_type should be one of {[type for type in FeatureSelection.__members__]}!")
        
        self._raise_if_beta_dist_params_not_valid(alg_configs['sampling'])
        
        if alg_configs['max_num_features'] < 0:
            raise ValueError(f"alg.max_num_features: Should be a positive integer but {alg_configs['max_num_features']} was given!")
        
        SaveModelTypes.raise_if_not_valid(alg_configs['save_models_on'])

    
    def _raise_if_wells_config_not_valid(self, config_dict:dict):
        wells_config = config_dict['wells']

        if len(wells_config['coords']) == 0:
            raise ValueError(f"wells.coords: There should be at least one well coords provided!")

        for index, well_dict in enumerate(wells_config['coords']):
            if well_dict['x'] < 0 or well_dict['y'] < 0:
                raise ValueError(f"wells.coords: Error at well {index} index: x and y should be non negative integers!")

    def _raise_if_beta_dist_params_not_valid(self, config_dict:dict):
        beta_dist_dict = config_dict['beta_dist']
        if beta_dist_dict['beta'] < 0:
            raise ValueError(f"alg.sampling.beta_dist.beta: Beta value cant be negative!")
        
        if beta_dist_dict['alpha'] < 0:
            raise ValueError(f"alg.sampling.beta_dist.alpha: Alpha value cant be negative!")
        
    def _has_positives_only(self, my_list:list) -> bool:
        return all([element > 0 for element in my_list])
    
    def _under_max_value_only(self, my_list:list, max_value:float) -> bool:
        return all([element < max_value for element in my_list])
    
    def _base_config(self) -> dict:
        base_configs = dict()
        base_configs['features_folder'] = "./features/"
        base_configs['porosity_cube_output_path'] = "./results/porosity.npy"
        base_configs['alg'] = self._base_alg_config()
        base_configs['wells'] = self._base_wells_config()
        return base_configs

    def _base_alg_config(self) -> dict:
        base_config = dict()
        base_config['starting_it'] = 0
        base_config['num_its'] = 10
        base_config['generate_porosity_cube'] = True
        base_config['save_cube_every_n_its'] = -1
        base_config['validation_only_wells'] = list()
        base_config['test_only_wells'] = list()
        base_config['sampling'] = self._base_sampling_config()
        base_config['feature_selection_type'] = FeatureSelection['FORWARD']
        base_config['max_num_features'] = 10
        base_config['max_exec_time'] = -1
        base_config['metrics_by_it'] = True
        base_config['save_models_on'] = 'last'
        return base_config
    
    def _base_sampling_config(self) -> dict:
        base_config = dict()
        base_config['its_window_size'] = -1
        base_config['max_points'] = -1
        base_config['beta_dist'] = self._base_penalty_sampling_func_config()
        return base_config
    
    def _base_penalty_sampling_func_config(self) -> dict:
        base_config = dict()
        base_config['alpha'] = 1
        base_config['beta'] = 1
        return base_config

    def _base_wells_config(self) -> dict:
        base_config = dict()
        base_config['porosity_path'] = ''
        base_config['coords'] = list()
        return base_config


    @property
    def wells(self):
        return self.config['wells']

    @wells.setter
    def wells(self, new_wells):
        raise AttributeError("wells config is read only!")

    @property
    def alg(self):
        return self.config['alg']
    
    @alg.setter
    def alg(self, new_alg):
        raise AttributeError("alg config is read only!")
    
    @property
    def features_folder(self):
        return self.config['features_folder']

    @features_folder.setter
    def features_folder(self, new_feat_folder):
        raise AttributeError("features_folder config is read only!")
    
    @property
    def por_cube_output_path(self):
        return self.config['porosity_cube_output_path']
    
    @por_cube_output_path.setter
    def por_cube_output_path(self, new_path):
        raise AttributeError("por_cube_output_path is read only!")
    
class YAMLConfig(Config):
    def __init__(self, config_path: str|pathlib.Path = None,
                 config_dict: dict = None, config_str: str = None):
        super().__init__(config_path, config_dict, config_str)
    
    def _parse_config_file(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader)
        
        return data
    
    def _parse_config_str(self, config_str: str) -> dict:
        return yaml.load(config_str, Loader)

    def _treat_input_config(self, config_dict: dict) -> dict:
        treated_dict = dict()
        if 'alg' in config_dict:
            treated_dict['alg'] = self._treat_alg_configs(config_dict['alg'])
        
        if 'wells' in config_dict:
            treated_dict['wells'] = self._treat_wells_configs(config_dict['wells'])
        
        return treated_dict
    
    def _treat_alg_configs(self, alg_configs:dict) -> dict:
        treated_alg_configs = dict()

        #So non treated/non expected keys remain in treated_alg_configs
        treated_alg_configs.update(alg_configs)

        key_func_to_apply_dict = {
            'starting_it': int,
            'num_its': int,
            'save_cube_every_n_its': int,
            'validation_only_wells': self._list_func_applier_decorator(int),
            'test_only_wells': self._list_func_applier_decorator(int),
            'max_num_features': int,
            'max_exec_time': int,
            'generate_porosity_cube': self._treat_boolean_from_yaml,
            'metrics_by_it': self._treat_boolean_from_yaml,
            'feature_selection_type': FeatureSelection.new_from_key
        }

        self._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map= key_func_to_apply_dict,
            base_dict=alg_configs,
            dict_to_modify=treated_alg_configs
        )
        
        if 'sampling' in alg_configs:
            treated_sampling_config = self._treat_sampling_configs(alg_configs['sampling'])
            treated_alg_configs['sampling'] = treated_sampling_config

        return treated_alg_configs
    
    def _treat_sampling_configs(self, sampling_configs: dict) -> dict:
        treated_sampling_config = dict()

        #So non treated/non expected keys remain in treated_sampling_config
        treated_sampling_config.update(sampling_configs)

        key_func_to_apply_dict = {
            'its_window_size': int,
            'max_points': int
        }

        self._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map= key_func_to_apply_dict,
            base_dict=sampling_configs,
            dict_to_modify=treated_sampling_config
        )

        if 'beta_dist' in sampling_configs:
            treated_beta_dist_configs = self._treat_beta_dist_configs(sampling_configs['beta_dist'])
            treated_sampling_config['beta_dist'] = treated_beta_dist_configs

        return treated_sampling_config
    
    def _treat_beta_dist_configs(self, beta_dist_configs:dict) -> dict:
        treated_beta_dist_configs = dict()

        #So non treated/non expected keys remain in treated_sampling_config
        treated_beta_dist_configs.update(beta_dist_configs)
        
        key_func_to_apply_dict = {
            'alpha': int,
            'beta': int
        }

        self._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map=key_func_to_apply_dict,
            base_dict=beta_dist_configs,
            dict_to_modify=treated_beta_dist_configs
        )

        return treated_beta_dist_configs

    def _treat_wells_configs(self, wells_configs:dict) -> dict:
        treated_wells_configs = dict()
        
        #So non treated/non expected keys remain in treated_sampling_config
        treated_wells_configs.update(wells_configs)

        if 'coords' in wells_configs:
            treated_coords_configs = self._treat_wells_coords_configs(wells_configs['coords'])
            treated_wells_configs['coords'] = treated_coords_configs
        
        return treated_wells_configs

    def _treat_wells_coords_configs(self, coords_configs: list) -> list:
        """
        Wells coords is a list composed of (x, y) tuples, [x, y] lists or {'x':value, 'y':value} dicts.
        They can all be present.
        """
        #So non treated elements remain in treated_coords_configs        
        treated_coords_configs = list(coords_configs)

        func_to_apply = self._list_func_applier_decorator(self._treats_every_coord)

        treated_coords_configs = func_to_apply(coords_configs)

        return treated_coords_configs
    
    def _treats_every_coord(self, coords: list|dict):
        """
        Assumes the coordnates are x and y integers
        """
        #its a list/tuple with two elements (x, y) or [x, y]
        if isinstance(coords, list) or isinstance(coords, tuple):
            if len(coords) != 2:
                raise ValueError(f"A coord must have 2 elements but {len(coords)} were given!")
            
            x, y = coords
            return {'x': int(x), 'y': int(y)}
        #its a dict with x and y keys
        elif isinstance(coords, dict):
            return {key:int(value) for key, value in coords.items()}
    
    def _treat_boolean_from_yaml(self, yaml_bool:str) -> bool:
        if yaml_bool.lower() in ['y', 'yes', 'on', 'true']:
            return True
        elif yaml_bool.lower() in ['n', 'no', 'off', 'false']:
            return False
        else:
            raise ValueError(f" {yaml_bool} not identified as a valid boolean!")

    def _apply_key_func_mapping_to_dict_and_modify_target_dict(self, key_func_map:dict,
                                                                base_dict:dict,
                                                                dict_to_modify:dict):
        """
        Apply the func associated with the key in base_dict but its result is saved in dict_to_modify
        """
        for key, func in key_func_map.items():
            self._apply_func_to_simple_key_if_present(base_dict = base_dict,
                                                      key = key,
                                                      func = func,
                                                      dict_to_modify = dict_to_modify)
    
    def _apply_func_to_simple_key_if_present(self, base_dict:dict, key, func, dict_to_modify:dict):
        if key in base_dict:
            dict_to_modify[key] = func(base_dict[key])
    
    def _list_func_applier_decorator(self, func):
        """
        This is a decorator. It returns a function that applies func to every element of a iterable.
        This exists to pass the 'underlying function' as a parameter to other functions.
        """
        def apply_func_to_every_element(my_iterable):
            """
            Applies a func to every element in my_iterable
            """
            treated_iterable = [func(element) for element in my_iterable]
            return treated_iterable
        return apply_func_to_every_element