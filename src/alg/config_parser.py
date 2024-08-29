import collections.abc
from typing import List, Tuple
import pathlib
import enum

import yaml
try:
    from yaml import CBaseLoader as Loader
except ImportError:
    from yaml import BaseLoader as Loader


class InvalidNewParamError(Exception):
    pass


class FeatureSelection(enum.Enum):
    FORWARD = 0
    NONE = 1

    @staticmethod
    def new_from_key(key):
        try:
            new_feat_selection = FeatureSelection[key]
            return new_feat_selection
        except:
            error_msg = (f"Key error for FeatureSelectionType. "
                         f"Got {key} but should be one of "
                         f"{[key for key in FeatureSelection.__members__]}")
            raise KeyError(error_msg)


class SaveModelTypes:
    ALL = "all"
    LAST = "last"

    @staticmethod
    def _has_positives_only(my_list: list) -> bool:
        return all([element >= 0 for element in my_list])

    @staticmethod
    def raise_if_invalid(save_model_type):
        if isinstance(save_model_type, list):
            if not SaveModelTypes._has_positives_only(save_model_type):
                raise ValueError("alg.save_models_on: The list should have "
                                 "positive integers only!")
        elif isinstance(save_model_type, str):
            valid_strs = [SaveModelTypes.ALL, SaveModelTypes.LAST]
            if save_model_type[:6] == "every_":
                try:
                    n = int(save_model_type[6:])
                    if n == 0:
                        raise
                except:
                    raise ValueError("alg.save_models_on: 'n' "
                                     "should be a positive integer!")
            elif save_model_type not in valid_strs:
                raise ValueError(f"alg.save_models_on: This str is not valid. "
                                 f"Should be one of {valid_strs} but "
                                 f"'{save_model_type}' was given!")


def list_func_applier_decorator(func):
    """
    This is a decorator. It returns a function that applies func to every
    element of a iterable. This exists to pass the 'underlying function'
    as a parameter to other functions.
    """

    def apply_func_to_every_element(my_iterable):
        """
        Applies a func to every element in my_iterable
        """
        treated_iterable = [func(element) for element in my_iterable]
        return treated_iterable

    return apply_func_to_every_element


class ConfigTypeCaster:
    """
    This class acts as a type casting/checker for the user's input config
    provided. This is mostly file format independent, but could have
    differences between the formats accepted.
    """

    @classmethod
    def treat_input_config(cls, config_dict: dict) -> dict:
        treated_dict = dict()

        treated_dict.update(config_dict)
        if "alg" in config_dict:
            treated_dict["alg"] = cls._type_cast_alg_configs(config_dict["alg"])

        if "wells" in config_dict:
            treated_dict["wells"] = cls._treat_wells_configs(
                config_dict["wells"])

        return treated_dict

    @classmethod
    def _type_cast_alg_configs(cls, alg_configs: dict) -> dict:
        treated_alg_configs = dict()

        # So non treated/non expected keys remain in treated_alg_configs
        treated_alg_configs.update(alg_configs)

        key_func_to_apply_dict = {
            "starting_it": int,
            "num_its": int,
            "save_cube_every_n_its": int,
            "validation_only_wells": list_func_applier_decorator(int),
            "test_only_wells": list_func_applier_decorator(int),
            "max_num_features": int,
            "window": int,
            "layers_to_predict": int,
            "max_exec_time": int,
            "generate_porosity_cube": cls._which_python_bool_value,
            "metrics_by_it": cls._which_python_bool_value,
            "feature_selection_type": FeatureSelection.new_from_key,
        }

        cls._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map=key_func_to_apply_dict,
            base_dict=alg_configs,
            dict_to_modify=treated_alg_configs,
        )

        if "sampling" in alg_configs:
            treated_sampling_config = cls._type_cast_sampling_configs(
                alg_configs["sampling"])
            treated_alg_configs["sampling"] = treated_sampling_config

        if "parallel" in alg_configs:
            treated_parallel_config = cls._type_cast_parallel_configs(
                alg_configs["parallel"])
            treated_alg_configs["parallel"] = treated_parallel_config

        return treated_alg_configs

    @classmethod
    def _type_cast_sampling_configs(cls, sampling_configs: dict) -> dict:
        treated_sampling_config = dict()

        # So non treated/non expected keys remain in treated_sampling_config
        treated_sampling_config.update(sampling_configs)

        key_func_to_apply_dict = {
            "layers_window_size": int,
            "max_points": int,
            "seed": int
        }

        cls._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map=key_func_to_apply_dict,
            base_dict=sampling_configs,
            dict_to_modify=treated_sampling_config,
        )

        if "beta_dist" in sampling_configs:
            treated_beta_dist_configs = cls._type_cast_beta_dist_configs(
                sampling_configs["beta_dist"])
            treated_sampling_config["beta_dist"] = treated_beta_dist_configs

        return treated_sampling_config

    @classmethod
    def _type_cast_parallel_configs(cls, parallel_configs: dict) -> dict:
        treated_parallel_config = dict()

        # So non treated/non expected keys remain in treated_sampling_config
        treated_parallel_config.update(parallel_configs)

        key_func_to_apply_dict = {"max_points_per_chunk": int}

        cls._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map=key_func_to_apply_dict,
            base_dict=parallel_configs,
            dict_to_modify=treated_parallel_config,
        )

        return treated_parallel_config

    @classmethod
    def _type_cast_beta_dist_configs(cls, beta_dist_configs: dict) -> dict:
        treated_beta_dist_configs = dict()

        # So non treated/non expected keys remain in treated_sampling_config
        treated_beta_dist_configs.update(beta_dist_configs)

        key_func_to_apply_dict = {"alpha": int, "beta": int}

        cls._apply_key_func_mapping_to_dict_and_modify_target_dict(
            key_func_map=key_func_to_apply_dict,
            base_dict=beta_dist_configs,
            dict_to_modify=treated_beta_dist_configs,
        )

        return treated_beta_dist_configs

    @classmethod
    def _treat_wells_configs(cls, wells_configs: dict) -> dict:
        treated_wells_configs = dict()

        # So untreated/unexpected keys remain in treated_sampling_config
        treated_wells_configs.update(wells_configs)

        assert 'window' in wells_configs, "[config_parser] Missing "\
                "wells.window' configuration."

        treated_wells_configs['window'] = int(wells_configs["window"])

        if "coords" in wells_configs:
            treated_coords_configs = cls._treat_wells_coords_configs(
                wells_configs["coords"])
            treated_wells_configs["coords"] = treated_coords_configs

        return treated_wells_configs

    @classmethod
    def _treat_wells_coords_configs(cls, coords_configs: list) -> list:
        """
        Wells coords is a list composed of (x, y) tuples, [x, y] lists or
        {'x':value, 'y':value} dicts. They can all be present.
        """

        # # So non treated elements remain in treated_coords_configs
        # treated_coords_configs = list(coords_configs)
        # func_to_apply = list_func_applier_decorator(cls._treats_every_coord)
        # treated_coords_configs = func_to_apply(coords_configs)

        treated_coords_configs = [
            cls._treats_every_coord(c) for c in coords_configs
        ]

        return treated_coords_configs

    @classmethod
    def _treats_every_coord(cls, coords):
        """
        Assumes the coordnates are x and y integers.
        The window displacement is applied here.
        """
        # its a list/tuple with two elements (x, y) or [x, y]
        if isinstance(coords, list) or isinstance(coords, tuple):
            if len(coords) != 2:
                raise ValueError(
                    f"A coord must have 2 elements but {len(coords)} were given!"
                )

            x, y = coords
            return {"x": int(x), "y": int(y)}
        # its a dict with x and y keys
        elif isinstance(coords, dict):
            return {key: int(value) for key, value in coords.items()}

    @staticmethod
    def _which_python_bool_value(yaml_bool: str) -> bool:
        raise NotImplementedError(
            "Not implemented! This should be file format dependent!")

    @classmethod
    def _apply_key_func_mapping_to_dict_and_modify_target_dict(
            cls, key_func_map: dict, base_dict: dict, dict_to_modify: dict):
        """
        Apply the func associated with the key in base_dict but its result is
        saved in dict_to_modify
        """
        for key, func in key_func_map.items():
            cls._apply_func_to_simple_key_if_present(
                base_dict=base_dict,
                key=key,
                func=func,
                dict_to_modify=dict_to_modify,
            )

    @staticmethod
    def _apply_func_to_simple_key_if_present(base_dict: dict, key, func,
                                             dict_to_modify: dict):
        if key in base_dict:
            dict_to_modify[key] = func(base_dict[key])


class YAMLConfigTypeCaster(ConfigTypeCaster):

    @staticmethod
    def _which_python_bool_value(input_bool: str) -> bool:
        if input_bool.lower() in ["y", "yes", "on", "true"]:
            return True
        elif input_bool.lower() in ["n", "no", "off", "false"]:
            return False
        else:
            raise ValueError(
                f" {input_bool} not identified as a valid yaml boolean!")


class ConfigValidator:
    """
    This class acts as a value validator for the user's input config. Even if
    the config passes the type validation, some values may not be valid for
    a given config. This is file format independent.
    """

    @classmethod
    def raise_if_invalid_config(cls, config_dict: dict):
        cls._raise_if_alg_config_invalid(config_dict)
        cls._raise_if_wells_config_invalid(config_dict["wells"])

    @classmethod
    def _raise_if_alg_config_invalid(cls, config_dict: dict):
        alg_configs = config_dict["alg"]
        if alg_configs["starting_it"] < 0:
            # Mudar mensagem para must be an integer
            raise ValueError(
                f"alg.starting_it must be a non negative integer! {alg_configs['starting_it']} was given!"
            )

        if alg_configs["num_its"] < 1:
            raise ValueError(
                f"alg.num_its must be at least 1! {alg_configs['num_its']} was given!"
            )

        if not cls._has_positives_only(alg_configs["validation_only_wells"]):
            raise ValueError(
                "alg.validation_only_wells: well index can't be negative!")

        num_wells = len(config_dict["wells"]["coords"])
        if not cls._under_max_value_only(alg_configs["validation_only_wells"],
                                         num_wells):
            raise ValueError(
                f"alg.validation_only_wells: Invalid well index. Max: {num_wells-1}"
            )

        if not cls._has_positives_only(alg_configs["test_only_wells"]):
            raise ValueError(
                "alg.test_only_wells: well index can't be negative!")

        if not cls._under_max_value_only(alg_configs["test_only_wells"],
                                         num_wells):
            raise ValueError(
                f"alg.test_only_wells: Invalid well index. Max: {num_wells-1}")

        try:
            # Just try to access a feature selection type
            FeatureSelection[alg_configs["feature_selection_type"].name]
        except:
            raise ValueError(
                f"alg.feature_selection_type must be one of {[type for type in FeatureSelection.__members__]}!"
            )

        cls._raise_if_sampling_params_invalid(alg_configs["sampling"])

        if alg_configs["max_num_features"] <= 0:
            raise ValueError(
                f"alg.max_num_features must be a positive integer! {alg_configs['max_num_features']} was given!"
            )

        if alg_configs["window"] < 0:
            raise ValueError(
                f"alg.window must be a non negative integer! {alg_configs['window']} was given!"
            )

        if alg_configs["layers_to_predict"] < 1:
            raise ValueError(
                f"alg.layers_to_predict must be a positive integer! {alg_configs['layers_to_predict']} was given!"
            )

        SaveModelTypes.raise_if_invalid(alg_configs["save_models_on"])

        if alg_configs['parallel']['max_points_per_chunk'] == 0:
            raise ValueError(
                f"alg.parallel.max_points_per_chunk: Can't be zero!")

    @staticmethod
    def _raise_if_wells_config_invalid(wells_config: dict):

        if len(wells_config["coords"]) == 0:
            raise ValueError(
                f"wells.coords: There should be at least one well coords provided!"
            )

        for index, well_dict in enumerate(wells_config["coords"]):
            if well_dict["x"] < 0 or well_dict["y"] < 0:
                raise ValueError(
                    f"wells.coords: Error at well {index} index: x and y should be non negative integers!"
                )

    @staticmethod
    def _raise_if_sampling_params_invalid(samp_config_dict: dict):
        ConfigValidator._raise_if_beta_dist_params_invalid(samp_config_dict)

        if samp_config_dict['seed'] < 0:
            raise ValueError(
                f"alg.sampling.seed: Seed value can't be negative!")

    @staticmethod
    def _raise_if_beta_dist_params_invalid(samp_config_dict: dict):
        beta_dist_dict = samp_config_dict["beta_dist"]
        if beta_dist_dict["beta"] < 0:
            raise ValueError(
                f"alg.sampling.beta_dist.beta: Beta value cant be negative!")

        if beta_dist_dict["alpha"] < 0:
            raise ValueError(
                f"alg.sampling.beta_dist.alpha: Alpha value cant be negative!")

    @staticmethod
    def _has_positives_only(my_list: list) -> bool:
        return all([element >= 0 for element in my_list])

    @staticmethod
    def _under_max_value_only(my_list: list, max_value: float) -> bool:
        return all([element < max_value for element in my_list])


class Config:
    """
    This is a config parser.
    If config_path is provided, all other params are ignored and the config will be read from the file.
    If config_path is not provided, it will try to read config_dict. If it is not None, config_str will be ignored.
    If config_path and config_dict are both None, it will try to parse the config_str.
    If all parameters are None, then it will have the default configuration.
    """

    TOP_LEVEL_BASE_CONFIGS = [
        "features_folder",
        "porosity_cube_output_path",
        "starting_porosity_cube_path",
        "alg",
        "wells",
    ]

    def __init__(self,
                 config_path: str = None,
                 config_dict: dict = None,
                 config_str: str = None):
        input_config = self._get_input_config(config_path, config_dict,
                                              config_str)

        input_config = self._treat_input_config(input_config)

        self.config = self._base_config()

        self.config = self.update_recursivelly(self.config, input_config)

        ConfigValidator.raise_if_invalid_config(self.config)

    def _treat_input_config(self, config: dict) -> dict:
        config_type_caster = self._get_config_type_caster()
        return config_type_caster.treat_input_config(config)

    @classmethod
    def _get_config_type_caster(cls) -> ConfigTypeCaster:
        raise NotImplementedError(
            "Not implemented! This should be file type dependent!")

    def update_recursivelly(self, target, input):
        """
        https://stackoverflow.com/a/3233356/16264901
        """
        for k, v in input.items():
            if isinstance(v, collections.abc.Mapping):
                target[k] = self.update_recursivelly(target.get(k, {}), v)
            else:
                target[k] = v
        return target

    def _get_input_config(self,
                          config_path: str = None,
                          config_dict: dict = None,
                          config_str: str = None):
        """
        Tries to choose where are the input configs. If every param is none, return a empty dict.
        """

        if config_path is not None:
            if isinstance(config_path, str) or isinstance(
                    config_path, pathlib.Path):
                self._raise_if_path_doesnt_exists_or_isnt_file(config_path)
                return self._parse_config_file(config_path)

            else:
                raise TypeError(
                    f"config_path should be a str|pathlib.Path but {type(config_path)} was given!"
                )

        if config_dict is not None:
            if isinstance(config_dict, dict):
                return config_dict
            else:
                raise TypeError(
                    f"config_dict should be a dict but {type(config_dict)} was given!"
                )

        if config_str is not None:
            if isinstance(config_str, str):
                return self._parse_config_str(config_str)
            else:
                raise TypeError(
                    f"config_str should be a str but {type(config_str)} was given!"
                )

        # No input config
        return dict()

    def _raise_if_path_doesnt_exists_or_isnt_file(self, file_path: str):
        file_pathlib = pathlib.Path(file_path)
        if not (file_pathlib.exists() and file_pathlib.is_file()):
            raise FileNotFoundError(f"File at {file_path} doesn't exists!")

    def _parse_config_file(self, file_path: str) -> dict:
        raise NotImplementedError("_parse_config_file is not implemented!")

    def _parse_config_str(self, config_str: str) -> dict:
        raise NotImplementedError("_parse_config_str is not implemented!")

    def _base_config(self) -> dict:
        base_configs = dict()
        base_configs["features_folder"] = "./features/"
        base_configs["porosity_cube_output_path"] = "./results/porosity.npy"
        base_configs["starting_porosity_cube_path"] = "nope..."
        base_configs["alg"] = self._base_alg_config()
        base_configs["wells"] = self._base_wells_config()
        return base_configs

    def _base_alg_config(self) -> dict:
        base_config = dict()
        base_config["starting_it"] = 0
        base_config["num_its"] = 10
        base_config["generate_porosity_cube"] = True
        base_config["save_cube_every_n_its"] = -1
        base_config["validation_only_wells"] = list()
        base_config["test_only_wells"] = list()
        base_config["sampling"] = self._base_sampling_config()
        base_config["parallel"] = self._base_parallel_config()
        base_config["feature_selection_type"] = FeatureSelection["FORWARD"]
        base_config["max_num_features"] = 1
        base_config["window"] = 3
        base_config["layers_to_predict"] = 1
        base_config["max_exec_time"] = -1
        base_config["metrics_by_it"] = True
        base_config["save_models_on"] = "last"
        return base_config

    def _base_sampling_config(self) -> dict:
        base_config = dict()
        base_config["layers_window_size"] = -1
        base_config["max_points"] = -1
        base_config["seed"] = 42
        base_config["beta_dist"] = self._base_penalty_sampling_func_config()
        return base_config

    def _base_parallel_config(self) -> dict:
        base_config = dict()
        base_config["max_points_per_chunk"] = 10000
        return base_config

    def _base_penalty_sampling_func_config(self) -> dict:
        base_config = dict()
        base_config["alpha"] = 1
        base_config["beta"] = 1
        return base_config

    def _base_wells_config(self) -> dict:
        base_config = dict()
        base_config["coords"] = list()
        #as wells.window is required, we dont have to set it a base value
        return base_config

    def add_param(self, param_name: str, param_value):
        """
        This method's goal is to add params to the config object other than
        the base ones.
        """
        if param_name not in Config.TOP_LEVEL_BASE_CONFIGS:
            self.config[param_name] = param_value
        else:
            raise InvalidNewParamError(
                "This new param's name is equal to a base param!")

    def remove_param(self, param_name: str):
        """
        This method's goal is to remove params from the config object other
        than the base ones.
        """
        if param_name not in Config.TOP_LEVEL_BASE_CONFIGS:
            del self.config[param_name]
        else:
            raise InvalidNewParamError(
                "This new param's name is equal to a base param!")

    def get_param(self, param_name: str):
        """
        Returns None if the param is not found
        """
        return self.config.get(param_name, None)

    def ring_range_to_expand(self, it: int) -> Tuple[int, int]:
        """
        Returns the exact ring range [start, end] to expand/predict based on
        the it and the num of layers we must expand/predict on each iteration.
        This assumes that iterations starts at 0
        Example:
        it: 3
        layers_to_predict: 3
        return (7, 9)
        """
        start_ring = it * self.get_param('alg')['layers_to_predict']
        end_ring = start_ring + self.get_param('alg')['layers_to_predict'] - 1
        return start_ring, end_ring

    def get_coords_of_target_wells_ids(self, wells_ids: list) -> List[Tuple]:
        """
        Returns the coords of the target wells ids in wells_ids
        """
        coords = list()
        for well_id in wells_ids:
            curr_well_coords_dict = self.config["wells"]["coords"][well_id]
            coords.append(
                (curr_well_coords_dict['x'], curr_well_coords_dict['y']))

        return coords

    @property
    def wells(self):
        return self.config["wells"]

    @wells.setter
    def wells(self, new_wells):
        raise AttributeError("wells config is read only!")

    @property
    def wells_as_simple_list(self) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples with the wells coords:
        [(1, 2),(3, 4),(5, 6)...]
        """
        return self.get_coords_of_target_wells_ids(range(len(self.wells)+1))

    @property
    def train_wells_coords(self) -> List[Tuple[int, int]]:
        """
        Return the coords of the training wells.
        all wells: [(1, 2), (3, 4), (5, 6)]
        test_only_wells: [1]
        returns: [(1, 2), (5, 6)]
        """
        return [(w['x'], w['y'])
                for id, w in enumerate(self.config['wells']['coords'])
                if id not in self.alg['test_only_wells']]

    @property
    def train_wells_ids(self) -> List[int]:
        """
        Returns the training wells ids.
        Example: 
        all wells: [(1, 2), (3, 4), (5, 6)]
        test_only_wells: [1]
        returns: [0, 2]
        """
        return [
            id for id, _ in enumerate(self.config['wells']['coords'])
            if id not in self.alg['test_only_wells']
        ]

    @property
    def test_wells_coords(self) -> List[Tuple[int, int]]:
        """
        Return the coords of the test wells.
        all wells: [(1, 2), (3, 4), (5, 6)]
        test_only_wells: [1]
        returns: [(3, 4)]
        """
        return [(w['x'], w['y'])
                for id, w in enumerate(self.config['wells']['coords'])
                if id in self.alg['test_only_wells']]

    @property
    def test_wells_ids(self) -> List[int]:
        """
        Returns the training wells ids.
        Example: 
        all wells: [(1, 2), (3, 4), (5, 6)]
        test_only_wells: [1]
        returns: [1]
        """
        return list(self.alg['test_only_wells'])

    @property
    def alg(self):
        return self.config["alg"]

    @alg.setter
    def alg(self, new_alg):
        raise AttributeError("alg config is read only!")

    @property
    def features_folder(self):
        return self.config["features_folder"]

    @features_folder.setter
    def features_folder(self, new_feat_folder: str):
        self.config["features_folder"] = new_feat_folder

    @property
    def starting_porosity_cube_path(self):
        return self.config["starting_porosity_cube_path"]

    @starting_porosity_cube_path.setter
    def starting_porosity_cube_path(self, new_feat_folder: str):
        self.config["starting_porosity_cube_path"] = new_feat_folder

    @property
    def por_cube_output_path(self):
        """
        Porosity cube output path
        """
        return self.config["porosity_cube_output_path"]

    @por_cube_output_path.setter
    def por_cube_output_path(self, new_path):
        raise AttributeError("por_cube_output_path is read only!")

    @property
    def features_files_paths(self):
        """
        Returns a list with the complete path to every feature in the feature folder
        """
        features_paths = self._get_feat_files_paths()
        complete_paths = [path.absolute() for path in features_paths]
        return complete_paths

    @property
    def features_files_names(self):
        """
        Returns a list with the name of every feature in the feature folder.
        A feature name is equal to the name of its file without the suffix.
        Example: feature1.h5 -> name:feature1
        """
        features_paths = self._get_feat_files_paths()
        features_names = [path.stem for path in features_paths]
        return features_names

    def _get_feat_files_paths(self) -> List[pathlib.Path]:
        features_folder_path = pathlib.Path(self.config["features_folder"])
        features_paths = sorted([
            path for path in list(features_folder_path.glob("*"))
            if path.is_file()
        ])
        return features_paths

    def __str__(self):
        return str(self.config)


class YAMLConfig(Config):

    def __init__(self,
                 config_path: str = None,
                 config_dict: dict = None,
                 config_str: str = None):
        super().__init__(config_path, config_dict, config_str)

    def _parse_config_file(self, file_path: str) -> dict:
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader)

        return data

    def _parse_config_str(self, config_str: str) -> dict:
        return yaml.load(config_str, Loader)

    @classmethod
    def _get_config_type_caster(cls) -> ConfigTypeCaster:
        return YAMLConfigTypeCaster
