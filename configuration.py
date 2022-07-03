from abc import ABC, abstractmethod
from typing import Dict
import yaml
from copy import deepcopy
from logger import logger

NOT_IMPLEMENTED_ERROR_MSG: str = "Method must be implemented in concrete subclass."

class Configuration(ABC):

    @property
    def path_to_config(self) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG)

    @path_to_config.setter
    def path_to_config(self, path_to_config: str) -> None:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG)

    @property
    def configuration(self) -> Dict:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG)

    @configuration.setter
    def configuration(self, configuration: Dict) -> None:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG)

    @abstractmethod
    def export_configuration(self, export_path: str ) -> None:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG)


class BiPGCLSTMConfiguration(Configuration):
    def __init__(self, path_to_config: str = "config/bipgclstm_config.yaml") -> None:
        self.path_to_config: str = path_to_config

        try:
            with open(path_to_config, "r", encoding="utf-8") as f:
                self.configuration: Dict = yaml.load(f, yaml.Loader)

        except IOError as e:
            logger.error(e)
            raise IOError

    @property
    def path_to_config(self) -> str:
        return self._path_to_config

    @path_to_config.setter
    def path_to_config(self, path_to_config: str) -> None:
        self._path_to_config = path_to_config

    @property
    def configuration(self) -> Dict:
        return self._configuration

    @configuration.setter
    def configuration(self, configuration: Dict) -> None:
        self._configuration = configuration

    def export_configuration(self, export_path: str) -> None:
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                config_tmp: Dict = deepcopy(self.configuration)
                yaml.dump(config_tmp, f, yaml.Dumper)

        except (IOError, Exception) as e:
            logger.error(e)
            raise IOError

class RaoetalPGCLSTMConfiguration(Configuration):
    def __init__(self, path_to_config: str = "config/raoetalpgclstm_config.yaml") -> None:
        self.path_to_config: str = path_to_config

        try:
            with open(path_to_config, "r", encoding="utf-8") as f:
                self.configuration: Dict = yaml.load(f, yaml.Loader)

        except IOError as e:
            logger.error(e)
            raise IOError

    @property
    def path_to_config(self) -> str:
        return self._path_to_config

    @path_to_config.setter
    def path_to_config(self, path_to_config: str) -> None:
        self._path_to_config = path_to_config

    @property
    def configuration(self) -> Dict:
        return self._configuration

    @configuration.setter
    def configuration(self, configuration: Dict) -> None:
        self._configuration = configuration

    def export_configuration(self, export_path: str) -> None:
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                config_tmp: Dict = deepcopy(self.configuration)
                yaml.dump(config_tmp, f, yaml.Dumper)

        except (IOError, Exception) as e:
            logger.error(e)
            raise IOError


# TODO: TransformerPGC -> Use config from Deep Phonemizer Lib?