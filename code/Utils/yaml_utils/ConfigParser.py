import yaml
import traceback

from Utils.yaml_utils.dict_as_dot_notation import Map


class ConfigParser(object):
    def __init__(self, config_path):
        self._config_path = config_path

    def parse(self):
        parsed = None
        with open(self._config_path, 'r') as stream:
            try:
                parsed = Map(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                traceback.print_exc(exc)

        return parsed
