import json
import yaml

from distutils.util import strtobool
from argparse import ArgumentTypeError
from typing import List, Dict


class SpecInputParsers:
    """Utility class to define argparse validator methods."""

    @staticmethod
    def nullable_string_argument(value):
        """Strips strings and returns None if they are empty."""
        value = value.strip()
        if not value:
            return None
        return value

    @staticmethod
    def _yaml_or_json_str(value):
        if value == "" or value == None:
            return None
        try:
            return json.loads(value)
        except:
            return yaml.safe_load(value)

    @staticmethod
    def yaml_or_json_list(value):
        """Parses a YAML or JSON list to a Python list."""
        parsed = SpecInputParsers._yaml_or_json_str(value)
        if parsed is not None and not isinstance(parsed, List):
            raise ArgumentTypeError(f"{value} (type {type(value)}) is not a list")
        return parsed

    @staticmethod
    def yaml_or_json_dict(value):
        """Parses a YAML or JSON dictionary to a Python dictionary."""
        parsed = SpecInputParsers._yaml_or_json_str(value)
        if parsed is not None and not isinstance(parsed, Dict):
            raise ArgumentTypeError(f"{value} (type {type(value)}) is not a dictionary")
        return parsed

    @staticmethod
    def str_to_bool(value):
        """Converts a string interpretation of a boolean to a Python bool."""
        # This distutils function returns an integer representation of the boolean
        # rather than a True/False value. This simply hard casts it.
        return bool(strtobool(value))