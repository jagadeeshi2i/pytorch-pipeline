import os
import sys
import re
import signal
import string
import logging
import json
from enum import Enum, auto
from types import FunctionType
import yaml
import random
from pathlib import Path
from time import sleep, strftime, gmtime
from abc import abstractmethod
from typing import Any, Type, Dict, List, NamedTuple, Optional

from .pytorch_component_spec import PytorchComponentSpec
from .pytorch_common_inputs import (
    PytorchComponentBaseOutputs,
    PytorchComponentBaseInputs
)

_component_decorator_handler: Optional[FunctionType] = None

def ComponentMetadata(name: str, description: str, spec: object):
    """Decorator for Pytorch components.
    Used to define necessary metadata attributes about the component which will
    be used for logging output and for the component specification file.
    Usage:
    ```python
    @ComponentMetadata(
        name="Pytorch - Component Name",
        description="A cool new component we made!",
        spec=MyComponentSpec
    )
    """

    def _component_metadata(cls):
        cls.COMPONENT_NAME = name
        cls.COMPONENT_DESCRIPTION = description
        cls.COMPONENT_SPEC = spec

        # Add handler for compiler
        if _component_decorator_handler:
            return _component_decorator_handler(cls) or cls
        return cls

    return _component_metadata


class PytorchComponent:
    """Base class for a KFP Pytorch component.
    An instance of a subclass of this component represents an instantiation of the
    component within a pipeline run. Use the `@ComponentMetadata` decorator to
    modify the component attributes listed below.
    Attributes:
        COMPONENT_NAME: The name of the component as displayed to the user.
        COMPONENT_DESCRIPTION: The description of the component as displayed to
            the user.
        COMPONENT_SPEC: The correspending spec associated with the component.
        STATUS_POLL_INTERVAL: Number of seconds between polling for the job
            status.
    """

    COMPONENT_NAME = ""
    COMPONENT_DESCRIPTION = ""
    COMPONENT_SPEC = PytorchComponentSpec

    STATUS_POLL_INTERVAL = 30

    def __init__(self):
        """Initialize a new component."""
        self._initialize_logging()

    def _initialize_logging(self):
        """Initializes the global logging structure."""
        logging.getLogger().setLevel(logging.INFO)

    def Do(
        self,
        inputs: PytorchComponentBaseInputs,
        outputs: PytorchComponentBaseOutputs,
        output_paths: PytorchComponentBaseOutputs,
    ):
        """The main execution entrypoint for a component at runtime.
        Args:
            inputs: A populated list of user inputs.
            outputs: An unpopulated list of component output variables.
            output_paths: Paths to the respective output locations.
        """
        self._do(inputs, outputs, output_paths)

    def _do(
        self,
        inputs: PytorchComponentBaseInputs,
        outputs: PytorchComponentBaseOutputs,
        output_paths: PytorchComponentBaseOutputs,
    ):

        self._run_pipeline_step(inputs, outputs)

    @abstractmethod
    def _run_pipeline_step(
        self,
        inputs: PytorchComponentBaseInputs,
        outputs: PytorchComponentBaseOutputs,
    ):
        """Creates the boto3 request object to execute the component.
        Args:
            inputs: A populated list of user inputs.
            outputs: An unpopulated list of component output variables.
        """
        pass