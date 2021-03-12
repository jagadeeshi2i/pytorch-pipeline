import argparse

from typing import (
    Callable,
    Generic,
    List,
    TypeVar,
    Dict,
    Any,
    NewType,
    Optional,
    Union
)

from dataclasses import dataclass

@dataclass(frozen=True)
class PytorchComponentBaseInputs:
    """The base class for all component inputs."""

    pass


@dataclass
class PytorchComponentBaseOutputs:
    """A base class for all component outputs."""

    pass


@dataclass(frozen=True)
class PytorchComponentInputValidator:
    """Defines the structure of a component input to be used for validation."""

    input_type: Callable
    description: str
    required: bool = False
    choices: Optional[List[object]] = None
    default: Optional[object] = None

    def to_argparse_mapping(self):
        """Maps each property to an argparse argument.
        See: https://docs.python.org/3/library/argparse.html#adding-arguments
        """
        return {
            "type": self.input_type,
            "help": self.description,
            "required": self.required,
            "choices": self.choices,
            "default": self.default,
        }


@dataclass(frozen=True)
class PytorchComponentOutputValidator:
    """Defines the structure of a component output."""

    description: str


# The value of an input or output can be arbitrary.
# This could be replaced with a generic, as well.
PytorchIOValue = NewType("PytorchIOValue", object)

# Allow the component input to represent either the validator or the value itself
# This saves on having to rewrite the struct for both types.
# Could replace `object` with T if TypedDict supported generics (see above).
PytorchComponentInput = NewType(
    "PytorchComponentInput", Union[PytorchComponentInputValidator, PytorchIOValue]
)
PytorchComponentOutput = NewType(
    "PytorchComponentOutput",
    Union[PytorchComponentOutputValidator, PytorchIOValue],
)


