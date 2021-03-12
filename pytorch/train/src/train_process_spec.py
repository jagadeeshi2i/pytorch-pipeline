from dataclasses import dataclass
from typing import List

from common.pytorch_common_inputs import PytorchComponentInput as Input
from common.pytorch_common_inputs import \
    PytorchComponentInputValidator as InputValidator
from common.pytorch_common_inputs import PytorchComponentOutput as Output
from common.pytorch_common_inputs import \
    PytorchComponentOutputValidator as OutputValidator
from common.pytorch_component_spec import PytorchComponentSpec
from common.pytorch_spec_input_parser import SpecInputParsers


@dataclass(frozen=True)
class PytorchTrainInputs():
    """Defines the set of inputs for the process component."""

    container_entrypoint: Input
    input_data: Input
    output_data: Input
    input_parameters: Input
    source_code: Input
    source_code_path: Input


@dataclass
class PytorchTrainOutputs():
    """Definesinput_data the set of outputs for the process component."""

    output_artifacts: Output


class PytorchTrainSpec(
    PytorchComponentSpec[PytorchTrainInputs, PytorchTrainOutputs]
):

    INPUTS = PytorchTrainInputs(
        container_entrypoint=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="The entrypoint for the training job. This is in the form of a list of strings that make a command.",
            default=[],
        ),
        input_data=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Parameters that specify inputs for a training job.",
            default=[],
        ),
        output_data=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Parameters that output path for a training job.",
            default=[],
        ),
        input_parameters=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Parameters that specify inputs for a training job.",
            default=[],
        ),
        source_code=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Training source code path",
            default=[],
        ),
        source_code_path=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Training source code path for internal storage",
            default=[],
        )
    )

    OUTPUTS = PytorchTrainOutputs(
        output_artifacts=OutputValidator(
            description="A dictionary containing the output S3 artifacts."
        ),
    )

    def __init__(self, arguments: List[str]):
        super().__init__(arguments, PytorchTrainInputs, PytorchTrainOutputs)

    @property
    def inputs(self) -> PytorchTrainInputs:
        return self._inputs

    @property
    def outputs(self) -> PytorchTrainOutputs:
        return self._outputs

    @property
    def output_paths(self) -> PytorchTrainOutputs:
        return self._output_paths
