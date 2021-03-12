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
class PytorchProcessInputs():
    """Defines the set of inputs for the process component."""

    container_entrypoint: Input
    input_data: Input
    output_data: Input
    source_code: Input
    source_code_path: Input


@dataclass
class PytorchProcessOutputs():
    """Definesinput_data the set of outputs for the process component."""

    output_artifacts: Output


class PytorchProcessSpec(
    PytorchComponentSpec[PytorchProcessInputs, PytorchProcessOutputs]
):

    INPUTS = PytorchProcessInputs(
        container_entrypoint=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="The entrypoint for the processing job. This is in the form of a list of strings that make a command.",
            default=[],
        ),
        input_data=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Parameters that specify inputs for a processing job.",
            default=[],
        ),
        output_data=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Parameters that specify inputs for a processing job.",
            default=[],
        ),
        source_code=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Data prep source code path",
            default=[],
        ),
        source_code_path=InputValidator(
            input_type=SpecInputParsers.yaml_or_json_list,
            required=False,
            description="Data prep source code path for internal storage",
            default=[],
        )
    )

    OUTPUTS = PytorchProcessOutputs(
        output_artifacts=OutputValidator(
            description="A dictionary containing the output S3 artifacts."
        ),
    )

    def __init__(self, arguments: List[str]):
        super().__init__(arguments, PytorchProcessInputs, PytorchProcessOutputs)

    @property
    def inputs(self) -> PytorchProcessInputs:
        return self._inputs

    @property
    def outputs(self) -> PytorchProcessOutputs:
        return self._outputs

    @property
    def output_paths(self) -> PytorchProcessOutputs:
        return self._output_paths
