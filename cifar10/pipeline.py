import kfp
import json
import os
import copy
from kfp import components
from kfp import dsl
from kfp.aws import use_aws_secret
from kfp.components import load_component_from_file, load_component_from_url


cur_file_dir = os.path.dirname(__file__)
components_dir = os.path.join(cur_file_dir, "../pytorch")

cifar10_data_prep_op = components.load_component_from_file(
    components_dir + "/data_prep/component.yaml"
)

cifar10_train_op = components.load_component_from_file(
    components_dir + "/train/component.yaml"
)


@dsl.pipeline(name="Training pipeline", description="Sample training job test")
def training(input_directory = "/pvc/input",
    output_directory = "/pvc/output",  handlerFile = "image_classifier"):

    vop = dsl.VolumeOp(
        name="volume_creation",
        resource_name="pvcm",
        modes=dsl.VOLUME_MODE_RWO,
        size="1Gi"
    )

    prep_output = cifar10_data_prep_op(
        input_data =
            [],
        container_entrypoint = [
            "python",
            "/pvc/input/cifar10_pre_process.py",
        ],
        output_data = ["/pvc/output/processing"],
        source_code = ["https://kubeflow-dataset.s3.us-east-2.amazonaws.com/cifar10_pre_process.py"],
        source_code_path = ["/pvc/input"]
    ).add_pvolumes({"/pvc":vop.volume})

    train_output = cifar10_train_op(
        input_data = ["/pvc/output/processing"],
        container_entrypoint = [
            "python",
            "/pvc/input/cifar10_train.py",
        ],
        output_data = ["/pvc/output/train/models"],
        input_parameters = [{"tensorboard_root": "/pvc/output/train/tensorboard", 
        "max_epochs": 1, "gpus": 0, "train_batch_size": None, "val_batch_size": None, "train_num_workers": 4, 
        "val_num_workers": 4 , "learning_rate": 0.001, 
        "accelerator": None}],
        source_code = ["https://kubeflow-dataset.s3.us-east-2.amazonaws.com/cifar10_train.py"],
        source_code_path = ["/pvc/input"]
    ).add_pvolumes({"/pvc":vop.volume}).after(prep_output)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(training, package_path="pytorch_cifar10.yaml")