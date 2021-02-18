# pytype: skip-file

from __future__ import absolute_import

import argparse
import os
from os import path
import logging

import data_prep_step as dp_package

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='input data gcs path')
    parser.add_argument('--output_data',  type=str, help='output data gcs path')

    args = parser.parse_args()

    print(args)

    fixed_output_dir = args.output_data + "/prefix"
    print("fixed output dir: {}".format(fixed_output_dir))

    pipeline_options = {
        "input" : args.input_data, 
        "output" : fixed_output_dir,
        "save_main_session" : True
    }

    dp_package.run_pipeline_component(pipeline_options)


