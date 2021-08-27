# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A script to export RN50 QAT saved model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf
from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_model
from quantize import quantize_model

def export_quantized_rn50(args):

    # Set data format to NCHW
    tf.keras.backend.set_image_data_format('channels_first')

    model = resnet_model.resnet50(num_classes=1001, rescale_inputs=False)
    # Introduce QDQ layers around convolution/FC inputs and weights
    model = quantize_model(model)

    checkpoint = tf.train.Checkpoint(model=model)
    # expect_partial is called to avoid warnings from optimizer variables in the checkpoint.
    checkpoint.restore(args.ckpt).expect_partial()

    tf.saved_model.save(model, args.output)

    print("Saved the quantized model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to finetuned RN50 QAT checkpoint")
    parser.add_argument("--output", type=str,  default='rn50_qat_saved_model', help="output path to RN50 saved model")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to enable verbose logging")
    args = parser.parse_args()
    export_quantized_rn50(args)
