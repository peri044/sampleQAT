# TensorRT inference of Resnet-50 trained with QAT.

**Table Of Contents**

- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
  * [Step 1: Quantization Aware Training](#step-1-quantization-aware-training)
  * [Step 2: Export a RN50 QAT saved model](#step-2-export-a-rn50-qat-saved-model)
  * [Step 3: Conversion to ONNX](#step-3-conversion-to-onnx)
  * [Step 4: Conversion to TensorRT engine](#step-6-conversion-to-tensorrt-engine)
  * [Step 5: TensorRT Inference](#step-5-tensorrt-inference)
- [Additional resources](#additional-resources)
- [Changelog](#changelog)
- [Known issues](#known-issues)
- [License](#license)



## Description

This sample demonstrates workflow for training and inference of Resnet-50 model trained using Quantization Aware Training in Tensorflow 2.X from Tensorflow model garden. The inference uses TensorRT SDK which performs additional optimizations to QAT models to run in INT8 precision.

## Pre-requisites

* Install the necessary requirements

```
pip install -r requirements.txt
```

* Install TF2ONNX from <a href="https://github.com/peri044/tensorflow-onnx/tree/qdq_changes">this branch</a>. This branch has the changes to ensure we remove redundant transpose nodes at weighted layers and disable constant folding to avoid pruning `QuantizeAndDequantize` nodes at weights of conv/FC layers. A pull request to upstream is in progress.

* Clone the models from Tensorflow model garden.

```sh
git clone https://github.com/tensorflow/models.git
git checkout tag/2.6.0
cd models
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Running the sample

### Step 1: Quantization Aware Training

Please follow detailed instructions on how to <a href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quantization-aware-training">finetune a RN50 model using QAT</a>.

This stage involves

* Finetune a RN50 model with quantization nodes and save the final checkpoint.
* Post process the above RN50 QAT checkpoint by reshaping the weights of final FC layer into a 1x1 conv layer.

### Step 2 : Export a RN50 QAT saved model

Once you've finetuned the QAT model, export it by running

```python
python export_rn50_qat.py --ckpt <path_to_ckpt> --output <path_to_saved_model>
```

This script applies quantization to the model, restores the checkpoint and exports it in a saved_model format. This script will generate `rn50_qdq_step_45k_acc_76.39_regen_new`  which is a directory containing saved model.

Arguments:

* `--ckpt` : Path to finetuned QAT checkpoint to be loaded.
* `--output` : Name of output TF saved model

### Step 3 :  Conversion to ONNX

Convert the saved model into ONNX by running

```python
python -m tf2onnx.convert --saved-model <path_to_saved_model> --output rn50_qat.onnx  --opset 13 --disable_constfold
```

By default, tf2onnx uses TF's graph optimizers to performs constant folding after a saved model is loaded. `--disable_constfold` is necessary to disable constant folding of `QuantizeAndDequantize` nodes around weights of convolutional/fully connected layers.

Arguments:

* `--saved-model` : Name of TF saved-model
* `--output` : Name of ONNX output graph
* `--opset` : ONNX opset version (opset 13 or higher must be used)
* `--disable_constfold` : This flag disables constant folding performed by Tensorflow's grappler optimizer.

### Step 4 : Conversion to TensorRT Engine

Convert the ONNX model into TRT and save the engine

```
python build_engine.py --onnx rn50_qat.onnx -v
```

Arguments:

* `--onnx` : Path to RN50 QAT onnx graph
* `--engine` : Output file name of TensorRT engine.
* `-v, --verbose` : Flag to enable verbose logging

### Step 5: TensorRT Inference

Command to run inference on a sample image

```
python infer.py --engine <input_trt_engine>
```

Arguments:

* `--engine` : Path to input RN50 TensorRT engine.
* `--labels` : Path to imagenet 1k labels text file provided.
* `--image` : Path to the sample image
* `--verbose` : Flag to enable verbose logging

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:

```
usage: <python <filename>.py> [-h]
```

# Additional resources

The following resources provide a deeper understanding about Quantization aware training, TF2ONNX and importing a model into TensorRT using Python:

**Quantization Aware Training**

* <a href="https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/">Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT</a>

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)
- [Quantization Aware Training guide](https://www.tensorflow.org/model_optimization/guide/quantization/training)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**Parsers**

- [TF2ONNX Converter](https://github.com/onnx/tensorflow-onnx)
- [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**Documentation**

- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# Changelog

June 2020: Initial release of this sample based on TF 1.15 and TRT 7.1

August 2021: Updated the sample based on TF 2.6 and TRT 8.0. TRT introduces dedicated layers and optimizations for quantization layers which are demonstrated in the inference.

# License

The sampleQAT license can be found in the LICENSE file.
