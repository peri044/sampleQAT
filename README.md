# TensorRT inference of Resnet-50 trained with QAT.

**Table Of Contents**

- [Description](#description)
- [Prerequisites](#pre-requisites)
- [Running the sample](#running-the-sample)
  * [Step 1: Quantization Aware Training](#step-1-quantization-aware-training)
  * [Step 2: Export a RN50 QAT saved model](#step-2-export-a-rn50-qat-saved-model)
  * [Step 3: Conversion to ONNX](#step-3-conversion-to-onnx)
  * [Step 4: Conversion to TensorRT engine](#step-4-conversion-to-tensorrt-engine)
  * [Step 5: TensorRT Inference](#step-5-tensorrt-inference)
- [Additional resources](#additional-resources)
- [Changelog](#changelog)
- [License](#license)


## Description

This sample demonstrates workflow for training and inference of Resnet-50 model trained using Quantization Aware Training in Tensorflow 2.X from Tensorflow model garden. The inference uses TensorRT SDK which performs additional optimizations to QAT models to run in INT8 precision.

## Pre-requisites

* TensorRT 8.0.1.6, CUDA-11.2, CUDNN 8.2.1
* Clone the sample, setup `PYTHONPATH` accordingly, and install the necessary requirements.

```sh
git clone https://github.com/NVIDIA/sampleQAT.git
cd sampleQAT
export PYTHONPATH=$PWD:$PYTHONPATH
pip install -r requirements.txt
```

* Install TF2ONNX from <a href="https://github.com/peri044/tensorflow-onnx/tree/qdq_changes">this branch</a>. This branch has the changes to ensure we remove redundant transpose nodes at weighted layers and disable constant folding to avoid pruning `QuantizeAndDequantize` nodes at weights of conv/FC layers. A pull request to upstream is in progress.
```sh
git clone -b qdq_changes --single-branch https://github.com/peri044/tensorflow-onnx.git
cd tensorflow-onnx
python setup.py install
```

* Clone the models from Tensorflow model garden.

```sh
git clone https://github.com/tensorflow/models.git
pushd models && git checkout tags/v2.6.0 && popd
export PYTHONPATH=$PWD/models:$PYTHONPATH
pip install -r models/official/requirements.txt
```

## Running the sample

Clone the sample and setup `PYTHONPATH` accordingly.

```sh
git clone https://github.com/NVIDIA/sampleQAT.git
cd sampleQAT
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Step 1: Quantization Aware Training

The workflow for this step is

* Instantiate Resnet 50 model, load the pretrained checkpoint
* Apply `quantize_model` which transforms the original Resnet50 graph by inserting QDQ nodes.
* Finetune this model on ImageNet dataset and save the QAT model checkpoints.

NVIDIA recommends to insert `QuantizeAndDequantize`  (QDQ) nodes before inputs and weights of weighted layers. Please take a look at this <a href="https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/"> blogpost</a>  for detailed recommendations.

We provide automatic way to insert these QDQ nodes at the inputs of conv/FC, Maxpool layers and weights of conv/FC layers.  Please refer to [`quantize.py`](quantize.py), [`quantize_wrapper.py`](quantize_wrapper.py) and [`quantizers.py`](quantizers.py).

For this sample, we use <a href="https://github.com/tensorflow/models/tree/v2.6.0/official/vision/image_classification/resnet">Resnet50 from Tensorflow model garden codebase</a> for training. We use custom training loop implementation of this codebase to finetune our model using quantization nodes.

`resnet_runnable.py` script is responsible for instantiating the model. To apply NVIDIA's recipe of adding QDQ nodes, you need to change this script at
https://github.com/tensorflow/models/blob/v2.6.0/official/vision/image_classification/resnet/resnet_runnable.py#L64 as follows

```python
self.model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        batch_size=flags_obj.batch_size,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

self.model.load_weights(<path_to_pretrained_ckpt>)

from quantize import quantize_model
self.model = quantize_model(self.model)
```
> Note: `<path_to_pretrained_ckpt>` = directory where the pre-trained checkpoint was saved + filename root (i.e., `model.ckpt--0090`).

  `quantize_model` walks through the graph, identifies `Conv` and `FC` layers  and inserts QDQ nodes around them. The `min_var` and `max_var` determine the dynamic range of a particular layer which is used to compute the scale factors used for INT8 conversion. This scale factor computation is performed internally in TF2ONNX when the QAT finetuned model is converted to ONNX.

Download <a href="https://github.com/tensorflow/models/tree/v2.6.0/official/vision/image_classification/resnet#pretrained-models">pretrained checkpoint for RN50</a> and set `path_to_pretrained_ckpt` accordingly in the above snippet of `resnet_runnable.py` script.

With these modifications, you can proceed with the finetuning of the model with QAT using the <a href="https://github.com/tensorflow/models/tree/v2.6.0/official/vision/image_classification/resnet#resnet-custom-training-loop">instructions provided in the TF model garden codebase</a>. The input image shape is in `NHWC` format (1, 224, 224, 3). The training data is passed in `NHWC` format as well.
> Example: `python resnet_ctl_imagenet_main.py --model_dir=<checkpoints_finetuned_save_dir> --num_gpus=1 --batch_size=128 --train_epochs=1 --train_steps=10 --use_synthetic_data=false --data_dir=<dir_to_imagenet_train_val_tfrecord> --skip_eval --enable_checkpoint_and_export`

### Step 2: Export a RN50 QAT saved model

Once you've finetuned the QAT model, export it by running

```python
python export_rn50_qat.py --ckpt <path_to_ckpt> --output <path_to_saved_model>
```

This script applies quantization to the model, restores the checkpoint and exports it in a saved_model format. This script will generate `rn50_qat_saved_model`  which is a directory containing saved model. We set the overall graph data format to `NCHW` by using `tf.keras.backend.set_image_data_format('channels_first')`. TensorRT expects `NCHW` format for graphs trained with QAT for better optimizations. Due to this, a transpose layer is introduced at the input of the graph for the RN50 model. The graph looks as follows

![Alt text](qat_pb.png?raw=true "RN50 QAT graph in NCHW format")

Arguments:

* `--ckpt` : Path to finetuned QAT checkpoint to be loaded.
* `--output` : Name of output TF saved model

### Step 3: Conversion to ONNX

Convert the saved model into ONNX by running

```python
python -m tf2onnx.convert --saved-model <path_to_saved_model> --output rn50_qat.onnx  --opset 13 --disable_constfold
```

By default, tf2onnx uses TF's graph optimizers to performs constant folding after a saved model is loaded. `--disable_constfold` is necessary to disable constant folding of `QuantizeAndDequantize` nodes around weights of convolutional/fully connected layers. Note: the transpose node introduced due to `NHWC->NCHW` data format change is moved around in the graph after `QuantizeLinear` node due to the `TransposeOptimizer` in TF2ONNX.  You can checkout the ONNX graph for RN50 with QDQ nodes ([`rn50_qat_graph.png`](rn50_qat_graph.png)) can be seen for reference.

Arguments:

* `--saved-model` : Name of TF saved-model
* `--output` : Name of ONNX output graph
* `--opset` : ONNX opset version (opset 13 or higher must be used)
* `--disable_constfold` : This flag disables constant folding performed by Tensorflow's grappler optimizer.

### Step 4: Conversion to TensorRT Engine

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
usage: python <filename>.py [-h]
```

## Results

These are the results we've seen in our initial experiments. Accuracy can differ based on your hyperparameters and number of epochs the model was trained for. When measuring accuracy in Tensorflow, the image decoding and processing utilities were using TF APIs, while `numpy` and `PIL` were used for image processing in TRT 8 accuracy calculation. This discrepancy might lead to numerical differences.

| Model/Framework                                | Accuracy |
| ---------------------------------------------- | -------- |
| Resnet 50 (without QAT) in Tensorflow          | 76.47 %  |
| Resnet 50 (with QAT) in Tensorflow             | 76.39 %  |
| Resnet 50 (with QAT) deployed using TensorRT 8 | 76.16 %  |

On 2080 Ti using TensorRT, we observed a speedup of `2.37x` when comparing RN50 (without QAT, FP32 precision) with RN50 (with QAT, INT8 precision).

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

August 2021: Updated the sample based on TF 2.6 and TRT 8.0GA. TRT introduces dedicated layers and optimizations for quantization layers which are demonstrated in the inference.

# License

The sampleQAT license can be found in the LICENSE file.
