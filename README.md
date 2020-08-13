# warp-crf

TensorFlow binding for CTC-CRF. An extension of [thu-spmi](https://github.com/thu-spmi) [CAT](https://github.com/thu-spmi/CAT) for Tensorflow.

## Introduction

This is a modified version of [thu-spmi/CAT](https://github.com/thu-spmi/CAT). I just modify the code to the new CPP Extensions API style of Tensorflow and refact the `gpu_den` code.

## Requirements

- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)/[CuDNN](https://developer.nvidia.com/cudnn)
- [gcc/g++ 5.5.0](https://gcc.gnu.org)
- [Python3](https://www.python.org/download/releases/3.0/)
- [TensorFlow](https://www.tensorflow.org)
- [OpenFst](http://www.openfst.org)
- [Kaldi](https://kaldi-asr.org)

## Installation

Because CTC-CRF operator is based on CUDA Toolkit, so you should setting CUDA environment. For details, you can follow this [link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) or TensorFlow official [link](https://www.tensorflow.org/install/pip?hl=zh-cn).

1. Install CUDA Toolkit

- Follow this [link](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64) to download and install CUDA Toolkit 10.1 for your Linux distribution.
- Installation instructions can be found [here](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-linux/index.html)

- Install CUDNN
  - Go to https://developer.nvidia.com/rdp/cudnn-download
  - Create a user profile if needed and log in
  - Select [cuDNN v7.6.5 (Nov 5, 2019), for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-download#a-collapse765-101)
  - Download [cuDNN v7.6.5 Library for Linux](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-linux-x64-v7.6.5.32.tgz)
  - Follow the instructions under Section 2.3.1 of the [CuDNN Installation Guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux) to install CuDNN.

2. Environment Setup

   Append the following lines to `~/.bashrc` or `~/.zshrc`.

   ```shell
   export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

3. Install TensorFlow with Anaconda virtual environment

   Create a virtual environment is recommended. You can choose [Conda](https://www.tensorflow.org/install/pip#conda) or [venv](https://docs.python.org/3/library/venv.html). Here I use Conda as an example.
   
     ```shell
   # Install TensorFlow/cuda/nvcc first, reference is here:
   conda create --name tf pip python==3.7
   conda activate tf
   conda install tensorflow-gpu==1.15.0
     ```
   
4. Install CTC-CRF TensorFlow wrapper [warp-ctc-crf](https://github.com/TeaPoly/warp-ctc-crf)

   setting your `TENSORFLOW_SRC_PATH` and `OPENFST`.

   NOTE: This is an example, please don't simply copy to your terminal:

   ```shell
   # Create a symlink libtensorflow_framework.so.1 which references the original file  libtensorflow_framework.so
   ln -s /home/huanglk/anaconda3_202002/envs/tf_subcomp/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 /home/huanglk/anaconda3_202002/envs/tf_subcomp/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so
   
   # export TENSORFLOW_SRC_PATH
   export TENSORFLOW_SRC_PATH=/home/huanglk/anaconda3_202002/envs/tf_subcomp/lib/python3.7/site-packages/tensorflow_core/
   
   # export OPENFST
   export OPENFST=/usr/local/
   ```

   - It will compile three modules with gcc/g++, include `GPUCTC`, `PATHWEIGHT` and `GPUDEN`.
   - It is worth mentioning that if the version of gcc/g++ >= 5.0.0 and less than 6.0.0 will be helpful for following pipeline.
   - Finally, `Makefile` will exetucate `python3 ./setup.py install` for CTC-CRF TensorFlow wrapper.

   Now, you can install CTC-Crf TensorFlow wrapper `warp-ctc-crf`.

   ```shell
   # Install warp_ctc_crf
   cd warp_ctc_crf
   make -j 32
   ```

## Contents

- Test:

  - `tests/test_ctc_crf_op.py` is a simple test for Tensorflow API for CAT.

- Tensorflow API：

  - `setup.py`: ctc_crf_loss setup script. It only test in g++ 5.0+ and Tensorflow 1.14.0
  - `ctc_crf_op_kernel.cc`: Tensorflow C++ binding.
  - `ctc_crf_tensorflow/__init__.py`: Tensorflow API for ctc_crf_loss.

- Others:

  - `gpu_ctc/`: Just follow [thu-spmi/CAT](https://github.com/thu-spmi/CAT)
  - `gpu_den/`: Code refactoring from [thu-spmi/CAT](https://github.com/thu-spmi/CAT).


## References

CAT. https://github.com/thu-spmi/CAT.
