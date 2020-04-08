# CAT-Tensorflow

An extension of [thu-spmi](https://github.com/thu-spmi) [CAT](https://github.com/thu-spmi/CAT) for Tensorflow.

## Introduction

This is a modified version of [thu-spmi/CAT](https://github.com/thu-spmi/CAT). I just modify the code to the new CPP Extensions API style of Tensorflow.

My task is all in Tensorflow, so I modify the source codes [thu-spmi/CAT](https://github.com/thu-spmi/CAT) in Tensorflow API.

**NOTE: It has some problem in GPU-DEN calculation. It has different result from CAT Pytorch API. Hope you can help me to find the problem.** 

## Contents

- Test:
  - `tests/test_ctc_crf_base_pytorch_binding.py` is a simple test for PyTorch API in [thu-spmi/CAT](https://github.com/thu-spmi/CAT).

  - `tests/test_ctc_crf_op.py` is a simple test for Tensorflow API for CAT.

- Tensorflow API：

  - `setup.py`: ctc_crf_loss setup script. It only test in g++ 5.0+ and Tensorflow 1.14.0
  - `ctc_crf_op_kernel.cc`: Tensorflow C++ binding.
  - `ctc_crf_tensorflow/__init__.py`: Tensorflow API for ctc_crf_loss.

- Others:

  - `gpu_ctc/`: Just follow [thu-spmi/CAT](https://github.com/thu-spmi/CAT)
  - `gpu_den/`: Just follow [thu-spmi/CAT](https://github.com/thu-spmi/CAT), only remove some usless code that I think will not used.

## Requirements

- kaldi
- Tensorflow 1.14.0+
- openfst
- python 2.7+ or python3

Note: test environment: Tensorflow 1.14.0

## References

CAT. https://github.com/thu-spmi/CAT.