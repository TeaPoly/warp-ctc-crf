# warp-crf

TensorFlow binding for CTC-CRF. An extension of [thu-spmi](https://github.com/thu-spmi) [CAT](https://github.com/thu-spmi/CAT) for Tensorflow.

## Introduction

This is a modified version of [thu-spmi/CAT](https://github.com/thu-spmi/CAT). I just modify the code to the new CPP Extensions API style of Tensorflow and refact the `gpu_den` code.

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

## Requirements

- kaldi
- Tensorflow 1.14.0+
- openfst
- python 2.7+ or python3

Note: test environment: Tensorflow 1.14.0

## References

CAT. https://github.com/thu-spmi/CAT.
