"""setup.py script for warp-ctc TensorFlow wrapper"""

from __future__ import print_function

import os
import platform
import re
import setuptools
import sys
import unittest
import warnings
from setuptools.command.build_ext import build_ext as orig_build_ext

CUDA_HOME='/usr/local/cuda'
den_dir = os.path.realpath("./gpu_den/build")
ctc_dir = os.path.realpath("./gpu_ctc/build")

# We need to import tensorflow to find where its include directory is.
try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError("Tensorflow must be installed to build the tensorflow wrapper.")

if "TENSORFLOW_SRC_PATH" not in os.environ: # /home/huanglk/anaconda3/envs/tensorflow_env/lib/python3.6/site-packages/tensorflow
    print("Please define the TENSORFLOW_SRC_PATH environment variable.\n"
          "This should be a path to the Tensorflow source directory.",
          file=sys.stderr)
    sys.exit(1)

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
# ctc_crf_includes = [os.path.join(root_path, '../include')]
ctc_crf_includes = ['./gpu_ctc','./gpu_den']
include_dirs = tf_includes + ctc_crf_includes

#include_dirs += [tf_include + '/../../external/nsync/public']

if os.getenv("TF_CXX11_ABI") is not None:
    TF_CXX11_ABI = os.getenv("TF_CXX11_ABI")
else:
    warnings.warn("Assuming tensorflow was compiled without C++11 ABI. "
                  "It is generally true if you are using binary pip package. "
                  "If you compiled tensorflow from source with gcc >= 5 and didn't set "
                  "-D_GLIBCXX_USE_CXX11_ABI=0 during compilation, you need to set "
                  "environment variable TF_CXX11_ABI=1 when compiling this bindings. "
                  "Also be sure to touch some files in src to trigger recompilation. "
                  "Also, you need to set (or unsed) this environment variable if getting "
                  "undefined symbol: _ZN10tensorflow... errors")
    TF_CXX11_ABI = "0"

extra_compile_args = ['-std=c++11', '-fPIC'] # , '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI
# current tensorflow code triggers return type errors, silence those for now
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
    extra_link_args = ['-L' + tf.sysconfig.get_lib(), '-ltensorflow_framework']

include_dirs += [os.path.join(CUDA_HOME, 'include')]

# mimic tensorflow cuda include setup so that their include command work
if not os.path.exists(os.path.join(root_path, "include")):
    os.mkdir(os.path.join(root_path, "include"))

cuda_inc_path = os.path.join(root_path, "include/cuda")
if not os.path.exists(cuda_inc_path) or os.readlink(cuda_inc_path) != CUDA_HOME:
    if os.path.exists(cuda_inc_path):
        os.remove(cuda_inc_path)
    os.symlink(CUDA_HOME, cuda_inc_path)
include_dirs += [os.path.join(root_path, 'include')]

# Ensure that all expected files and directories exist.
for loc in include_dirs:
    if not os.path.exists(loc):
        print(("Could not find file or directory {}.\n"
               "Check your environment variables and paths?").format(loc),
              file=sys.stderr)
        sys.exit(1)

lib_srcs = ['ctc_crf_op_kernel.cc']

ext = setuptools.Extension('ctc_crf_tensorflow.kernels',
                           sources = lib_srcs,
                           language = 'c++',
                           include_dirs = include_dirs,
                           library_dirs = [den_dir, ctc_dir],
                           runtime_library_dirs = [den_dir, ctc_dir],
                           libraries = ['crf_fst_den','crf_fst_read','crf_warpctc', 'tensorflow_framework'],
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args)

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

setuptools.setup(
    name = "ctc_crf_tensorflow",
    version = "0.1",
    description = "TensorFlow wrapper for ctc-crf",
    license = "Apache",
    packages = ["ctc_crf_tensorflow"],
    ext_modules = [ext],
    cmdclass = {'build_ext': build_tf_ext},
)
