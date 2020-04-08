import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

lib_file = imp.find_module('kernels', __path__)[1]
_ctc_crf = tf.load_op_library(lib_file)


def ctc_crf_init_env(fst_name, gpus):
  return _ctc_crf.ctc_crf_init(gpus, fst_name)


def ctc_crf_release_env(gpus):
  return _ctc_crf.ctc_crf_release(gpus)

def ctc_crf_loss(logits, labels, input_lengths,
                 blank_label=0, lamb=0.1):
  '''Computes the CTC loss between a sequence of logits and a
  ground truth labeling.

  Args:
      logits: A 3-D Tensor of floats. The dimensions
                   should be (t, n, a), where t is the time index, n
                   is the minibatch index, and a indexes over
                   logits for each symbol in the alphabet.

      labels: An int32 SparseTensor. labels.indices[i, :] == [b, t] means 
              labels.values[i] stores the id for (batch b, time t). 
              labels.values[i] must take on values in [0, num_labels).

      input_lengths: A 1-D Tensor of ints, the number of time steps
                     for each sequence in the minibatch.

      blank_label: int, the label value/index that the CTC
                   calculation should use as the blank label.

      lamb: float, A weight α for CTC Loss. 
                  Combined with the CRF loss to help convergence.

  Returns:
      1-D float Tensor, the cost of each example in the minibatch
      (as negative log probabilities).

  * This class performs the softmax operation internally.

  * The label reserved for the blank symbol should be label 0.

  '''
  # The input of the warp-ctc is modified to be the log-softmax output of the bottom neural network.
  activations = tf.nn.log_softmax(logits) # (t, n, a)
  activations_ = tf.transpose(activations, (1, 0, 2)) # (n, t, a)
  log_prob_ctc, grad_ctc, grad_den, log_prob_den = _ctc_crf.ctc_crf_loss(
      activations, activations_, labels.indices, labels.values,
      input_lengths, blank_label, lamb)

  grad_den = tf.transpose(grad_den, (1, 0, 2)) # (t, n, a)
  grad = grad_den - (1+lamb)*grad_ctc # (t, n, a)
  # average with batch size.
  grad /= tf.cast(_get_dim(grad, 1), dtype=tf.float32) # (t, n, a)

  if True:
    # loss = ((-log_prob_ctc)-(-log_prob_den)) + lamb*(-log_prob_ctc)
    #      = log_prob_den - (1+lamb)*log_prob_ctc
    return (log_prob_den - (1 + lamb) * log_prob_ctc), log_prob_den, log_prob_ctc, grad # (n,) 
  else:
    return -log_prob_ctc

@ops.RegisterGradient("CtcCrfLoss")
def _CTCLossGrad(op, grad_loss, a, b, c):
  """The derivative provided by CTC Loss.

  Args:
     op: the CtcCrfLoss op.
     grad_loss: The backprop for cost.

  Returns:
     The CTC-CRF Loss gradient.
  """
  if True:
    lamb = op.get_attr('lamb')
    grad_ctc = op.outputs[1] # (t, n, a)
    grad_den = tf.transpose(op.outputs[2], (1, 0, 2)) # (t, n, a)
    # grad = (grad_ctc-grad_den)+lamb*(grad_ctc)
    #      = -grad_den + (1+lamb)*grad_ctc
    grad = grad_den - (1+lamb)*grad_ctc # (t, n, a)
    # average with batch size.
    grad /= tf.cast(_get_dim(grad, 1), dtype=tf.float32) # (t, n, a)

    # Return gradient for inputs and None for
    # activations_, labels_indices, labels_values and sequence_length.
    return [_BroadcastMul(grad_loss, grad), None, None, None, None]
  else:
    return [_BroadcastMul(grad_loss, op.outputs[1]), None, None, None, None]

@ops.RegisterShape("CtcCrfLoss")
def _CTCLossShape(op):
  inputs_shape = op.inputs[0].get_shape().with_rank(3)
  batch_size = inputs_shape[1]
  return [batch_size, inputs_shape]

def _get_dim(tensor, i):
  """Get value of tensor shape[i] preferring static value if available."""
  return tensor_shape.dimension_value(
      tensor.shape[i]) or array_ops.shape(tensor)[i]
