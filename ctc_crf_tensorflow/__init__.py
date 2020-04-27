import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

lib_file = imp.find_module('kernels', __path__)[1]
_ctc_crf = tf.load_op_library(lib_file)


def CRFGrad(grad_ctc, grad_den, lamb):
  # gradient
  grad = array_ops.transpose(grad_den, (1, 0, 2)) - \
      (1+lamb)*grad_ctc  # (T,B,N)
  # average with batch size.
  return grad / tf.cast(_get_dim(grad, 1), dtype=tf.float32)  # (T,B,N)


def CRFLoss(log_likelihood_ctc, log_likelihood_den, lamb):
  return log_likelihood_den - (1 + lamb) * log_likelihood_ctc


def ctc_crf_init_env(fst_name, gpus):
  return _ctc_crf.ctc_crf_init(gpus, fst_name)


def ctc_crf_release_env(gpus):
  return _ctc_crf.ctc_crf_release(gpus)


def ctc_crf_loss(time_major_logsoftmax, labels, input_lengths,
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
  log_likelihood_ctc, _, _, log_likelihood_den = _ctc_crf.ctc_crf_loss(
      time_major_logsoftmax,
      array_ops.transpose(time_major_logsoftmax, (1, 0, 2)),
      labels.indices,
      labels.values,
      input_lengths,
      lamb,
      blank_label)  # log_likelihood_ctc, grad_ctc, grad_den, log_likelihood_den

  if lamb >= 0:
    return CRFLoss(log_likelihood_ctc, log_likelihood_den, lamb)
  else:
    return -log_likelihood_ctc


@ops.RegisterGradient("CtcCrfLoss")
def _CTCLossGrad(op, grad_loss, grad_ctc, grad_den, log_likelihood_den):
  """The derivative provided by CTC Loss.

  Args:
     op: the CtcCrfLoss op.
     grad_loss: The backprop for cost.

  Returns:
     The CTC-CRF Loss gradient.
  """
  if op.get_attr('lamb') >= 0:
    grad = CRFGrad(op.outputs[1], op.outputs[2], op.get_attr('lamb'))
    # Return gradient for inputs and None for
    # batch_major_logsoftmax, labels_indices, labels_values and sequence_length.
    # return [_BroadcastMul(grad_loss, grad), None, None, None, None]
    return [grad, None, None, None, None]
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
