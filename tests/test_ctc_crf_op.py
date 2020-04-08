import tensorflow as tf
import numpy as np
from ctc_crf_tensorflow import ctc_crf_init_env, ctc_crf_release_env, ctc_crf_loss

LM_PATH = '/data/203_data/text_database/crf_aishell_train/den_lm.fst'
GPUS = [0]

def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).

  Args:
    x: a list of lists of type int

  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)

  return tf.SparseTensor(x_ix, x_val, x_shape)

class CtcCrfTest(tf.test.TestCase):
  def _test_init_final(self, gpus, fst_name):
    gpus = tf.constant(gpus)
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=False)
    init_ops = ctc_crf_init_env(fst_name, gpus)
    release_ops = ctc_crf_release_env(gpus)
    with self.test_session(use_gpu=True, force_gpu=True, config=config) as sess:
      sess.run([init_ops])
      sess.run([release_ops])

  def test_init_realese(self):
    self._test_init_final(
        gpus=np.array(GPUS, dtype=np.int32),
        fst_name=LM_PATH)

  def _run_ctc(self, fst_name, gpus, logits, input_lengths,
               labels):
    init_ops = ctc_crf_init_env(fst_name, gpus)
    release_ops = ctc_crf_release_env(gpus)

    logits_t = tf.transpose(tf.constant(logits), (1, 0, 2))
    input_lengths_t = tf.constant(input_lengths)
    # labels_t = tf.constant(labels)

    costs, loss_den, loss_ctc, grad_raw = ctc_crf_loss(
        logits=logits_t,
        labels=labels,
        input_lengths=input_lengths_t,
        blank_label=0)
    # costs = tf.reduce_mean(costs)
    grad = tf.gradients(costs, [logits_t])[0]
    # grad = grad / costs
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=False)

    with self.test_session(use_gpu=True, force_gpu=True, config=config) as sess:
      sess.run([init_ops])
      (tf_costs, tf_loss_den, tf_loss_ctc, tf_grad_raw, tf_grad) = sess.run([costs, loss_den, loss_ctc, grad_raw, grad])

      print("Part loss: ", tf_costs)
      print("Den loss: ", tf_loss_den)
      print("Ctc loss: ", tf_loss_ctc)
      print("Grad raw: \n", tf_grad_raw)
      # print(tf_grad)

      sess.run([release_ops])

  def test_basic(self):
    # Softmax logits for the following inputs:
    logits = np.array([
        [0.1, 0.6, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.1, 0.1]
    ], dtype=np.float32)

    # dimensions should be t, n, p: (t timesteps, n minibatches,
    # p prob of each alphabet). This is one instance, so expand
    # dimensions in the middle
    logits = np.expand_dims(logits, 0)
    labels = SimpleSparseTensorFrom([[1, 2]])
    input_lengths = np.asarray([2], dtype=np.int32)
    self._run_ctc(fst_name=LM_PATH,
                  gpus=np.array(GPUS, dtype=np.int32),
                  logits=logits,
                  input_lengths=input_lengths,
                  labels=labels)

  # def test_multiple_basic(self):
  #   # Softmax logits for the following inputs:
  #   logits_small = np.array([
  #       [0.1, 0.3, 0.1, 0.1, 0.1],
  #       [0.1, 0.1, 0.3, 0.1, 0.1]
  #   ], dtype=np.float32)

  #   logits_big = np.array([
  #       [0.1, 0.6, 0.1, 0.1, 0.1],
  #       [0.1, 0.1, 0.6, 0.1, 0.1]
  #   ], dtype=np.float32)

  #   logits_false = np.array([
  #       [0.1, 0.0, 0.1, 0.1, 0.1],
  #       [0.1, 0.1, 0.0, 0.1, 0.1]
  #   ], dtype=np.float32)

  #   # dimensions should be t, n, p: (t timesteps, n minibatches,
  #   # p prob of each alphabet). This is one instance, so expand
  #   # dimensions in the middle
  #   _logits_small = np.expand_dims(logits_small, 0)
  #   _logits_big = np.expand_dims(logits_big, 0)
  #   _logits_false = np.expand_dims(logits_false, 0)

  #   logits = np.concatenate([_logits_small, _logits_big[...], _logits_false[...]], axis=0)
  #   labels = SimpleSparseTensorFrom([[1, 2],[1, 2],[1, 2]])
  #   input_lengths = np.asarray([2, 2, 2], dtype=np.int32)
  #   self._run_ctc(fst_name=LM_PATH,
  #                 gpus=np.array(GPUS, dtype=np.int32),
  #                 logits=logits,
  #                 input_lengths=input_lengths,
  #                 labels=labels)


if __name__ == "__main__":
  tf.test.main()
