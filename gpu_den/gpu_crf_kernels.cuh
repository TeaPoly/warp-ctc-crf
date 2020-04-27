#pragma once

#include <stdio.h>

__host__ __device__
inline float log_plus(float a, float b) {
  if (a == -float(INFINITY)) { return b; }
  if (b == -float(INFINITY)) { return a; }
  float m = a > b ? a : b;
  return log1pf(expf(-fabs(a - b))) + m;
}

__device__ float atomic_log_plus(float* addr_f, float value) {
  int* addr = (int*)addr_f;
  float expected = *addr_f;
  float sum = log_plus(expected, value);
  int old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));

  while (old_value != __float_as_int(expected)) {
    expected = __int_as_float(old_value);
    sum = log_plus(expected, value);
    old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));
  }
  return __int_as_float(old_value);
}

// <<<batch_size, CU_BLOCK_CONST>>>
__global__ void alpha_first_kernel(float* alpha,
                                   const int alpha_size,
                                   const int batch_size,
                                   const int T,
                                   const float* const start_weight) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
    alpha[mini_batch_idx * alpha_size * (T + 1) + idx] = start_weight[idx];
  }
}

__global__ void alpha_kernel(float* alpha,
                             const float* const logits,
                             const int batch_size,
                             const int T,
                             const int t,
                             const int* const input_lengths,
                             const int alpha_size,
                             const int logits_size,
                             const IntPair* const alpha_transition_index,
                             const Transition* const alpha_transition) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  if (t > input_lengths[mini_batch_idx]) { return; }

  int idx1 = mini_batch_idx * alpha_size * (T + 1) + alpha_size * t;
  int idx2 = mini_batch_idx * alpha_size * (T + 1) + alpha_size * (t - 1);
  int idx3 = mini_batch_idx * logits_size * T + logits_size * (t - 1);

  for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
    int start = alpha_transition_index[idx].first;
    int end = alpha_transition_index[idx].second;
    float result = -float(INFINITY);
    for (int k = start; k <= end; k++) {
      result = log_plus(alpha[idx2 + alpha_transition[k].state] +
                        alpha_transition[k].weight + logits[idx3 + alpha_transition[k].label], result);
    }
    alpha[idx1 + idx] = result;
  }
}

__global__ void alpha_last_kernel(float* alpha,
                                  const int alpha_size,
                                  const int batch_size,
                                  const int T,
                                  const int* const input_lengths,
                                  const float* const end_weight) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  int alpha_start = mini_batch_idx * alpha_size * (T + 1);
  int cT = input_lengths[mini_batch_idx];

  for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
    alpha[alpha_start + cT * alpha_size + idx] += end_weight[idx];
  }
}

// <<< minibatch, N = 32,64,128...>>>
__global__ void alpha_lld_kernal(const float* const alpha,
                                 const int alpha_size,
                                 const int T,
                                 const int* const input_lengths,
                                 float* loglikelihood) {
  int mini_batch_idx = blockIdx.x;
  int idx = threadIdx.x;
  int block_dim = blockDim.x;
  int cT = input_lengths[mini_batch_idx];
  int last_idx = alpha_size * (T + 1) * mini_batch_idx + cT * alpha_size;
  // printf("enter alpha_lld_kernal, block.x: %d, thread.x: %d\n", blockIdx.x, threadIdx.x);

  extern __shared__ float sdata[];
  float temp = -float(INFINITY);

  for (int i = idx; i < alpha_size; i += block_dim) {
    temp = log_plus(temp, alpha[last_idx + i]);
  }
  sdata[idx] = temp;
  __syncthreads();

  for (int shift = block_dim / 2; shift > warpSize; shift >>= 1) {
    if (idx < shift) {
      sdata[idx] = log_plus(sdata[idx], sdata[idx + shift]);
    }
    __syncthreads();
  }

  if (idx < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sdata[idx] = log_plus(sdata[idx], sdata[idx + shift]);
    }
  }
  __syncthreads();

  if (idx == 0) {
    loglikelihood[mini_batch_idx] = sdata[0];
    // printf("alpha loglikelihod: %f mini_batch %d\n", loglikelihood[mini_batch_idx], mini_batch_idx);
  }
}

__global__ void beta_last_kernel(float* beta,
                                 const int beta_size,
                                 const int batch_size,
                                 const int* const input_lengths,
                                 const float* const end_weight) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  int cT = input_lengths[mini_batch_idx];

  for (int idx = tid; idx < beta_size; idx += blockDim.x) {
    beta[mini_batch_idx * 2 * beta_size + (cT % 2) * beta_size + idx] = end_weight[idx];
  }
}

__global__ void beta_first_kernel(float* beta,
                                  const int beta_size,
                                  const int batch_size,
                                  const float* const start_weight) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;

  for (int idx = tid; idx < beta_size; idx += blockDim.x) {
    beta[mini_batch_idx * 2 * beta_size + idx] += start_weight[idx];
  }
}

__global__ void beta_kernel(float* beta,
                            const float* const alpha,
                            const float* const logits,
                            float* grad_storage,
                            const int batch_size,
                            const int T,
                            const int t,
                            const int* input_lengths,
                            const int beta_size,
                            const int logits_size,
                            const IntPair* const beta_transition_index,
                            const Transition* const beta_transition) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  if (t >= input_lengths[mini_batch_idx]) { return; }
  int idx1 = mini_batch_idx * beta_size * (T + 1) + beta_size * t;
  int idx2 = mini_batch_idx * beta_size * 2 + beta_size * ((t + 1) % 2);
  int idx3 = mini_batch_idx * beta_size * 2 + beta_size * (t % 2);
  int idx4 = mini_batch_idx * logits_size * T + logits_size * t;
  int idx5 = mini_batch_idx * logits_size * ATOMIC_CONST;

  for (int idx = tid; idx < beta_size; idx += blockDim.x) {
    int start = beta_transition_index[idx].first;
    int end = beta_transition_index[idx].second;

    float beta_result = -float(INFINITY);
    float temp_value = -float(INFINITY);

    for (int k = start; k <= end; k++) {
      temp_value = beta[idx2 + beta_transition[k].state] + beta_transition[k].weight +
                   logits[idx4 + beta_transition[k].label];
      beta_result = log_plus(temp_value, beta_result);
      float partial_grad = alpha[idx1 + idx] + temp_value;
      float* grad_position = grad_storage + idx5 + beta_transition[k].label * ATOMIC_CONST + threadIdx.x % ATOMIC_CONST;
      atomic_log_plus(grad_position, partial_grad);
    }
    beta[idx3 + idx] = beta_result;
  }
}

__global__ void copy_grad(float* grad_storage,
                          float* grad_net,
                          const float* const alpha_lld,
                          const int* const input_lengths,
                          const int batch_size,
                          const int logits_size,
                          const int T,
                          const int t) {
  int mini_batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  if (t >= input_lengths[mini_batch_idx]) { return; }

  float lld = alpha_lld[mini_batch_idx];
  for (int idx = tid; idx < logits_size; idx += blockDim.x) {
    float* grad_position = grad_net + mini_batch_idx * logits_size * T + t * logits_size + idx;
    int idx_storage = mini_batch_idx * logits_size * ATOMIC_CONST + idx * ATOMIC_CONST;

    float grad = -float(INFINITY);
    for (int i = 0; i < ATOMIC_CONST; i++) {
      grad = log_plus(grad_storage[idx_storage + i], grad);
      grad_storage[idx_storage + i] = -float(INFINITY);
    }
    *grad_position = expf(grad - lld);
  }
}

__global__ void beta_lld_kernal(const float* const beta,
                                const int beta_size,
                                float* loglikelihood) {
  int idx = threadIdx.x;
  int first_idx = beta_size * 2 * idx;
  loglikelihood[idx] = beta[first_idx];
}
