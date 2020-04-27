#pragma once

#include "gpu_crf_kernels.cuh"

class GpuCRF {
 public:
  GpuCRF(const int minibatch,
         const int max_time,
         const int alphabet_size,
         const int den_num_states,
         void* workspace,
         Transition** transition_alpha,
         Transition** transition_beta,
         IntPair** transition_index_alpha,
         IntPair** transition_index_beta,
         float** start_weight,
         float** end_weight,
         int* device_hash,
         CRFCUstream stream):
    minibatch_(minibatch),
    max_time_(max_time),
    alphabet_size_(alphabet_size),
    den_num_states_(den_num_states),
    gpu_workspace_(workspace),
    stream_(stream),
    transition_alpha_(transition_alpha),
    transition_beta_(transition_beta),
    transition_index_alpha_(transition_index_alpha),
    transition_index_beta_(transition_index_beta),
    start_weight_(start_weight),
    end_weight_(end_weight),
    device_hash_(device_hash) { };

  // Noncopyable
  GpuCRF(const GpuCRF&) = delete;
  GpuCRF& operator=(const GpuCRF&) = delete;

  crfStatus_t
  cost_and_grad(const float* const logits,
                const int* const input_lengths,
                float* loglikelihood,
                // float* loglikelihood_verify,
                float* grad);

 private:
  crfStatus_t
  setup_gpu_metadata(const int* const input_lengths);

  crfStatus_t
  compute_alpha(const float* const logits, float* loglikelihood);

  crfStatus_t
  compute_beta_and_grad(const float* const logits,
                        const float* const alpha_lld,
                        float* grad
                        // ,
                        // float* loglikelihood
                        );

  Transition** transition_alpha_;
  Transition** transition_beta_;
  IntPair** transition_index_alpha_;
  IntPair** transition_index_beta_;
  float** start_weight_;
  float** end_weight_;
  int* device_hash_;

  int minibatch_;
  int max_time_;
  int alphabet_size_;
  int den_num_states_;
  void* gpu_workspace_;
  int* utt_length_;
  CRFCUstream stream_;

  float* alpha_;
  float* beta_;
  float* grad_storage_;
  float* loglikelihood_;
  // float* loglikelihood_verify_;
};

crfStatus_t
GpuCRF::setup_gpu_metadata(const int* const input_lengths) {
  size_t gpu_bytes_used = 0;
  cudaError_t cuda_status;

  loglikelihood_ =
    reinterpret_cast<float*>(static_cast<char*>(gpu_workspace_) +
                             gpu_bytes_used);
  gpu_bytes_used += minibatch_ * sizeof(float);

  // loglikelihood_verify_ =
  //   reinterpret_cast<float*>(static_cast<char*>(gpu_workspace_) +
  //                            gpu_bytes_used);
  // gpu_bytes_used += minibatch_ * sizeof(float);

  alpha_ =
    reinterpret_cast<float*>(static_cast<char*>(gpu_workspace_) +
                             gpu_bytes_used);
  gpu_bytes_used += (max_time_ + 1) * minibatch_ * den_num_states_ * sizeof(float);


  beta_ =
    reinterpret_cast<float*>(static_cast<char*>(gpu_workspace_) +
                             gpu_bytes_used);
  gpu_bytes_used += 2 * minibatch_ * den_num_states_ * sizeof(float);


  grad_storage_ =
    reinterpret_cast<float*>(static_cast<char*>(gpu_workspace_) +
                             gpu_bytes_used);
  gpu_bytes_used += ATOMIC_CONST * minibatch_ * alphabet_size_ * sizeof(float);

  // Allocate memory for N
  utt_length_ =
    reinterpret_cast<int*>(static_cast<char*>(gpu_workspace_) +
                           gpu_bytes_used);
  gpu_bytes_used += minibatch_  * sizeof(int);

  cuda_status = cudaMemcpyAsync(utt_length_, input_lengths,
                                minibatch_ * sizeof(int),
                                cudaMemcpyHostToDevice, stream_);
  if (cuda_status != cudaSuccess) {
    return CRF_STATUS_MEMOPS_FAILED;
  }

  return CRF_STATUS_SUCCESS;
}

crfStatus_t
GpuCRF::cost_and_grad(const float* const logits,
                      const int* const input_lengths,
                      float* loglikelihood,
                      // float* loglikelihood_verify,
                      float* grad) {
  cudaError_t cuda_status_mem, cuda_status_sync;

  crfStatus_t status = setup_gpu_metadata(input_lengths);
  if (status != CRF_STATUS_SUCCESS) {
    return status;
  }

  status = compute_alpha(logits, loglikelihood_);
  if (status != CRF_STATUS_SUCCESS) {
    return status;
  }

  cuda_status_mem = cudaMemcpyAsync(loglikelihood, loglikelihood_,
                                    sizeof(float) * minibatch_,
                                    cudaMemcpyDeviceToHost, stream_);
  // cuda_status_sync = cudaStreamSynchronize(stream_);
  // if (cuda_status_mem != cudaSuccess || cuda_status_sync != cudaSuccess) {
  //   return CRF_STATUS_MEMOPS_FAILED;
  // }

  if (grad != NULL) {
    status = compute_beta_and_grad(
      logits, 
      loglikelihood_, 
      grad
      // , 
      // loglikelihood_verify_
      );
    if (status != CRF_STATUS_SUCCESS) {
      return status;
    }

    // cuda_status_mem = cudaMemcpyAsync(loglikelihood_verify, loglikelihood_verify_,
    //                                   sizeof(float) * minibatch_,
    //                                   cudaMemcpyDeviceToHost, stream_);
  }

  cuda_status_sync = cudaStreamSynchronize(stream_);
  if (cuda_status_mem != cudaSuccess || cuda_status_sync != cudaSuccess) {
    return CRF_STATUS_MEMOPS_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return CRF_STATUS_EXECUTION_FAILED;
  }

  return CRF_STATUS_SUCCESS;
}

crfStatus_t
GpuCRF::compute_alpha(const float* const logits, float* loglikelihood) {
  int device = 0;

  CHECK_CUDA(cudaGetDevice(&device));
  int gid = device_hash_[device];

  int alpha_lld_dim = 128;
  alpha_first_kernel<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(alpha_, den_num_states_, minibatch_, max_time_, start_weight_[gid]);

  for (int t = 1; t <= max_time_; t++) {
    alpha_kernel<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(alpha_, logits, minibatch_, max_time_, t, utt_length_,
        den_num_states_, alphabet_size_, transition_index_alpha_[gid], transition_alpha_[gid]);
  }

  alpha_last_kernel<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(alpha_, den_num_states_, minibatch_, max_time_, utt_length_, end_weight_[gid]);

  alpha_lld_kernal<<<minibatch_, alpha_lld_dim, sizeof(float)*alpha_lld_dim, stream_>>>(alpha_, den_num_states_, max_time_, utt_length_, loglikelihood);

  // cudaDeviceSynchronize();
  return CRF_STATUS_SUCCESS;
}

crfStatus_t
GpuCRF::compute_beta_and_grad(const float* const logits,
                              const float* const alpha_lld,
                              float* grad
                              // ,
                              // float* loglikelihood
                              ) {
  int device = 0;

  CHECK_CUDA(cudaGetDevice(&device));
  int gid = device_hash_[device];

  copy_grad<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(grad_storage_, grad, alpha_lld, utt_length_, minibatch_, alphabet_size_, max_time_, 0);

  beta_last_kernel<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(beta_, den_num_states_, minibatch_, utt_length_, end_weight_[gid]);

  for (int t = max_time_ - 1; t >= 0; t--) {
    beta_kernel<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(beta_, alpha_, logits, grad_storage_, minibatch_, max_time_, t, utt_length_, den_num_states_, alphabet_size_,
        transition_index_beta_[gid], transition_beta_[gid]);

    copy_grad<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(grad_storage_, grad, alpha_lld, utt_length_, minibatch_, alphabet_size_, max_time_, t);
  }

  beta_first_kernel<<<minibatch_, CU_BLOCK_DIM, 0, stream_>>>(beta_, den_num_states_, minibatch_, start_weight_[gid]);

  // beta_lld_kernal<<<1, minibatch_, 0, stream_>>>(beta_, den_num_states_, loglikelihood);

  // cudaDeviceSynchronize();
  return CRF_STATUS_SUCCESS;
}
