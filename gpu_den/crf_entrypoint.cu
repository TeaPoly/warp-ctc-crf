#include <iostream>
#include <cstdlib>
#include <vector>

#include "crf.h"
#include "gpu_crf.cuh"

#include <fst/fstlib.h>

using namespace fst;

static int kDenNumStates = 0;
static Transition** kTransitionAlpha = NULL;
static Transition** kTransitionBeta = NULL;
static IntPair** kTransitionIndexAlpha = NULL;
static IntPair** kTransitionIndexBeta = NULL;
static float** kStartWeight = NULL;
static float** kEndWeight = NULL;
static int* kDeviceHash = NULL;

extern "C" {
  const char* crfGetStatusString(crfStatus_t status) {
    switch (status) {
      case CRF_STATUS_SUCCESS:
        return "no error";
      case CRF_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
      case CRF_STATUS_INVALID_VALUE:
        return "invalid value";
      case CRF_STATUS_EXECUTION_FAILED:
        return "execution failed";
      case CRF_STATUS_UNKNOWN_ERROR:
      default:
        return "unknown error";
    }
  }

  static void ReadFst(const char* fst_name,
                      std::vector<std::vector<int> >& alpha_next,
                      std::vector<std::vector<int> >& beta_next,
                      std::vector<std::vector<int> >& alpha_ilabel,
                      std::vector<std::vector<int> >& beta_ilabel,
                      std::vector<std::vector<float> >& alpha_weight,
                      std::vector<std::vector<float> >& beta_weight,
                      std::vector<float>& start_weight,
                      std::vector<float>& end_weight,
                      int& num_states,
                      int& num_arcs) {

    // assume that they are proper initialized
    StdVectorFst* fst = StdVectorFst::Read(fst_name);
    num_states = fst->NumStates();

    num_arcs = 0;
    for (StateIterator<StdVectorFst> siter(*fst); !siter.Done(); siter.Next()) {
      num_arcs += fst->NumArcs(siter.Value());
    }

    // std::cout << "num_arcs: " << num_arcs << std::endl;
    // std::cout << "num_states: " << num_states << std::endl;

    alpha_next.resize(num_states, std::vector<int>());
    beta_next.resize(num_states, std::vector<int>());
    alpha_ilabel.resize(num_states, std::vector<int>());
    beta_ilabel.resize(num_states, std::vector<int>());
    alpha_weight.resize(num_states, std::vector<float>());
    beta_weight.resize(num_states, std::vector<float>());

    start_weight.resize(num_states, -float(INFINITY));
    end_weight.resize(num_states, -float(INFINITY));

    start_weight[fst->Start()] = 0.;

    for (StateIterator<StdVectorFst> siter(*fst); !siter.Done(); siter.Next()) {
      if (fst->Final(siter.Value()) != StdArc::Weight::Zero()) {
        end_weight[siter.Value()] = -fst->Final(siter.Value()).Value();
      }
      int state = siter.Value();

      for (ArcIterator<StdVectorFst> aiter(*fst, siter.Value()); !aiter.Done(); aiter.Next()) {
        beta_next[state].push_back(aiter.Value().nextstate);
        alpha_next[aiter.Value().nextstate].push_back(state);

        beta_ilabel[state].push_back(aiter.Value().ilabel - 1);
        alpha_ilabel[aiter.Value().nextstate].push_back(aiter.Value().ilabel - 1);

        beta_weight[state].push_back(-aiter.Value().weight.Value());
        alpha_weight[aiter.Value().nextstate].push_back(-aiter.Value().weight.Value());
      }
    }
  }

  crfStatus_t crfInit(const char* fst_name, const int n_gpus, const int* const gpus) {
    std::vector<std::vector<int> > alpha_next;
    std::vector<std::vector<int> > beta_next;
    std::vector<std::vector<int> > alpha_ilabel;
    std::vector<std::vector<int> > beta_ilabel;
    std::vector<std::vector<float> > alpha_weight;
    std::vector<std::vector<float> > beta_weight;
    std::vector<float> start_weight;
    std::vector<float> end_weight;

    int num_states = 0;
    int num_arcs = 0;

    std::ifstream fstfile(fst_name);
    if (fstfile.fail()) {
      return CRF_STATUS_INVALID_VALUE;
    }

    ReadFst(fst_name, alpha_next, beta_next, alpha_ilabel, beta_ilabel,
            alpha_weight, beta_weight, start_weight, end_weight, num_states, num_arcs);

    kDenNumStates = num_states;

    std::vector<Transition> transition_alpha(num_arcs);
    std::vector<Transition> transition_beta(num_arcs);
    std::vector<IntPair> transition_index_alpha(num_states);
    std::vector<IntPair> transition_index_beta(num_states);

    int count = 0;
    for (int i = 0; i < num_states; i++) {
      if (alpha_next[i].empty()) {
        transition_index_alpha[i].first = 1;
        transition_index_alpha[i].second = 0;
      } else {
        transition_index_alpha[i].first = count;
        for (size_t j = 0; j < alpha_next[i].size(); j++) {
          transition_alpha[count].state = alpha_next[i][j];
          transition_alpha[count].label = alpha_ilabel[i][j];
          transition_alpha[count].weight = alpha_weight[i][j];
          count++;
        }
        transition_index_alpha[i].second = count - 1;
      }
    }

    if (count != num_arcs) {
      std::cerr << "[GPU-DEN] Init count does not equal to num_arcs" << std::endl;
      return CRF_STATUS_UNKNOWN_ERROR;
    }

    count = 0;
    for (int i = 0; i < num_states; i++) {
      if (beta_next[i].empty()) {
        transition_index_beta[i].first = 1;
        transition_index_beta[i].second = 0;
      } else {
        transition_index_beta[i].first = count;
        for (size_t j = 0; j < beta_next[i].size(); j++) {
          transition_beta[count].state = beta_next[i][j];
          transition_beta[count].label = beta_ilabel[i][j];
          transition_beta[count].weight = beta_weight[i][j];
          count++;
        }
        transition_index_beta[i].second = count - 1;
      }
    }

    if (count != num_arcs) {
      std::cerr << "[GPU-DEN] Init count does not equal to num_arcs" << std::endl;
      return CRF_STATUS_UNKNOWN_ERROR;
    }

    int max_gpu = 0;
    for (int i = 0; i < n_gpus; i++) {
      if (gpus[i] > max_gpu) { max_gpu = gpus[i]; }
    }

    kDeviceHash = new int[max_gpu + 1];
    memset(kDeviceHash, 0, sizeof(int) * (max_gpu + 1));
    for (int i = 0; i < n_gpus; i++) { kDeviceHash[gpus[i]] = i; }

    kTransitionAlpha = new Transition*[n_gpus];
    kTransitionBeta = new Transition*[n_gpus];
    kTransitionIndexAlpha = new IntPair*[n_gpus];
    kTransitionIndexBeta = new IntPair*[n_gpus];
    kStartWeight = new float*[n_gpus];
    kEndWeight = new float*[n_gpus];

    std::cout << "max_gpu: " << max_gpu << std::endl;
    std::cout << "n_gpus: " << n_gpus << std::endl;

    int prev_device = 0;
    CHECK_CUDA(cudaGetDevice(&prev_device));
    for (int i = 0; i < n_gpus; i++) {
      std::cout << "gpus[i]: " << gpus[i] << std::endl;
      CHECK_CUDA(cudaSetDevice(gpus[i]));
      CHECK_CUDA(cudaMalloc((void**)&kTransitionAlpha[i], sizeof(Transition)*num_arcs));
      CHECK_CUDA(cudaMalloc((void**)&kTransitionBeta[i], sizeof(Transition)*num_arcs));
      CHECK_CUDA(cudaMalloc((void**)&kTransitionIndexAlpha[i], sizeof(IntPair)*num_states));
      CHECK_CUDA(cudaMalloc((void**)&kTransitionIndexBeta[i], sizeof(IntPair)*num_states));
      CHECK_CUDA(cudaMalloc((void**)&kStartWeight[i], sizeof(float)*num_states));
      CHECK_CUDA(cudaMalloc((void**)&kEndWeight[i], sizeof(float)*num_states));

      CHECK_CUDA(cudaMemcpy(kTransitionAlpha[i], transition_alpha.data(), sizeof(Transition)*num_arcs, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(kTransitionBeta[i], transition_beta.data(), sizeof(Transition)*num_arcs, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(kTransitionIndexAlpha[i], transition_index_alpha.data(), sizeof(IntPair)*num_states, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(kTransitionIndexBeta[i], transition_index_beta.data(), sizeof(IntPair)*num_states, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(kStartWeight[i], start_weight.data(), sizeof(float)*num_states, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(kEndWeight[i], end_weight.data(), sizeof(float)*num_states, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaSetDevice(prev_device));

    return CRF_STATUS_SUCCESS;
  }

  crfStatus_t crfRelease(const int n_gpus, const int* const gpus) {
    int prev_device = 0;
    CHECK_CUDA(cudaGetDevice(&prev_device));

    for (int i = 0; i < n_gpus; i++) {
      CHECK_CUDA(cudaSetDevice(gpus[i]));

      CHECK_CUDA(cudaFree(kTransitionAlpha[i]));
      CHECK_CUDA(cudaFree(kTransitionBeta[i]));
      CHECK_CUDA(cudaFree(kTransitionIndexAlpha[i]));
      CHECK_CUDA(cudaFree(kTransitionIndexBeta[i]));
      CHECK_CUDA(cudaFree(kStartWeight[i]));
      CHECK_CUDA(cudaFree(kEndWeight[i]));
    }
    CHECK_CUDA(cudaSetDevice(prev_device));
    delete[] kTransitionAlpha;
    delete[] kTransitionBeta;
    delete[] kTransitionIndexAlpha;
    delete[] kTransitionIndexBeta;
    delete[] kStartWeight;
    delete[] kEndWeight;

    kTransitionAlpha = NULL;
    kTransitionBeta = NULL;
    kTransitionIndexAlpha = NULL;
    kTransitionIndexBeta = NULL;
    kStartWeight = NULL;
    kEndWeight = NULL;

    delete[] kDeviceHash;
    kDeviceHash = NULL;
    return CRF_STATUS_SUCCESS;
  }

  crfStatus_t crf_get_workspace_size(int max_time, int alphabet_size, int minibatch, size_t* size_bytes) {
    if (size_bytes == nullptr ||
        alphabet_size <= 0 ||
        max_time <= 0 ||
        minibatch <= 0) {
      return CRF_STATUS_INVALID_VALUE;
    }

    *size_bytes = 0;

    // loglikelihood
    *size_bytes += sizeof(float) * minibatch;

    // // loglikelihood_verify
    // *size_bytes += sizeof(float) * minibatch;

    // alpha
    *size_bytes += sizeof(float) * (max_time + 1) * minibatch * kDenNumStates;

    // beta
    *size_bytes += sizeof(float) * 2 * minibatch * kDenNumStates;

    // grad_storage
    *size_bytes += sizeof(float) * ATOMIC_CONST * minibatch * alphabet_size;

    //utt_length
    *size_bytes += sizeof(int) * minibatch;

    return CRF_STATUS_SUCCESS;
  }

  crfStatus_t crf_compute(const float* const logits,
                          const int minibatch,
                          const int max_time,
                          const int alphabet_size,
                          const int* const input_lengths,
                          float* loglikelihood,
                          // float* loglikelihood_verify,
                          float* grad,
                          void* workspace,
                          CRFCUstream stream) {
    if (logits == nullptr ||
        input_lengths == nullptr ||
        loglikelihood == nullptr ||
        // loglikelihood_verify == nullptr ||
        workspace == nullptr ||
        minibatch <= 0 ||
        max_time <= 0 ||
        kDenNumStates <= 0 ||
        alphabet_size <= 0
       ) {
      return CRF_STATUS_INVALID_VALUE;
    }
    // std::cout << "minibatch:" << minibatch << std::endl;
    // std::cout << "max_time:" << max_time << std::endl;
    // std::cout << "alphabet_size:" << alphabet_size << std::endl;
    // std::cout << "input_lengths[0]:" << input_lengths[0] << std::endl;
    // std::cout << "kDenNumStates:" << kDenNumStates << std::endl;

    GpuCRF crf(minibatch, max_time, alphabet_size, kDenNumStates, workspace,
               kTransitionAlpha, kTransitionBeta,
               kTransitionIndexAlpha, kTransitionIndexBeta,
               kStartWeight, kEndWeight,
               kDeviceHash,
               stream);

    return crf.cost_and_grad(logits, 
                            input_lengths, 
                            loglikelihood,
                            // loglikelihood_verify, 
                            grad);
  }

} /* extern "C" { */
