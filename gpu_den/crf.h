/** \file crf.h
 * Contains a simple C interface to call fast GPU based computation
 * of the CRF loss.
 */

#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

//forward declare of CUDA typedef to avoid needing to pull in CUDA headers
typedef struct CUstream_st* CRFCUstream;

typedef enum {
  CRF_STATUS_SUCCESS = 0,
  CRF_STATUS_MEMOPS_FAILED = 1,
  CRF_STATUS_INVALID_VALUE = 2,
  CRF_STATUS_EXECUTION_FAILED = 3,
  CRF_STATUS_UNKNOWN_ERROR = 4
} crfStatus_t;

struct Transition {
  float weight = -float(INFINITY);
  int label = 0;
  int state = 0;
};

struct IntPair {
  int first = 1;
  int second = 0;
};

/* CUDA */
#define CHECK_CUDA(call) \
{  \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        return CRF_STATUS_EXECUTION_FAILED; \
    } \
}

#define CU_BLOCK_DIM 1024
#define ATOMIC_CONST 32

/** Returns a string containing a description of status that was passed in
 *  \param[in] status identifies which string should be returned
 *  \return C style string containing the text description
 *  */
const char* crfGetStatusString(crfStatus_t status);

crfStatus_t crfInit(const char* fst_name, const int n_gpus, const int* const gpus);

crfStatus_t crfRelease(const int n_gpus, const int* const gpus);

/** Compute the CRF loss.
 * \param [in] activations pointer to the activations in either CPU or GPU
 *             addressable memory, depending on info. We assume a fixed
 *             memory layout for this 3 dimensional tensor, which has dimension
 *             (b, t, v), where b is the minibatch index, t is the time index,
 *             and v indexes over probabilities of each symbol in the alphabet
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [in]  max_time is maximum length of time.
 * \param [in]  alphabet_size The number of possible output symbols.  There
 *              should be this many probabilities for each time step.
 * \param [in]  input_lengths Always in CPU memory. The number of time steps
 *              for each sequence in the minibatch.
 * \param [out] loglikelihood Always in GPU memory. The cost of each example in the
 *              minibatch.
 * \param [out] loglikelihood_verify Always in GPU memory. The cost of each example in the
 *              minibatch.
 * \param [out] grad if not NULL, then grad are computed.  Should be
 *              allocated in the same memory space as probs and memory
 *              ordering is identical.
 * \param [in,out] workspace In same memory space as probs. Should be of
 *                 size requested by crf_get_workspace_size.
 * \param [in] stream is an asynchronous stream.
 *
 *  \return Status information
 *
 * */
crfStatus_t crf_compute(const float* const logits,
                        const int minibatch,
                        const int max_time,
                        const int alphabet_size,
                        const int* const input_lengths,
                        float* loglikelihood,
                        // float* loglikelihood_verify,
                        float* grad,
                        void* workspace,
                        CRFCUstream stream);

/** For a given set of max sequence length and minibatch size return the required
 *  workspace size. This will need to be allocated in the same memory space as your
 *  probabilities.
 * \param [in]  max_time is maximum length of time.
 * \param [in]  alphabet_size The number of possible output symbols.  There
 *              should be this many probabilities for each time step.
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [out] size_bytes is pointer to a scalar where the memory
 *              requirement in bytes will be placed. This memory should be allocated
 *              at the same place, CPU or GPU, that the probs are in
 *
 *  \return Status information
 **/
crfStatus_t crf_get_workspace_size(int max_time, int alphabet_size, int minibatch, size_t* size_bytes);

#ifdef __cplusplus
}
#endif
