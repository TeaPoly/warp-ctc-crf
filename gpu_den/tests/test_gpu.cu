#include "../crf.h"

#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

inline static void throw_on_error(crfStatus_t status, const char* message) {
  if (status != CRF_STATUS_SUCCESS) {
    throw std::runtime_error(message + (", stat = " +
                                        std::string(crfGetStatusString(status))));
  }
}

inline static void throw_on_error(cudaError_t error, const char* message) {
  if (error) {
    throw thrust::system_error(error, thrust::cuda_category(), message);
  }
}

// static std::vector<float>
// genActs(int size) {
//   std::vector<float> arr(size);
//   std::mt19937 gen(0);
//   std::uniform_real_distribution<> dis(0, 1);
//   for (int i = 0; i < size; ++i) {
//     arr[i] = dis(gen);
//   }
//   return arr;
// }

// static std::vector<int>
// genLabels(int alphabet_size, int L) {
//   std::vector<int> label(L);

//   std::mt19937 gen(1);
//   std::uniform_int_distribution<> dis(1, alphabet_size - 1);

//   for (int i = 0; i < L; ++i) {
//     label[i] = dis(gen);
//   }
//   // guarantee repeats for testing
//   if (L >= 3) {
//     label[L / 2] = label[L / 2 + 1];
//     label[L / 2 - 1] = label[L / 2];
//   }
//   return label;
// }

// static float rel_diff(const std::vector<float>& grad,
//                const std::vector<float>& num_grad) {
//   float diff = 0.;
//   float tot = 0.;
//   for (size_t idx = 0; idx < grad.size(); ++idx) {
//     diff += (grad[idx] - num_grad[idx]) * (grad[idx] - num_grad[idx]);
//     tot += grad[idx] * grad[idx];
//   }

//   return diff / tot;
// }

// Numerically stable softmax for a minibatch of 1
static void log_softmax(const float* const acts,
                 int alphabet_size, int T,
                 float* probs) {

  for (int t = 0; t < T; ++t) {

    // float max_activation =
    //   -std::numeric_limits<float>::infinity();

    // for (int a = 0; a < alphabet_size; ++a)
    //   max_activation =
    //     std::max(max_activation, acts[t * alphabet_size + a]);

    float denom = 0;
    for (int a = 0; a < alphabet_size; ++a) {
      denom += std::exp(acts[t * alphabet_size + a]); //  - max_activation
    }

    for (int a = 0; a < alphabet_size; ++a) {
      probs[t * alphabet_size + a] =
        std::log(std::exp(acts[t * alphabet_size + a]) / denom); //  - max_activation
    }
  }
}

enum { alphabet_size = 218 };

bool small_test() {
    const int batch_size = 2;
    const int T = 2;

    float act[alphabet_size*T] = { 
      0.6659549,0.0706198,0.6259198,0.9417950,0.3073056,0.0626193,0.0854988,0.5743809,0.8985361,0.4634696,0.9069236,0.3761943,0.2976301,0.8777218,0.7042622,0.6778419,0.4311874,0.8057035,0.0907360,0.8996002,0.7308410,0.7396838,0.5338481,0.5025592,0.0494631,0.3042897,0.2045899,0.8200772,0.5412629,0.2979132,0.7797728,0.1439820,0.8446441,0.1672087,0.6306186,0.1249345,0.0187438,0.1735469,0.0628915,0.0411880,0.5104687,0.3104592,0.8063287,0.6219791,0.3521686,0.5289502,0.1306192,0.0796579,0.6828654,0.3899171,0.0998497,0.6851936,0.5352813,0.9509663,0.5225685,0.9741059,0.9162824,0.7937816,0.1956039,0.6719426,0.2711840,0.6592409,0.6182947,0.6084988,0.1303659,0.5016577,0.4635343,0.0145267,0.0489267,0.0995309,0.3323533,0.7156003,0.8131294,0.7331543,0.6361103,0.6444416,0.4472543,0.1680797,0.1297974,0.4833745,0.9962018,0.5436660,0.6265333,0.7909642,0.5090982,0.9815393,0.8259279,0.5057329,0.4013359,0.5210385,0.0715396,0.8733593,0.7783773,0.0477381,0.8672659,0.0945814,0.7299692,0.5741210,0.2302605,0.1884476,0.8046268,0.9613922,0.9954312,0.7298516,0.2151605,0.0455125,0.6614475,0.9975435,0.9596593,0.6666301,0.1679782,0.1610559,0.9205895,0.5002145,0.8713506,0.0454579,0.2227163,0.7837267,0.5874741,0.7059594,0.4860485,0.0594719,0.3790697,0.6197810,0.0430820,0.7563603,0.3651275,0.9772140,0.1428265,0.6381198,0.3364716,0.7592684,0.8764758,0.5439317,0.1361732,0.1965775,0.1330228,0.0442671,0.3402627,0.2868053,0.7664023,0.5227604,0.9124285,0.0805220,0.2753370,0.4023059,0.3215532,0.1589668,0.6564071,0.0152940,0.7201407,0.9447734,0.2671686,0.6744439,0.9005883,0.5793588,0.3133799,0.2263679,0.6893255,0.4702658,0.3339460,0.5029213,0.5431880,0.9632081,0.3518784,0.0994645,0.6197739,0.1804397,0.1424534,0.1210999,0.2074187,0.7301162,0.7330002,0.3443730,0.1793908,0.6507486,0.9559347,0.8578040,0.8736977,0.7831688,0.0572165,0.0963062,0.1748205,0.2937742,0.7889294,0.5656075,0.8252894,0.9331890,0.7628291,0.9811106,0.9147210,0.8637609,0.4862151,0.2595758,0.2774674,0.1037078,0.3203333,0.0744082,0.7371400,0.0564954,0.3671610,0.0351312,0.8896585,0.3360196,0.4913137,0.7083586,0.1043625,0.9565435,0.2778155,0.6950832,0.7173885,0.2766467,0.4206542,0.1095873,0.2694365,0.8567437,0.3375752,0.9855187
      ,
      0.5578931,0.4804044,0.3118265,0.5245464,0.6332869,0.1995611,0.3215812,0.9865746,0.3477587,0.1280169,0.0994182,0.9950141,0.6783327,0.0609730,0.5020059,0.5135063,0.8006369,0.5128694,0.4737951,0.4707350,0.8571222,0.9242232,0.9844769,0.9664838,0.2513129,0.2559852,0.5264842,0.1268266,0.0997906,0.8515209,0.6164192,0.0656983,0.1724206,0.0271587,0.5860362,0.3684284,0.3806152,0.9019353,0.5844208,0.9005370,0.4316886,0.7187292,0.5349603,0.7991169,0.1447082,0.3584106,0.4046971,0.7703515,0.5124529,0.2405102,0.8685294,0.9275674,0.0123095,0.8413261,0.6462959,0.5952583,0.9021689,0.2371418,0.0970380,0.8338151,0.5264531,0.7667438,0.7165620,0.4992088,0.5681107,0.4100059,0.3414189,0.2103459,0.1722168,0.2090131,0.4280712,0.5903218,0.4738508,0.2019919,0.8754460,0.6106824,0.5350101,0.7799936,0.9239641,0.9844203,0.1816834,0.8106236,0.2089193,0.9566391,0.4582093,0.7437538,0.0415671,0.8135289,0.5103018,0.4885974,0.5262952,0.6052835,0.1538476,0.9188785,0.5290305,0.3265282,0.0078114,0.0497387,0.4091345,0.5379531,0.9372357,0.4378975,0.9899122,0.9491029,0.8370127,0.7819784,0.2154676,0.0044671,0.2750414,0.5196161,0.5241804,0.7042552,0.9125771,0.1860544,0.6536873,0.0701264,0.4372029,0.8003593,0.0503716,0.1975532,0.6491239,0.4931818,0.5026495,0.3078116,0.1652866,0.1769626,0.2960937,0.0993892,0.3253632,0.4897227,0.8568098,0.3725121,0.8965799,0.5696692,0.9776480,0.8052380,0.4855493,0.9823304,0.1556187,0.1081211,0.1038485,0.2659554,0.4366129,0.2820219,0.8297300,0.4350556,0.7602876,0.0738857,0.6051488,0.4234479,0.3737955,0.8117978,0.2845039,0.9252389,0.2539751,0.1701375,0.9891652,0.5211643,0.5896762,0.8763697,0.5639211,0.5860334,0.8757634,0.1116636,0.1637479,0.4779717,0.0844546,0.4797719,0.1533458,0.5418778,0.2430188,0.9400853,0.5093962,0.9989519,0.8013016,0.7045762,0.2398756,0.1680408,0.1652720,0.8323123,0.4453561,0.4178149,0.2268084,0.6690651,0.8545278,0.0451003,0.0455554,0.0821302,0.8707637,0.9116980,0.3597097,0.7093065,0.5794834,0.3094945,0.2261248,0.3481374,0.4845700,0.8382207,0.2499820,0.5695265,0.8850192,0.8612077,0.0803505,0.8138733,0.4300781,0.4265665,0.9457681,0.8089249,0.0386239,0.5362303,0.0900319,0.8006067,0.5317548,0.5039142,0.6743562,0.1330474,0.0245858,0.9651904
    };

    std::vector<float> activations(act, act+sizeof(act)/sizeof(act[0]));

    // std::cout << "# activations.size(): " << activations.size() << std::endl;

    // Calculate the score analytically
    std::vector<float> log_prob(activations.size()*batch_size); 
    for (int i = 0; i < batch_size; i++) {
      log_softmax(activations.data(), alphabet_size, T, log_prob.data()+i*activations.size());
    }

    // float* lp = log_prob.data();
    // for (int b = 0; b < batch_size; ++b) {
    //   for (int t = 0; t < T; ++t) {
    //     for (int a = 0; a < alphabet_size; ++a) {
    //       std::cout << lp[b*T*alphabet_size+t*alphabet_size + a] << " ";
    //     }
    //     std::cout << std::endl;
    //   }
    //     std::cout << std::endl;
    // }

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *log_prob_gpu;
    throw_on_error(cudaMalloc(&log_prob_gpu,
                   log_prob.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(log_prob_gpu, log_prob.data(),
                                   log_prob.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> lengths;
    for (int i = 0; i < batch_size; i++) {
      lengths.push_back(T);
    }

    float score[batch_size] = { 0 };
    // float score_verf[batch_size] = { 0 };

    // std::cout << "# T: " << T << std::endl;
    // std::cout << "# alphabet_size: " << alphabet_size << std::endl;
    // std::cout << "# minibatch: " << lengths.size() << std::endl;

    size_t gpu_alloc_bytes;
    throw_on_error(crf_get_workspace_size(T, alphabet_size, lengths.size(), &gpu_alloc_bytes),
                   "Error: crf_get_workspace_size in small_test");
    // std::cout << "gpu_alloc_bytes: " << gpu_alloc_bytes << std::endl;

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    float *grad;
    throw_on_error(cudaMalloc(&grad, sizeof(float)*lengths.size()*T*alphabet_size),
                   "cudaMalloc");

    /* -3.4989526 */
    throw_on_error(crf_compute(log_prob_gpu,
                        lengths.size(),
                        T,
                        alphabet_size,
                        lengths.data(),
                        score,
                        // score_verf,
                        grad,
                        ctc_gpu_workspace,
                        stream),
                   "Error: crf_compute in small_test");

    throw_on_error(cudaFree(grad),
                   "cudaFree grad ");
    throw_on_error(cudaFree(log_prob_gpu),
                   "cudaFree log_prob_gpu");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree ctc_gpu_workspace");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy stream");

    for (int i = 0; i < batch_size; i++) {
      std::cout << "i: " << i << std::endl;
      std::cout << "\tscore: " << std::setprecision(10) << score[i] << std::endl;
      // std::cout << "\tscore_verf: " << std::setprecision(10) << score_verf[i] << std::endl;
    }

    return true;
}

int main(void) {
    const char fst_name[] = "/ark/repo/ctc_crf/egs/aishell/data/den_meta/den_lm.fst";
    int gpus[4] = {0};

    throw_on_error(crfInit(fst_name, 1, gpus),
                   "crfInit");

    std::cout << "Running GPU tests" << std::endl;
    throw_on_error(cudaSetDevice(0), "cudaSetDevice");

    bool status = true;
    status &= small_test();

    throw_on_error(crfRelease(1, gpus),
                   "crfRelease");

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}
