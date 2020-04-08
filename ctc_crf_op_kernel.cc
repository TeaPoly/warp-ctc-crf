#define EIGEN_USE_GPU
#include <cuda.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "ctc.h"

#include "tensorflow/core/framework/shape_inference.h"

extern int DEN_NUM_ARCS;
extern int DEN_NUM_STATES;

#undef ATOMIC_CONST
#define ATOMIC_CONST 32

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

void Init(const char* fst_name, const int n_gpus, const int* const gpus);

void Release(const int n_gpus, const int* const gpus);

void compute_alpha(float* alpha,
                   const float* const logits,
                   const int batch_size,
                   int T,
                   const int alpha_size,
                   int logits_size,
                   const int* const input_lengths,
                   float* loglikelihood,
                   cudaStream_t stream);

// FIXME(huanglk): 20200401, Not using.
// float* loglikelihood,
void compute_beta_and_grad(float* beta,
                           const float* const alpha,
                           const float* const logits,
                           const float* const alpha_lld,
                           float* grad_storage,
                           float* grad_net,
                           const int batch_size,
                           const int T,
                           const int beta_size,
                           const int logits_size,
                           const int* const input_lengths,
                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif

namespace tf = tensorflow;

namespace ctc_crf {

REGISTER_OP("CtcCrfInit")
.Input("gpus: int32")
.Attr("fst_name: string");

class CTCCRFInitOp : public tf::OpKernel {
 public:
  explicit CTCCRFInitOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fst_name", &fst_name_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    const tf::Tensor* gpus;
    OP_REQUIRES_OK(ctx, ctx->input("gpus", &gpus));
    auto gpus_t = gpus->vec<int32_t>();
    int n_gpus = gpus->shape().dim_size(0);
    Init(fst_name_.c_str(), n_gpus, gpus_t.data());
  }

 private:
  std::string fst_name_;
};

REGISTER_KERNEL_BUILDER(Name("CtcCrfInit").Device(::tensorflow::DEVICE_GPU).HostMemory("gpus"), CTCCRFInitOp);

REGISTER_OP("CtcCrfRelease")
.Input("gpus: int32");

class CTCCRFReleaseOp : public tf::OpKernel {
 public:
  explicit CTCCRFReleaseOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
  }

  void Compute(tf::OpKernelContext* ctx) override {
    const tf::Tensor* gpus;
    OP_REQUIRES_OK(ctx, ctx->input("gpus", &gpus));
    auto gpus_t = gpus->vec<int32_t>();
    int n_gpus = gpus->shape().dim_size(0);
    Release(n_gpus, gpus_t.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("CtcCrfRelease").Device(::tensorflow::DEVICE_GPU).HostMemory("gpus"), CTCCRFReleaseOp);

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::Status;

// FIXME(huanglk): 20200401, Not using.
// .Output("costs_beta: float32")

REGISTER_OP("CtcCrfLoss")
.Input("time_major_activations: float32")
.Input("activations: float32")
.Input("labels_indices: int64")
.Input("labels_values: int32")
.Input("input_lengths: int32")
.Attr("lamb: float = 0.1")
.Attr("blank_label: int = 0")
.Output("costs: float32")
.Output("gradients: float32")
.Output("grad_net: float32")
.Output("costs_alpha: float32")
.SetShapeFn([](InferenceContext* c) {

  ShapeHandle time_major_activations;
  ShapeHandle activations;
  ShapeHandle labels_indices;
  ShapeHandle labels_values;
  ShapeHandle input_lengths;

  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &time_major_activations));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &activations));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &labels_indices));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &labels_values));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &input_lengths));

  // Get batch size from inputs and sequence_length, and update inputs
  // with the merged batch_size since it is returned.
  DimensionHandle batch_size;
  TF_RETURN_IF_ERROR(
    c->Merge(c->Dim(time_major_activations, 1), c->Dim(input_lengths, 0), &batch_size));
  TF_RETURN_IF_ERROR(c->ReplaceDim(time_major_activations, 1, batch_size, &time_major_activations));

  c->set_output(0, c->Vector(batch_size));
  c->set_output(1, time_major_activations);
  c->set_output(2, activations);
  c->set_output(3, c->Vector(batch_size));

  // FIXME(huanglk): 20200401, Not using.
  // c->set_output(4, c->Vector(batch_size));

  return Status::OK();
});

class CTCCRFLossOpBase : public tf::OpKernel {
 public:
  explicit CTCCRFLossOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lamb", &lamb_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    auto options = create_options(ctx);

    // Grab the input tensors
    const tf::Tensor* time_major_activations;
    const tf::Tensor* labels_indices;
    const tf::Tensor* labels_values;
    const tf::Tensor* input_lengths;
    const tf::Tensor* activations;

    OP_REQUIRES_OK(ctx, ctx->input("time_major_activations", &time_major_activations));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths));
    OP_REQUIRES_OK(ctx, ctx->input("activations", &activations));

    OP_REQUIRES(ctx, time_major_activations->shape().dims() == 3,
                tf::errors::InvalidArgument("time_major_activations is not a 3-Tensor"));
    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(labels_indices->shape()),
                tf::errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(labels_values->shape()),
                tf::errors::InvalidArgument("labels_values is not a vector"));
    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lengths->shape()),
                tf::errors::InvalidArgument("input_lengths is not a vector"));
    OP_REQUIRES(ctx, activations->shape().dims() == 3,
                tf::errors::InvalidArgument("activations is not a 3-Tensor"));

    const auto& time_major_acts_shape = time_major_activations->shape();
    const auto max_time = time_major_acts_shape.dim_size(0);
    const auto batch_size = time_major_acts_shape.dim_size(1);
    const auto num_classes_raw = time_major_acts_shape.dim_size(2);
    const auto& batch_major_acts_shape = activations->shape();

    auto time_major_activations_t = time_major_activations->tensor<float, 3>();

    OP_REQUIRES(
      ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
      tf::errors::InvalidArgument("num_classes cannot exceed max int"));
    const auto alphabet_size = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
      ctx, batch_size == input_lengths->dim_size(0),
      tf::errors::InvalidArgument("len(input_lengths) != batch_size.  ",
                                  "len(input_length):  ", input_lengths->dim_size(0),
                                  " batch_size: ", batch_size));
    auto input_lengths_t = input_lengths->vec<int32_t>();

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                tf::errors::InvalidArgument(
                  "labels_indices and labels_values must contain the "
                  "same number of rows, but saw shapes: ",
                  labels_indices->shape().DebugString(), " vs. ",
                  labels_values->shape().DebugString()));

    auto labels_shape = tf::TensorShape({batch_size, max_time});
    auto order = std::vector<tf::int64> {0, 1};
    auto labels_sp = tf::sparse::SparseTensor(*labels_indices, *labels_values,
                     labels_shape, order);
    auto labels_sp_valid = labels_sp.IndicesValid();
    OP_REQUIRES(ctx, labels_sp_valid.ok(),
                tf::errors::InvalidArgument("label SparseTensor is not valid: ",
                                            labels_sp_valid.error_message()));

    auto label_lengths = std::vector<int> {};
    for (const auto& g : labels_sp.group( {0})) { // iterate by batch
      const auto batch_indices = g.group()[0];
      OP_REQUIRES(ctx, tf::FastBoundsCheck(batch_indices, batch_size),
                  tf::errors::InvalidArgument("labels batch index must be between ",
                                              0, " and ", batch_size, " but saw: ",
                                              batch_indices));

      auto values = g.values<int32_t>();
      label_lengths.push_back(values.size());
    }
    auto label_values_t = labels_values->vec<int>();

    // check that labels are in the alphabet?
    for (int b = 0; b < batch_size; b++) {
      OP_REQUIRES(ctx, input_lengths_t(b) <= max_time,
                  tf::errors::InvalidArgument("input_lengths(", b, ") <= ", max_time));
    }

    tf::Tensor* costs = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", input_lengths->shape(), &costs));
    auto costs_t = costs->vec<float>();

    tf::Tensor* grads = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("gradients", time_major_activations->shape(),
                   &grads));
    set_zero(grads);
    auto grads_t = grads->tensor<float, 3>();

    options.blank_label = blank_label_;

    size_t workspace_size_bytes;
    auto warp_status = get_workspace_size(label_lengths.data(),
                                          input_lengths_t.data(),
                                          alphabet_size, batch_size,
                                          options, &workspace_size_bytes);

    OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                tf::errors::Internal("warp_ctc error in get_workspace_size: ",
                                     ctcGetStatusString(warp_status)));

    auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
    tf::Tensor workspace;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
    auto workspace_t = workspace.flat<uint8_t>();

    // compute CTC
    warp_status = compute_ctc_loss(time_major_activations_t.data(),
                                   grads_t.data(),
                                   label_values_t.data(),
                                   label_lengths.data(),
                                   input_lengths_t.data(),
                                   alphabet_size, batch_size,
                                   costs_t.data(), workspace_t.data(), options);

    OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                tf::errors::Internal("warp_ctc error in compute_ctc_loss: ",
                                     ctcGetStatusString(warp_status)));

    // compute DEN
    tf::Tensor* grad_net = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("grad_net", batch_major_acts_shape, &grad_net));
    set_zero(grad_net);
    auto grad_net_t = grad_net->tensor<float, 3>();

    tf::Tensor* costs_alpha = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("costs_alpha", input_lengths->shape(), &costs_alpha));
    auto costs_alpha_t = costs_alpha->vec<float>();

    // FIXME(huanglk): 20200401, Not using.
    // tf::Tensor* costs_beta = nullptr;
    // OP_REQUIRES_OK(ctx, ctx->allocate_output("costs_beta", input_lengths->shape(), &costs_beta));
    // auto costs_beta_t = costs_beta->vec<float>();
    auto activations_t = activations->tensor<float, 3>();

    tf::Tensor alpha;
    auto alpha_shape = tf::TensorShape{static_cast<int64_t>((max_time + 1) * batch_size * DEN_NUM_STATES)};
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DataTypeToEnum<float>::value, alpha_shape, &alpha));
    auto alpha_t = alpha.vec<float>();

    tf::Tensor beta;
    auto beta_shape = tf::TensorShape{static_cast<int64_t>(2 * batch_size * DEN_NUM_STATES)};
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DataTypeToEnum<float>::value, beta_shape, &beta));
    auto beta_t = beta.vec<float>();

    tf::Tensor grad_storage;
    auto grad_shape = tf::TensorShape{static_cast<int64_t>(ATOMIC_CONST* batch_size * alphabet_size)};
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DataTypeToEnum<float>::value, grad_shape, &grad_storage));
    auto grad_storage_t = grad_storage.vec<float>();

    compute_alpha(alpha_t.data(),
                  activations_t.data(),
                  batch_size,
                  max_time,
                  DEN_NUM_STATES,
                  alphabet_size,
                  input_lengths_t.data(),
                  costs_alpha_t.data(), options.stream);

    compute_beta_and_grad(beta_t.data(),
                          alpha_t.data(),
                          activations_t.data(),
                          costs_alpha_t.data(),
                          grad_storage_t.data(),
                          grad_net_t.data(),
                          batch_size,
                          max_time,
                          DEN_NUM_STATES,
                          alphabet_size,
                          input_lengths_t.data(), options.stream);
    // FIXME(huanglk): 20200401, Not using.
    // costs_beta_t.data(),
  }

 private:
  float lamb_;
  int blank_label_;
  virtual void set_zero(tf::Tensor* t) = 0;
  virtual ctcOptions create_options(tf::OpKernelContext* ctx) = 0;
};

class CTCCRFLossOpGPU : public CTCCRFLossOpBase {
 public:
  explicit CTCCRFLossOpGPU(tf::OpKernelConstruction* ctx) : CTCCRFLossOpBase(ctx) {
  }

 private:
  void set_zero(tf::Tensor* t) override {
    cudaMemset(t->flat<float>().data(), 0, t->NumElements()*sizeof(float));
  }

  ctcOptions create_options(tf::OpKernelContext* ctx) override {
    auto cuda_stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
    auto options = ctcOptions{};
    options.stream = cuda_stream;
    return options;
  }
};

REGISTER_KERNEL_BUILDER(Name("CtcCrfLoss").Device(::tensorflow::DEVICE_GPU)
                        .HostMemory("labels_indices")
                        .HostMemory("labels_values")
                        .HostMemory("input_lengths")
                        .HostMemory("activations")
                        .HostMemory("costs")
                        .HostMemory("costs_alpha"),
                        CTCCRFLossOpGPU);

// FIXME(huanglk): 20200401, Not using.
// .HostMemory("costs_beta"),
}

#undef EIGEN_USE_GPU
