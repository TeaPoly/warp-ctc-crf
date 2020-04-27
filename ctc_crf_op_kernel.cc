#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <algorithm>

#include "ctc.h"
#include "crf.h"

namespace tf = tensorflow;

namespace ctc_crf {

REGISTER_OP("CtcCrfInit")
.Attr("gpus: list(int)")
.Attr("fst_name: string");

class CTCCRFInitOp : public tf::OpKernel {
 public:
  explicit CTCCRFInitOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fst_name", &fst_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gpus", &gpus_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    auto warp_status = crfInit(fst_name_.c_str(), gpus_.size(), gpus_.data());
    OP_REQUIRES(ctx, warp_status == CRF_STATUS_SUCCESS,
                tf::errors::Internal("ctc_crf_loss CTCCRFInitOp error: ",
                                     crfGetStatusString(warp_status)));
  }

 private:
  std::string fst_name_;
  std::vector<int> gpus_;
};

REGISTER_KERNEL_BUILDER(Name("CtcCrfInit").Device(::tensorflow::DEVICE_CPU), CTCCRFInitOp);

REGISTER_OP("CtcCrfRelease")
.Attr("gpus: list(int)");

class CTCCRFReleaseOp : public tf::OpKernel {
 public:
  explicit CTCCRFReleaseOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gpus", &gpus_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    auto warp_status = crfRelease(gpus_.size(), gpus_.data());
    OP_REQUIRES(ctx, warp_status == CRF_STATUS_SUCCESS,
                tf::errors::Internal("ctc_crf_loss CTCCRFReleaseOp error: ",
                                     crfGetStatusString(warp_status)));
  }
 private:
  std::vector<int> gpus_;
};

REGISTER_KERNEL_BUILDER(Name("CtcCrfRelease").Device(::tensorflow::DEVICE_CPU), CTCCRFReleaseOp);

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::Status;

REGISTER_OP("CtcCrfLoss")
.Input("time_major_logsoftmax: float32")
.Input("batch_major_logsoftmax: float32")
.Input("labels_indices: int64")
.Input("labels_values: int32")
.Input("input_lengths: int32")
.Attr("lamb: float = 0.1")
.Attr("blank_label: int = 0")
.Output("log_likelihood_ctc: float32")
.Output("grad_ctc: float32")
.Output("grad_den: float32")
.Output("log_likelihood_den: float32")
// .Output("log_likelihood_den_verfiy: float32")
.SetShapeFn([](InferenceContext* c) {

  ShapeHandle time_major_logsoftmax;
  ShapeHandle batch_major_logsoftmax;
  ShapeHandle labels_indices;
  ShapeHandle labels_values;
  ShapeHandle input_lengths;

  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &time_major_logsoftmax));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &batch_major_logsoftmax));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &labels_indices));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &labels_values));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &input_lengths)); // log_likelihood_den_verfiy

  // Get batch size from inputs and sequence_length, and update inputs
  // with the merged batch_size since it is returned.
  DimensionHandle batch_size;
  TF_RETURN_IF_ERROR(
    c->Merge(c->Dim(time_major_logsoftmax, 1), c->Dim(input_lengths, 0), &batch_size));
  TF_RETURN_IF_ERROR(c->ReplaceDim(time_major_logsoftmax, 1, batch_size, &time_major_logsoftmax));

  c->set_output(0, c->Vector(batch_size));
  c->set_output(1, time_major_logsoftmax);
  c->set_output(2, batch_major_logsoftmax);
  c->set_output(3, c->Vector(batch_size));
  // c->set_output(4, c->Vector(batch_size)); // log_likelihood_den_verfiy

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
    options.blank_label = blank_label_;

    // Grab the input tensors
    const tf::Tensor* time_major_logsoftmax;
    const tf::Tensor* labels_indices;
    const tf::Tensor* labels_values;
    const tf::Tensor* input_lengths;
    const tf::Tensor* batch_major_logsoftmax;

    OP_REQUIRES_OK(ctx, ctx->input("time_major_logsoftmax", &time_major_logsoftmax));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths));
    OP_REQUIRES_OK(ctx, ctx->input("batch_major_logsoftmax", &batch_major_logsoftmax));

    OP_REQUIRES(ctx, time_major_logsoftmax->shape().dims() == 3,
                tf::errors::InvalidArgument("ctc_crf_loss time_major_logsoftmax is not a 3-Tensor"));
    auto time_major_logsoftmax_t = time_major_logsoftmax->tensor<float, 3>();
    OP_REQUIRES(ctx, batch_major_logsoftmax->shape().dims() == 3,
                tf::errors::InvalidArgument("ctc_crf_loss batch_major_logsoftmax is not a 3-Tensor"));
    auto batch_major_logsoftmax_t = batch_major_logsoftmax->tensor<float, 3>();

    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(labels_indices->shape()),
                tf::errors::InvalidArgument("ctc_crf_loss labels_indices is not a matrix"));
    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(labels_values->shape()),
                tf::errors::InvalidArgument("ctc_crf_loss labels_values is not a vector"));
    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lengths->shape()),
                tf::errors::InvalidArgument("ctc_crf_loss input_lengths is not a vector"));

    const auto& time_major_logsoftmax_shape = time_major_logsoftmax->shape();
    const auto max_time = time_major_logsoftmax_shape.dim_size(0);
    const auto batch_size = time_major_logsoftmax_shape.dim_size(1);
    const auto num_classes_raw = time_major_logsoftmax_shape.dim_size(2);

    // std::cout << "# max_time: " << max_time << std::endl;
    // std::cout << "# batch_size: " << batch_size << std::endl;
    // std::cout << "# num_classes_raw: " << num_classes_raw << std::endl;

    OP_REQUIRES(
      ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
      tf::errors::InvalidArgument("ctc_crf_loss num_classes cannot exceed max int"));
    const auto alphabet_size = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
      ctx, batch_size == input_lengths->dim_size(0),
      tf::errors::InvalidArgument("ctc_crf_loss len(input_lengths) != batch_size.  ",
                                  "len(input_length):  ", input_lengths->dim_size(0),
                                  " batch_size: ", batch_size));
    auto input_lengths_t = input_lengths->vec<int32_t>();

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                tf::errors::InvalidArgument(
                  "ctc_crf_loss labels_indices and labels_values must contain the "
                  "same number of rows, but saw shapes: ",
                  labels_indices->shape().DebugString(), " vs. ",
                  labels_values->shape().DebugString()));

    // get label length and label flat.
    auto labels_shape = tf::TensorShape({batch_size, max_time});
    auto order = std::vector<tf::int64> {0, 1};
    auto labels_sp = tf::sparse::SparseTensor(*labels_indices, *labels_values,
                     labels_shape, order);
    auto labels_sp_valid = labels_sp.IndicesValid();
    OP_REQUIRES(ctx, labels_sp_valid.ok(),
                tf::errors::InvalidArgument("ctc_crf_loss label SparseTensor is not valid: ",
                                            labels_sp_valid.error_message()));

    auto label_lengths = std::vector<int> {};
    for (const auto& g : labels_sp.group( {0})) { // iterate by batch
      const auto batch_indices = g.group()[0];
      OP_REQUIRES(ctx, tf::FastBoundsCheck(batch_indices, batch_size),
                  tf::errors::InvalidArgument("ctc_crf_loss labels batch index must be between ",
                                              0, " and ", batch_size, " but saw: ",
                                              batch_indices));

      auto values = g.values<int32_t>();
      label_lengths.push_back(values.size());
    }
    auto label_values_t = labels_values->vec<int>();

    // check that labels are in the alphabet?
    for (int b = 0; b < batch_size; b++) {
      OP_REQUIRES(ctx, input_lengths_t(b) <= max_time,
                  tf::errors::InvalidArgument("ctc_crf_loss input_lengths(", b, ") <= ", max_time));
    }

    // allocate output memory.
    tf::Tensor* log_likelihood_ctc = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("log_likelihood_ctc", input_lengths->shape(), &log_likelihood_ctc));
    auto log_likelihood_ctc_t = log_likelihood_ctc->vec<float>();

    tf::Tensor* grad_ctc = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("grad_ctc", time_major_logsoftmax->shape(),
                   &grad_ctc));
    set_zero(grad_ctc);
    auto grad_ctc_t = grad_ctc->tensor<float, 3>();

    tf::Tensor* grad_den = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("grad_den", batch_major_logsoftmax->shape(), &grad_den));
    set_zero(grad_den);
    auto grad_den_t = grad_den->tensor<float, 3>();

    tf::Tensor* log_likelihood_den = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("log_likelihood_den", input_lengths->shape(), &log_likelihood_den));
    auto log_likelihood_den_t = log_likelihood_den->vec<float>();

    // tf::Tensor* log_likelihood_den_verfiy = nullptr;
    // OP_REQUIRES_OK(ctx,
    //                ctx->allocate_output("log_likelihood_den_verfiy", input_lengths->shape(), &log_likelihood_den_verfiy));
    // auto log_likelihood_den_verfiy_t = log_likelihood_den_verfiy->vec<float>();

    // compute CTC.
    // allocate temp memory.
    tf::Tensor workspace;
    size_t workspace_size_bytes;
    auto warp_status = get_workspace_size(label_lengths.data(),
                                          input_lengths_t.data(),
                                          alphabet_size, batch_size,
                                          options, &workspace_size_bytes);
    OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                tf::errors::Internal("ctc_crf_loss error in get_workspace_size: ",
                                     ctcGetStatusString(warp_status)));
    auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
    auto workspace_t = workspace.flat<uint8_t>();

    warp_status = compute_ctc_loss(time_major_logsoftmax_t.data(),
                                   grad_ctc_t.data(),
                                   label_values_t.data(),
                                   label_lengths.data(),
                                   input_lengths_t.data(),
                                   alphabet_size,
                                   batch_size,
                                   log_likelihood_ctc_t.data(),
                                   workspace_t.data(),
                                   options);

    OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                tf::errors::Internal("ctc_crf_loss error in compute_ctc_loss: ",
                                     ctcGetStatusString(warp_status)));

    if (lamb_ >= 0) {
      // compute DEN.
      // allocate temp memory.
      tf::Tensor workspace_den;
      size_t workspace_den_size_bytes;
      auto crf_warp_status = crf_get_workspace_size(max_time, alphabet_size, batch_size, &workspace_den_size_bytes);
      OP_REQUIRES(ctx, crf_warp_status == CRF_STATUS_SUCCESS,
                  tf::errors::Internal("den error in crf_get_workspace_size: ",
                                       crfGetStatusString(crf_warp_status)));
      auto workspace_den_shape = tf::TensorShape{static_cast<int64_t>(workspace_den_size_bytes)};
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_den_shape, &workspace_den));
      auto workspace_den_t = workspace_den.flat<uint8_t>();

      crf_warp_status = crf_compute(batch_major_logsoftmax_t.data(),
                                    batch_size,
                                    max_time,
                                    alphabet_size,
                                    input_lengths_t.data(),
                                    log_likelihood_den_t.data(),
                                    // log_likelihood_den_verfiy_t.data(),
                                    grad_den_t.data(),
                                    workspace_den_t.data(),
                                    options.stream);

      OP_REQUIRES(ctx, crf_warp_status == CRF_STATUS_SUCCESS,
                  tf::errors::Internal("ctc_crf_loss error in crf_compute: ",
                                       crfGetStatusString(crf_warp_status)));
    }
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
                        .HostMemory("log_likelihood_ctc")
                        .HostMemory("log_likelihood_den")
                        // .HostMemory("log_likelihood_den_verfiy")
                        ,
                        CTCCRFLossOpGPU);
}

#undef EIGEN_USE_GPU
