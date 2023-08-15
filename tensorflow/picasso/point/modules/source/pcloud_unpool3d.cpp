#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

// for the unpooling modules, we have in_mpoint<out_npoint
REGISTER_OP("WeightedInterpolate")
    .Input("input: float32")   // input features: concat_Mp * in_channels
    .Input("weight: float32")  // weights: concat_Np * 3
    .Input("nn_index: int32")  // neighbor indices: concat_Np * 3
    .Output("output: float32") // unpooled features: concat_Np * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        ::tensorflow::shape_inference::ShapeHandle nn_index;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &nn_index));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nn_index, 0), c->Dim(input, 1)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("WeightedInterpolateGrad")
    .Input("input: float32")        // input features: concat_Mp * in_channels
    .Input("grad_output: float32")  // gradient of unpooled features: concat_Np * in_channels
    .Input("weight: float32")       // weights: concat_Np * 3
    .Input("nn_index: int32")       // neighbor indices: concat_Np * 3
    .Output("grad_input: float32")  // gradient of input features: concat_Mp * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void weightedInterpolateLauncher(int Np, int Mp, int C, int K, const int* nnIndex,
                                 const float* input, const float* weight, float* output);
class WeightedInterpolateGpuOp : public OpKernel {
    public:
        explicit WeightedInterpolateGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& weight_tensor   = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);

            // get the dims required by computations
            int Mp = input_tensor.shape().dim_size(0);     // number of input points
            int C  = input_tensor.shape().dim_size(1);     // number of input channels
            int Np = nn_index_tensor.shape().dim_size(0); // number of output points
            int K  = nn_index_tensor.shape().dim_size(1);  // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("rank of nn_index should be 2, i.e. (Nout,K)"));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto weight_flat   = weight_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Np,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in     = &(input_flat(0));
            const float* weight = &(weight_flat(0));
            const int* nnIndex  = &(nn_index_flat(0));

            float* out = &(output_flat(0));
            cudaMemset(out, 0, sizeof(float)*Np*C);
            weightedInterpolateLauncher(Np, Mp, C, K, nnIndex, in, weight, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("WeightedInterpolate").Device(DEVICE_GPU), WeightedInterpolateGpuOp);


void weightedInterpolateGradLauncher(int Np, int Mp, int C,  int K, const int* nnIndex,
                                     const float* gradOutput, const float* weight, float* gradInput);
class WeightedInterpolateGradGpuOp : public OpKernel {
    public:
        explicit WeightedInterpolateGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
           // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& weight_tensor = context->input(2);
            const Tensor& nn_index_tensor = context->input(3);

            // get the dims required by computations
            int Mp = input_tensor.shape().dim_size(0);        // number of input points
            int C  = input_tensor.shape().dim_size(1);        // number of input channels
            int Np = grad_output_tensor.shape().dim_size(0);  // number of output points
            int K  = nn_index_tensor.shape().dim_size(1);     // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("rank of nn_index should be 2, i.e. (Nout,K)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto weight_flat      = weight_tensor.flat<float>();
            auto nn_index_flat    = nn_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const float* weight  = &(weight_flat(0));
            const int* nnIndex   = &(nn_index_flat(0));

            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*Mp*C);
            weightedInterpolateGradLauncher(Np, Mp, C, K, nnIndex, gradOut, weight, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("WeightedInterpolateGrad").Device(DEVICE_GPU), WeightedInterpolateGradGpuOp);