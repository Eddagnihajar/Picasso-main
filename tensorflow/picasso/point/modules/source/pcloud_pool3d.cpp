#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("MaxPool3d")
    .Input("input: float32")    // input features: concat_Np * in_channels
    .Input("nn_count: int32")   // number of neighbors: concat_Mp
    .Input("nn_index: int32")   // neighbor and kernel bin indices: Nout * 2
    .Output("output: float32")  // pooled features: concat_Mp * in_channels
    .Output("max_index: int32") // the neighbor gives maximum activation: concat_Mp * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &nn_count));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nn_count, 0), c->Dim(input, 1)});
        c->set_output(0, output);
        c->set_output(1, output);
        return Status::OK();
    });
REGISTER_OP("MaxPool3dGrad")
    .Input("input: float32")       // input features: concat_Np * in_channels
    .Input("grad_output: float32") // gradient of pooled features: concat_Mp * in_channels
    .Input("max_index: int32")     // the neighbor gives maximum activation: concat_Mp * in_channels
    .Output("grad_input: float32") // gradient of input features: concat_Np * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
REGISTER_OP("AvgPool3d")
    .Input("input: float32")   // input features: concat_Np * in_channels
    .Input("nn_count: int32")  // number of neighbors: concat_Mp
    .Input("nn_index: int32")  // neighbor and kernel bin indices: Nout * 2
    .Output("output: float32") // pooled features: concat_Mp * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &nn_count));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nn_count, 0), c->Dim(input, 1)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("AvgPool3dGrad")
    .Input("input: float32")       // input features: concat_Np * in_channels
    .Input("grad_output: float32") // gradient of pooled features: concat_Mp * in_channels
    .Input("nn_count: int32")      // number of neighbors: concat_Mp
    .Input("nn_index: int32")      // neighbor and kernel bin indices: Nout * 2
    .Output("grad_input: float32") // gradient of input features: concat_Np * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void maxPool3dLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                       const float* input, float* output, int* maxIndex);
class MaxPool3dGpuOp : public OpKernel {
    public:
        explicit MaxPool3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& nn_count_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);    // number of input points
            int C    = input_tensor.shape().dim_size(1);    // number of input channels
            int Mp   = nn_count_tensor.shape().dim_size(0); // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0); // number of neighbor pairs

            OP_REQUIRES(context, nn_count_tensor.dims()==1,
                        errors::InvalidArgument("R of nn_count should be 1."));
            OP_REQUIRES(context, nn_index_tensor.dims()==2 && nn_index_tensor.shape().dim_size(1)==2,
                        errors::InvalidArgument("Shape of database points requires to be (Nout,2)."));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            Tensor* max_index_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp,C}, &output_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Mp,C}, &max_index_tensor));
            auto output_flat = output_tensor->flat<float>();
            auto max_index_flat = max_index_tensor->flat<int32>();

            const float* in    = &(input_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* nnIndex = &(nn_index_flat(0));

            float* out    = &(output_flat(0));
            int* maxIndex = &(max_index_flat(0));
            cudaMemset(out, 0, sizeof(float)*Mp*C);
            cudaMemset(maxIndex, 0, sizeof(int)*Mp*C);
            maxPool3dLauncher(Np, Mp, Nout, C, nnCount, nnIndex, in, out, maxIndex);
        }
};
REGISTER_KERNEL_BUILDER(Name("MaxPool3d").Device(DEVICE_GPU), MaxPool3dGpuOp);


void maxPool3dGradLauncher(int Np, int Mp, int C, const int* maxIndex, const float* gradOutput, float* gradInput);
class MaxPool3dGradGpuOp : public OpKernel {
    public:
        explicit MaxPool3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor       = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& max_index_tensor   = context->input(2);

            // get the dims required by computations
            int Np = input_tensor.shape().dim_size(0);       // number of input points
            int C  = input_tensor.shape().dim_size(1);       // number of input channels
            int Mp = grad_output_tensor.shape().dim_size(0); // number of output points

            OP_REQUIRES(context, max_index_tensor.dims()==2,
                        errors::InvalidArgument("rank of max_index should be 2, i.e. (Mp, in_channels)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto max_index_flat   = max_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Np,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* maxIndex  = &(max_index_flat(0));

            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*Np*C);
            maxPool3dGradLauncher(Np, Mp, C, maxIndex, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("MaxPool3dGrad").Device(DEVICE_GPU), MaxPool3dGradGpuOp);


void avgPool3dLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                       const float* input, float* output);
class AvgPool3dGpuOp : public OpKernel {
    public:
        explicit AvgPool3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& nn_count_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);    // number of input points
            int C    = input_tensor.shape().dim_size(1);    // number of input channels
            int Mp   = nn_count_tensor.shape().dim_size(0); // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0); // number of neighbor pairs

            OP_REQUIRES(context, nn_count_tensor.dims()==1,
                        errors::InvalidArgument("Rank of nn_count should be 1."));
             OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("Shape of nn_index should be (Nout,2)."));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in    = &(input_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* nnIndex = &(nn_index_flat(0));

            float* out = &(output_flat(0));
            cudaMemset(out, 0, sizeof(float)*Mp*C);
            avgPool3dLauncher(Np, Mp, Nout, C, nnCount, nnIndex, in, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("AvgPool3d").Device(DEVICE_GPU), AvgPool3dGpuOp);


void avgPool3dGradLauncher(int Np, int Mp, int Nout, int C, const int* nnCount,
                           const int* nnIndex, const float* gradOutput, float* gradInput);
class AvgPool3dGradGpuOp : public OpKernel {
    public:
        explicit AvgPool3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor       = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& nn_count_tensor    = context->input(2);
            const Tensor& nn_index_tensor    = context->input(3);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);       // number of input points
            int C    = input_tensor.shape().dim_size(1);       // number of input channels
            int Mp   = grad_output_tensor.shape().dim_size(1); // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0);    // number of neighbor pairs

            OP_REQUIRES(context, nn_count_tensor.dims()==1,
                        errors::InvalidArgument("Rank of nn_count should be 1."));
            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("Shape of nn_index should be (Nout,2)."));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Np,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* nnCount   = &(nn_count_flat(0));
            const int* nnIndex   = &(nn_index_flat(0));

            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*Np*C);
            avgPool3dGradLauncher(Np, Mp, Nout, C, nnCount, nnIndex, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("AvgPool3dGrad").Device(DEVICE_GPU), AvgPool3dGradGpuOp);