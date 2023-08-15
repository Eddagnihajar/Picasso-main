#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("DepthwiseConv3d")
    .Input("input: float32")   // input features: concat_Np * in_channels
    .Input("filter: float32")  // convolution filter parameters: filter_size * in_channels * channel_multiplier
    .Input("nn_count: int32")  // number of neighbors: concat_Mp
    .Input("nn_index: int32")  // neighbor indices, each pair (outIdx, inIdx): Nout * 2
    .Input("bin_index: int32") // kernel bin indices: Nout
    .Output("output: float32") // output features: concat_Mp * out_channels
                               // (out_channels = in_channels * channel_multiplier)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle filter;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter));
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &nn_count));
        ::tensorflow::shape_inference::DimensionHandle Cout;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(filter, 1), c->Dim(filter, 2), &Cout));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nn_count, 0), Cout});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("DepthwiseConv3dGrad")
    .Input("input: float32")        // input features: concat_Np * in_channels
    .Input("filter: float32")       // convolution filter parameters: filter_size * in_channels * channel_multiplier
    .Input("grad_output:float32")   // gradient of output features: concat_Mp * out_channels
    .Input("nn_count: int32")       // number of neighbors: concat_Mp
    .Input("nn_index: int32")       // neighbor indices: Nout * 2
    .Input("bin_index: int32")      // kernel bin indices: Nout
    .Output("grad_input: float32")  // gradient of input features: concat_Np * in_channels
    .Output("grad_filter: float32") // gradient of filter parameters: filter_size * in_channels * channel_multiplier
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
    });
REGISTER_OP("FuzzyDepthwiseConv3d")
    .Input("input: float32")        // input features: concat_Np * in_channels
    .Input("filter: float32")       // convolution filter parameters: filter_size * in_channels * channel_multiplier
    .Input("nn_count: int32")       // number of neighbors: concat_Mp
    .Input("nn_index: int32")       // neighbor indices: Nout * 2
    .Input("bin_index: int32")      // kernel bin indices: Nout * 3
    .Input("bin_coeff: float32")    // kernel bin coefficients: Nout * 3
    .Output("output: float32")      // output features: concat_Mp * out_channels
                                    // (out_channels = in_channels * channel_multiplier)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle filter;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter));
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &nn_count));
        ::tensorflow::shape_inference::DimensionHandle Cout;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(filter, 1), c->Dim(filter, 2), &Cout));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nn_count, 0), Cout});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("FuzzyDepthwiseConv3dGrad")
    .Input("input: float32")        // input features: concat_Np * in_channels
    .Input("filter: float32")       // convolution filter parameters: filter_size * in_channels * channel_multiplier
    .Input("grad_output:float32")   // gradient of output features: concat_Mp * out_channels
    .Input("nn_count: int32")       // number of neighbors: concat_Mp
    .Input("nn_index: int32")       // neighbor indices: Nout * 2
    .Input("bin_index: int32")      // kernel bin indices: Nout * 3
    .Input("bin_coeff: float32")    // kernel bin coefficients: Nout * 3
    .Output("grad_input: float32")  // gradient of input features: concat_Np * in_channels
    .Output("grad_filter: float32") // gradient of filter parameters: filter_size * in_channels * channel_multiplier
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
    });


void depthwiseConv3dLauncher(int Np, int Mp, int Nout, int F, int C, int r,
                             const int* nnCount, const int* nnIndex, const int* binIndex,
                             const float* input, const float* filter, float* output);
class DepthwiseConv3dGpuOp : public OpKernel {
    public:
        explicit DepthwiseConv3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& filter_tensor = context->input(1);
            const Tensor& nn_count_tensor = context->input(2);
            const Tensor& nn_index_tensor = context->input(3);
            const Tensor& bin_index_tensor = context->input(4);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);    // number of input points
            int C    = input_tensor.shape().dim_size(1);    // number of input channels
            int F    = filter_tensor.shape().dim_size(0);   // filter bin size
            int r    = filter_tensor.shape().dim_size(2);   // depthwise channel multiplier
            int Mp   = nn_count_tensor.shape().dim_size(0); // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0); // number of neighbor pairs

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C,
                        errors::InvalidArgument("Input Channel size error of the filter"));
            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("The rank of nn_index should be 2."));
            OP_REQUIRES(context, bin_index_tensor.dims()==1,
                        errors::InvalidArgument("The rank of bin_index should be 1."));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto bin_index_flat = bin_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp,C*r}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* filt = &(filter_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* binIndex = &(bin_index_flat(0));

            float* out = &(output_flat(0));
            cudaMemset(out,0,sizeof(float)*Mp*C*r);
            depthwiseConv3dLauncher(Np, Mp, Nout, F, C, r, nnCount, nnIndex, binIndex, in, filt, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv3d").Device(DEVICE_GPU), DepthwiseConv3dGpuOp);


void depthwiseConv3dGradLauncher(int Np, int Mp, int Nout, int F, int C, int r,
                                 const int* nnCount, const int* nnIndex, const int* binIndex,
                                 const float* input, const float* filter, const float* gradOutput,
                                 float* gradInput, float* gradFilter);
class DepthwiseConv3dGradGpuOp : public OpKernel {
    public:
        explicit DepthwiseConv3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& filter_tensor = context->input(1);
            const Tensor& grad_output_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);
            const Tensor& nn_index_tensor = context->input(4);
            const Tensor& bin_index_tensor = context->input(5);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);      // number of input points
            int C    = input_tensor.shape().dim_size(1);      // number of input channels
            int F    = filter_tensor.shape().dim_size(0);     // filter bin size
            int r    = filter_tensor.shape().dim_size(2);     // depthwise channel multiplier
            int Mp   = nn_count_tensor.shape().dim_size(0);   // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0);   // number of neighbor pairs

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C,
                        errors::InvalidArgument("Channel size error of the filter."));
            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("The rank of nn_index should be 2."));
            OP_REQUIRES(context, bin_index_tensor.dims()==1,
                        errors::InvalidArgument("The rank of bin_index should be 1."));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto bin_index_flat = bin_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            Tensor* grad_filter_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Np,C}, &grad_input_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{F,C,r}, &grad_filter_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();
            auto grad_filter_flat = grad_filter_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* filt = &(filter_flat(0));
            const float* gradOut = &(grad_output_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* binIndex = &(bin_index_flat(0));

            float* gradIn = &(grad_input_flat(0));
            float* gradFilt = &(grad_filter_flat(0));
            cudaMemset(gradIn,0,sizeof(float)*Np*C);
            cudaMemset(gradFilt,0,sizeof(float)*F*C*r);
            depthwiseConv3dGradLauncher(Np, Mp, Nout, F, C, r, nnCount, nnIndex, binIndex,
                                        in, filt, gradOut, gradIn, gradFilt);
        }
};
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv3dGrad").Device(DEVICE_GPU), DepthwiseConv3dGradGpuOp);


void fuzzyDepthwiseConv3dLauncher(int Np, int Mp, int Nout, int F, int C, int r, int T, const int* nnCount,
                                  const int* nnIndex, const int* binIndex, const float* binCoeff,
                                  const float* input, const float* filter, float* output);
class FuzzyDepthwiseConv3dGpuOp : public OpKernel {
    public:
        explicit FuzzyDepthwiseConv3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& filter_tensor = context->input(1);
            const Tensor& nn_count_tensor = context->input(2);
            const Tensor& nn_index_tensor = context->input(3);
            const Tensor& bin_index_tensor = context->input(4);
            const Tensor& bin_coeff_tensor = context->input(5);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);    // number of input points
            int C   = input_tensor.shape().dim_size(1);     // number of input channels
            int F    = filter_tensor.shape().dim_size(0);   // filter bin size
            int r    = filter_tensor.shape().dim_size(2);   // depthwise channel multiplier
            int Mp   = nn_count_tensor.shape().dim_size(0); // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0); // number of neighbor pairs
            int T    = bin_index_tensor.shape().dim_size(1);// maximum number of clusters/bins covered

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C,
                        errors::InvalidArgument("Input Channel size error of the filter"));
            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("The rank of nn_index should be 2."));
            OP_REQUIRES(context, bin_index_tensor.dims()==2,
                        errors::InvalidArgument("The rank of bin_index should be 2."));
            OP_REQUIRES(context, bin_coeff_tensor.dims()==2,
                        errors::InvalidArgument("The rank of bin_coeff should be 2."));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto bin_index_flat = bin_index_tensor.flat<int32>();
            auto bin_coeff_flat = bin_coeff_tensor.flat<float>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp,C*r}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* filt = &(filter_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* binIndex = &(bin_index_flat(0));
            const float* binCoeff = &(bin_coeff_flat(0));

            float* out = &(output_flat(0));
            cudaMemset(out,0,sizeof(float)*Mp*C*r);
            fuzzyDepthwiseConv3dLauncher(Np, Mp, Nout, F, C, r, T, nnCount, nnIndex, binIndex, binCoeff, in, filt, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("FuzzyDepthwiseConv3d").Device(DEVICE_GPU), FuzzyDepthwiseConv3dGpuOp);


void fuzzyDepthwiseConv3dGradLauncher(int Np, int Mp, int Nout, int F, int C, int r, int T,
                                      const int* nnCount, const int* nnIndex, const int* binIndex,
                                      const float* binCoeff, const float* input, const float* filter,
                                      const float* gradOutput, float* gradInput, float* gradFilter);
class FuzzyDepthwiseConv3dGradGpuOp : public OpKernel {
    public:
        explicit FuzzyDepthwiseConv3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& filter_tensor = context->input(1);
            const Tensor& grad_output_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);
            const Tensor& nn_index_tensor = context->input(4);
            const Tensor& bin_index_tensor = context->input(5);
            const Tensor& bin_coeff_tensor = context->input(6);

            // get the dims required by computations
            int Np   = input_tensor.shape().dim_size(0);      // number of input points
            int C    = input_tensor.shape().dim_size(1);      // number of input channels
            int F    = filter_tensor.shape().dim_size(0);     // filter bin size
            int r    = filter_tensor.shape().dim_size(2);     // depthwise channel multiplier
            int Mp   = nn_count_tensor.shape().dim_size(0);   // number of output points
            int Nout = nn_index_tensor.shape().dim_size(0);   // number of neighbor pairs
             int T   = bin_index_tensor.shape().dim_size(1);  // maximum number of clusters/bins covered

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C,
                        errors::InvalidArgument("Channel size error of the filter"));
            OP_REQUIRES(context, nn_index_tensor.dims()==2,
                        errors::InvalidArgument("The rank of nn_index should be 2."));
            OP_REQUIRES(context, bin_index_tensor.dims()==2,
                        errors::InvalidArgument("The rank of bin_index should be 2."));
            OP_REQUIRES(context, bin_coeff_tensor.dims()==2,
                        errors::InvalidArgument("The rank of bin_coeff should be 2."));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto bin_index_flat = bin_index_tensor.flat<int32>();
            auto bin_coeff_flat = bin_coeff_tensor.flat<float>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            Tensor* grad_filter_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Np,C}, &grad_input_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{F,C,r}, &grad_filter_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();
            auto grad_filter_flat = grad_filter_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* filt = &(filter_flat(0));
            const float* gradOut = &(grad_output_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* binIndex = &(bin_index_flat(0));
            const float* binCoeff = &(bin_coeff_flat(0));

            float* gradIn = &(grad_input_flat(0));
            float* gradFilt = &(grad_filter_flat(0));
            cudaMemset(gradIn,0,sizeof(float)*Np*C);
            cudaMemset(gradFilt,0,sizeof(float)*F*C*r);
            fuzzyDepthwiseConv3dGradLauncher(Np, Mp, Nout, F, C, r, T, nnCount, nnIndex, binIndex,
                                             binCoeff, in, filt, gradOut, gradIn, gradFilt);
        }
};
REGISTER_KERNEL_BUILDER(Name("FuzzyDepthwiseConv3dGrad").Device(DEVICE_GPU), FuzzyDepthwiseConv3dGradGpuOp);



