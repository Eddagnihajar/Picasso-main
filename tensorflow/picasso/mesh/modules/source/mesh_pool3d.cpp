#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("MeshMaxPool3d")
    .Input("input: float32")      // batch_npoints * in_channels
    .Input("vt_replace: int32")   // batch_npoints
    .Input("vt_map: int32")       // batch_npoints
    .Input("vt_out: float32")     // batch_mpoints * D
    .Output("output: float32")    // batch_mpoints * in_channels
    .Output("max_index: int32")   // batch_mpoints * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input, vt_out;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &vt_out));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(vt_out,0),
                                                                          c->Dim(input,1)});
        c->set_output(0, output);
        c->set_output(1, output);
        return Status::OK();
    });
REGISTER_OP("MeshMaxPool3dGrad")
    .Input("input: float32")       // batch_npoints * in_channels
    .Input("grad_output: float32") // batch_mpoints * in_channels
    .Input("max_index: int32")     // batch_mpoints * in_channels
    .Output("grad_input: float32") // batch_npoints * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
REGISTER_OP("MeshAvgPool3d")
    .Input("input: float32")      // batch_npoints * in_channels
    .Input("vt_replace: int32")   // batch_npoints
    .Input("vt_map: int32")       // batch_npoints
    .Input("vt_out: float32")     // batch_mpoints * D
    .Output("output: float32")    // batch_mpoints * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input, vt_out;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &vt_out));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(vt_out,0),
                                                                          c->Dim(input,1)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("MeshAvgPool3dGrad")
    .Input("input: float32")       // batch_npoints * in_channels
    .Input("grad_output: float32") // batch_mpoints * in_channels
    .Input("vt_replace: int32")    // batch_npoints
    .Input("vt_map: int32")        // batch_npoints
    .Output("grad_input: float32") // batch_npoints * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void maxPool3dLauncher(int nvIn, int C, const int* vtReplace, const int* vtMap,
                       const float* input, float* output, int* maxIndex);
class MaxPool3dGpuOp : public OpKernel {
    public:
        explicit MaxPool3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor     = context->input(0);
            const Tensor& vtReplace_tensor = context->input(1);
            const Tensor& vtMap_tensor     = context->input(2);
            const Tensor& vtOut_tensor     = context->input(3);

            // get the dims required by computations
            int nvIn  = input_tensor.shape().dim_size(0);   // number of input points
            int C     = input_tensor.shape().dim_size(1);   // number of input channels
            int nvOut = vtOut_tensor.shape().dim_size(0);   // number of output points

            OP_REQUIRES(context, input_tensor.dims()==2, errors::InvalidArgument(
            "rank of input should be 2, i.e. (batch_npoints,channels)"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtReplace_tensor.shape()),
                errors::InvalidArgument("vtReplace expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat     = input_tensor.flat<float>();
            auto vtReplace_flat = vtReplace_tensor.flat<int32>();
            auto vtMap_flat     = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor    = NULL;
            Tensor* max_index_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvOut,C}, &output_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{nvOut,C}, &max_index_tensor));
            auto output_flat    = output_tensor->flat<float>();
            auto max_index_flat = max_index_tensor->flat<int32>();

            const float* input   = &(input_flat(0));
            const int* vtReplace = &(vtReplace_flat(0));
            const int* vtMap     = &(vtMap_flat(0));

            float* output = &(output_flat(0));
            int* maxIndex = &(max_index_flat(0));
            cudaMemset(output, 0, sizeof(float)*nvOut*C);
            cudaMemset(maxIndex, 0, sizeof(int)*nvOut*C);
            maxPool3dLauncher(nvIn, C, vtReplace, vtMap, input, output, maxIndex);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshMaxPool3d").Device(DEVICE_GPU), MaxPool3dGpuOp);


void maxPool3dGradLauncher(int nvOut, int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput);
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
            int nvIn  = input_tensor.shape().dim_size(0);         // number of input points
            int C     = input_tensor.shape().dim_size(1);         // number of input channels
            int nvOut = grad_output_tensor.shape().dim_size(0);   // number of output points

            OP_REQUIRES(context, max_index_tensor.dims()==2, errors::InvalidArgument("rank of max_index should be 2, i.e. (batch_mpoints, in_channels)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto max_index_flat   = max_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvIn,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* maxIndex  = &(max_index_flat(0));

            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*nvIn*C);
            maxPool3dGradLauncher(nvOut, C, maxIndex, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshMaxPool3dGrad").Device(DEVICE_GPU), MaxPool3dGradGpuOp);


void avgPool3dLauncher(int Nv, int C, const int* vtReplace, const int* vtMap,
                       const float* input, float* output);
class AvgPool3dGpuOp : public OpKernel {
    public:
        explicit AvgPool3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor     = context->input(0);
            const Tensor& vtReplace_tensor = context->input(1);
            const Tensor& vtMap_tensor     = context->input(2);
            const Tensor& vtOut_tensor     = context->input(3);

            // get the dims required by computations
            int nvIn = input_tensor.shape().dim_size(0);  // number of input points
            int C    = input_tensor.shape().dim_size(1);  // number of input channels
             int nvOut = vtOut_tensor.shape().dim_size(0);   // number of output points

            OP_REQUIRES(context, input_tensor.dims()==2, errors::InvalidArgument(
            "rank of input should be 2, i.e. (batch_npoints,channels)"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtReplace_tensor.shape()),
                errors::InvalidArgument("vtReplace expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat     = input_tensor.flat<float>();
            auto vtReplace_flat = vtReplace_tensor.flat<int32>();
            auto vtMap_flat     = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvOut,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* input   = &(input_flat(0));
            const int* vtReplace = &(vtReplace_flat(0));
            const int* vtMap     = &(vtMap_flat(0));

            float* output = &(output_flat(0));
            cudaMemset(output, 0, sizeof(float)*nvOut*C);
            avgPool3dLauncher(nvIn, C, vtReplace, vtMap, input, output);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshAvgPool3d").Device(DEVICE_GPU), AvgPool3dGpuOp);


void avgPool3dGradLauncher(int nvIn, int C, const int* vtReplace, const int* vtMap,
                           const float* gradOutput, float* gradInput);
class AvgPool3dGradGpuOp : public OpKernel {
    public:
        explicit AvgPool3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor       = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& vtReplace_tensor   = context->input(2);
            const Tensor& vtMap_tensor       = context->input(3);

             // get the dims required by computations
            int nvIn = input_tensor.shape().dim_size(0);   // number of input points
            int C    = input_tensor.shape().dim_size(1);   // number of input channels

            OP_REQUIRES(context, input_tensor.dims()==2, errors::InvalidArgument(
            "rank of input should be 2, i.e. (batch_npoints,channels)"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtReplace_tensor.shape()),
                errors::InvalidArgument("vtReplace expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto vtReplace_flat   = vtReplace_tensor.flat<int32>();
            auto vtMap_flat       = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvIn,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* vtReplace = &(vtReplace_flat(0));
            const int* vtMap     = &(vtMap_flat(0));

            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*nvIn*C);
            avgPool3dGradLauncher(nvIn, C, vtReplace, vtMap, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshAvgPool3dGrad").Device(DEVICE_GPU), AvgPool3dGradGpuOp);


