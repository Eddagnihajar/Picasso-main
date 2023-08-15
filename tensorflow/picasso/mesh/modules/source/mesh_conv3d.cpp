#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("Facet2vertexConv3d")
    .Input("input: float32")   // input face features: concat_NfIn * in_channels
    .Input("filter: float32")  // convolution: filter_size * in_channels * channel_multiplier
    .Input("coeff: float32")   // face coefficients: concat_NfIn * filter_size
    .Input("face: int32")      // face vertex list: concat_NfIn * 3
    .Input("nf_count: int32")  // number of adjacent faces for each output vertex: concat_NvOut
    .Input("vt_map: int32")    // vertex mapping from input to output vertices: concat_NvIn
    .Output("output: float32") // output vertex features: concat_NvOut * out_channels  (out_channels = in_channels * channel_multiplier)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle filter;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter));
        ::tensorflow::shape_inference::ShapeHandle nf_count;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &nf_count));
        ::tensorflow::shape_inference::DimensionHandle Cout;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(filter, 1), c->Dim(filter, 2), &Cout));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nf_count, 0), Cout});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("Facet2vertexConv3dGrad")
    .Input("input: float32")        // input face features: concat_NfIn * in_channels
    .Input("filter: float32")       // convolution: filter_size * in_channels * channel_multiplier
    .Input("coeff: float32")        // face coefficients: concat_NfIn * filter_size
    .Input("grad_output: float32")  // gradient of output vertex features: concat_NvOut * out_channels
    .Input("face: int32")           // face vertex list: concat_NfIn * 3
    .Input("nf_count: int32")       // number of adjacent faces for each output vertex: concat_NvOut
    .Input("vt_map: int32")         // vertex mapping from input to output vertices: concat_NvIn
    .Output("grad_input: float32")  // gradient of input face features: concat_NfIn * in_channels
    .Output("grad_filter: float32") // gradient of convolution filter: filter_size * in_channels * channel_multiplier
    .Output("grad_coeff: float32")  // gradient of face coefficients: concat_NfIn * filter_size
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        c->set_output(2, c->input(2));
        return Status::OK();
    });
REGISTER_OP("Vertex2facetConv3d")
    .Input("input: float32")   // input vertex features: concat_NvIn * in_channels
    .Input("filter: float32")  // convolution: 3 * in_channels * channel_multiplier
    .Input("face: int32")      // face vertex list: concat_NfIn * 3
    .Output("output: float32") // output face features: concat_NfIn * out_channels  (out_channels = in_channels * channel_multiplier)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle filter;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter));
        ::tensorflow::shape_inference::ShapeHandle face;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &face));
        ::tensorflow::shape_inference::DimensionHandle Cout;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(filter, 1), c->Dim(filter, 2), &Cout));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(face, 0), Cout});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("Vertex2facetConv3dGrad")
    .Input("input: float32")       // input vertex features: concat_NvIn * in_channels
    .Input("filter: float32")      // convolution: 3 * in_channels * channel_multiplier
    .Input("grad_output: float32") // gradient of output face features: concat_NfIn * out_channels
    .Input("face: int32")          // face vertex list: concat_NfIn * 3
    .Output("grad_input: float32")  // gradient of input vertex features: concat_NvIn * in_channels
    .Output("grad_filter: float32")  // gradient of convolution filters: 3 * in_channels * channel_multiplier
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
    });
REGISTER_OP("Facet2facetConv3d")
    .Input("input: float32")        // input face features: [concat_ptNiK, in_channels]
    .Input("filter: float32")       // convolution: [out_channels, in_channels, 3]
    .Input("bary_coeff: float32")   // face Barycentric interpolation weights: [concat_ptNiK, 3]
    .Input("num_interior: int32")   // number of interior interpolated: concat_NfIn
    .Output("output: float32")      // output face features: [concat_NfIn, out_channels]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        ::tensorflow::shape_inference::ShapeHandle filter;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter));
        ::tensorflow::shape_inference::ShapeHandle bary_coeff;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &bary_coeff));
        ::tensorflow::shape_inference::ShapeHandle num_interior;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &num_interior));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(num_interior, 0), c->Dim(filter, 0)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("Facet2facetConv3dGrad")
    .Input("input: float32")        // input face features: [concat_ptNiK, in_channels]
    .Input("filter: float32")       // convolution: [out_channels, in_channels, 3]
    .Input("grad_output: float32")  // gradient of output face features: [concat_NfIn, out_channels]
    .Input("bary_coeff: float32")   // face Barycentric interpolation weights: [concat_ptNiK, 3]
    .Input("num_interior: int32")   // number of interior interpolated: [concat_NfIn]
    .Output("grad_input: float32")  // gradient of input face features: [concat_ptNiK, in_channels]
    .Output("grad_filter: float32") // gradient of convolution filter:  [out_channels, 3, in_channels]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
    });


void facet2vertexConv3dLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                                const float* coeff, const float* input, const float* filter, float* output);
class Facet2vertexConv3dGpuOp : public OpKernel {
    public:
        explicit Facet2vertexConv3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& filter_tensor   = context->input(1);
            const Tensor& coeff_tensor    = context->input(2);
            const Tensor& face_tensor     = context->input(3);
            const Tensor& nf_count_tensor = context->input(4);
            const Tensor& vt_map_tensor   = context->input(5);

            // get the dims required by computations
            int NfIn  = input_tensor.shape().dim_size(0);     // number of faces by concatenating batch samples
            int C     = input_tensor.shape().dim_size(1);     // number of input channels
            int K     = filter_tensor.shape().dim_size(0);    // number of template models in the filters
            int r     = filter_tensor.shape().dim_size(2);    // depthwise channel multiplier
            int NvOut = nf_count_tensor.shape().dim_size(0);  // number of output vertices/points
            int NvIn  = vt_map_tensor.shape().dim_size(0);    // number of input  vertices/points

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C, errors::InvalidArgument("Input Channel size error of the filter"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(nf_count_tensor.shape()),
                errors::InvalidArgument("nfCount expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vt_map_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto filter_flat   = filter_tensor.flat<float>();
            auto coeff_flat    = coeff_tensor.flat<float>();
            auto face_flat     = face_tensor.flat<int32>();
            auto nf_count_flat = nf_count_tensor.flat<int32>();
            auto vt_map_flat   = vt_map_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NvOut,C*r}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* input  = &(input_flat(0));
            const float* filter = &(filter_flat(0));
            const float* coeff  = &(coeff_flat(0));
            const int* face     = &(face_flat(0));
            const int* nfCount  = &(nf_count_flat(0));
            const int* vtMap    = &(vt_map_flat(0));

            float* output = &(output_flat(0));
            cudaMemset(output,0,sizeof(float)*NvOut*C*r);
            facet2vertexConv3dLauncher(NfIn, C, r, K, vtMap, nfCount, face, coeff, input, filter, output);
        }
};
REGISTER_KERNEL_BUILDER(Name("Facet2vertexConv3d").Device(DEVICE_GPU), Facet2vertexConv3dGpuOp);


void facet2vertexConv3dGradLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                            const float* coeff, const float* input, const float* filter, const float* gradOutput,
                            float* gradInput, float* gradFilter, float* gradCoeff);
class Facet2vertexConv3dGradGpuOp : public OpKernel {
    public:
        explicit Facet2vertexConv3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
             // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& filter_tensor   = context->input(1);
            const Tensor& coeff_tensor    = context->input(2);
            const Tensor& gradOut_tensor  = context->input(3);
            const Tensor& face_tensor     = context->input(4);
            const Tensor& nf_count_tensor = context->input(5);
            const Tensor& vt_map_tensor   = context->input(6);

            // get the dims required by computations
            int NfIn  = input_tensor.shape().dim_size(0);     // number of faces by concatenating batch samples
            int C     = input_tensor.shape().dim_size(1);     // number of input channels
            int K     = filter_tensor.shape().dim_size(0);    // number of template models in the filters
            int r     = filter_tensor.shape().dim_size(2);    // depthwise channel multiplier
            int NvOut = nf_count_tensor.shape().dim_size(0);  // number of output vertices/points
            int NvIn  = vt_map_tensor.shape().dim_size(0);    // number of input  vertices/points

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C, errors::InvalidArgument("Input Channel size error of the filter"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(nf_count_tensor.shape()),
                errors::InvalidArgument("nfCount expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vt_map_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto filter_flat   = filter_tensor.flat<float>();
            auto coeff_flat    = coeff_tensor.flat<float>();
            auto gradOut_flat  = gradOut_tensor.flat<float>();
            auto face_flat     = face_tensor.flat<int32>();
            auto nf_count_flat = nf_count_tensor.flat<int32>();
            auto vt_map_flat   = vt_map_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor  = NULL;
            Tensor* grad_filter_tensor = NULL;
            Tensor* grad_coeff_tensor  = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NfIn,C}, &grad_input_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{K,C,r},  &grad_filter_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{NfIn,K}, &grad_coeff_tensor));
            auto grad_input_flat  = grad_input_tensor->flat<float>();
            auto grad_filter_flat = grad_filter_tensor->flat<float>();
            auto grad_coeff_flat  = grad_coeff_tensor->flat<float>();

            const float* input   = &(input_flat(0));
            const float* filter  = &(filter_flat(0));
            const float* coeff   = &(coeff_flat(0));
            const float* gradOut = &(gradOut_flat(0));
            const int* face      = &(face_flat(0));
            const int* nfCount   = &(nf_count_flat(0));
            const int* vtMap     = &(vt_map_flat(0));

            float* gradInput  = &(grad_input_flat(0));
            float* gradFilter = &(grad_filter_flat(0));
            float* gradCoeff  = &(grad_coeff_flat(0));
            cudaMemset(gradInput,0,sizeof(float)*NfIn*C);
            cudaMemset(gradFilter,0,sizeof(float)*K*C*r);
            cudaMemset(gradCoeff,0,sizeof(float)*NfIn*K);
            facet2vertexConv3dGradLauncher(NfIn, C, r, K, vtMap, nfCount, face, coeff, input, filter,
                                   gradOut, gradInput, gradFilter, gradCoeff);
        }
};
REGISTER_KERNEL_BUILDER(Name("Facet2vertexConv3dGrad").Device(DEVICE_GPU), Facet2vertexConv3dGradGpuOp);


void vertex2facetConv3dLauncher(int NfIn, int C, int r, const int* face,
                                const float* input, const float* filter, float* output);
class Vertex2facetConv3dGpuOp : public OpKernel {
    public:
        explicit Vertex2facetConv3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& filter_tensor   = context->input(1);
            const Tensor& face_tensor     = context->input(2);

            // get the dims required by computations
            int NvIn = input_tensor.shape().dim_size(0);     // number of vertices by concatenating batch samples
            int C    = input_tensor.shape().dim_size(1);     // number of input channels
            int r    = filter_tensor.shape().dim_size(2);    // depthwise channel multiplier
            int NfIn = face_tensor.shape().dim_size(0);      // number of facets

            OP_REQUIRES(context, filter_tensor.shape().dim_size(0)==3 && filter_tensor.shape().dim_size(1)==C,
                        errors::InvalidArgument("3 filters required, or input Channel size error of the filter"));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto filter_flat   = filter_tensor.flat<float>();
            auto face_flat     = face_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NfIn,C*r}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* input  = &(input_flat(0));
            const float* filter = &(filter_flat(0));
            const int* face     = &(face_flat(0));

            float* output = &(output_flat(0));
            cudaMemset(output,0,sizeof(float)*NfIn*C*r);
            vertex2facetConv3dLauncher(NfIn, C, r, face, input, filter, output);
        }
};
REGISTER_KERNEL_BUILDER(Name("Vertex2facetConv3d").Device(DEVICE_GPU), Vertex2facetConv3dGpuOp);


void vertex2facetConv3dGradLauncher(int NfIn, int C, int r, const int* face,
                                    const float* input, const float* filter, const float* gradOutput,
                                    float* gradInput, float* gradFilter);
class Vertex2facetConv3dGradGpuOp : public OpKernel {
    public:
        explicit Vertex2facetConv3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& filter_tensor   = context->input(1);
            const Tensor& gradOut_tensor  = context->input(2);
            const Tensor& face_tensor     = context->input(3);

            // get the dims required by computations
            int NvIn = input_tensor.shape().dim_size(0);     // number of vertices by concatenating batch samples
            int C    = input_tensor.shape().dim_size(1);     // number of input channels
            int r    = filter_tensor.shape().dim_size(2);    // depthwise channel multiplier
            int NfIn = face_tensor.shape().dim_size(0);      // number of facets

            OP_REQUIRES(context, filter_tensor.shape().dim_size(0)==3 && filter_tensor.shape().dim_size(1)==C,
                        errors::InvalidArgument("3 filters required, or input Channel size error of the filter"));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto filter_flat   = filter_tensor.flat<float>();
            auto gradOut_flat  = gradOut_tensor.flat<float>();
            auto face_flat     = face_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor  = NULL;
            Tensor* grad_filter_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NvIn,C}, &grad_input_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{3,C,r},  &grad_filter_tensor));
            auto grad_input_flat  = grad_input_tensor->flat<float>();
            auto grad_filter_flat = grad_filter_tensor->flat<float>();

            const float* input   = &(input_flat(0));
            const float* filter  = &(filter_flat(0));
            const float* gradOut = &(gradOut_flat(0));
            const int*   face    = &(face_flat(0));

            float* gradInput  = &(grad_input_flat(0));
            float* gradFilter = &(grad_filter_flat(0));
            cudaMemset(gradInput,0,sizeof(float)*NvIn*C);
            cudaMemset(gradFilter,0,sizeof(float)*3*C*r);
            vertex2facetConv3dGradLauncher(NfIn, C, r, face, input, filter,
                                           gradOut, gradInput, gradFilter);
        }
};
REGISTER_KERNEL_BUILDER(Name("Vertex2facetConv3dGrad").Device(DEVICE_GPU), Vertex2facetConv3dGradGpuOp);


void facet2facetConv3dLauncher(int NfIn, int Cin, int Cout, const int* numInterior, const float* intrplWgts,
                               const float* input, const float* filter, float* output);
class Facet2facetConv3dGpuOp : public OpKernel {
    public:
        explicit Facet2facetConv3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& filter_tensor   = context->input(1);
            const Tensor& baryCoeff_tensor  = context->input(2);
            const Tensor& numInterior_tensor = context->input(3);

            // get the dims required by computations
            int NiK  = input_tensor.shape().dim_size(0);      // total number of interiors interpolated in NfIn faces
            int Cin  = input_tensor.shape().dim_size(1);      // number of input channels
            int Cout = filter_tensor.shape().dim_size(0);     // number of output channels
            int NfIn = numInterior_tensor.shape().dim_size(0);// number of faces by concatenating batch samples

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==Cin && filter_tensor.shape().dim_size(2)==3,
                        errors::InvalidArgument("The filter size should be (Cout, Cin, 3)"));
            OP_REQUIRES(context, baryCoeff_tensor.dims()==2 && baryCoeff_tensor.shape().dim_size(0)==NiK
                        && baryCoeff_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("The shape of input vertex should be (ptNiK, 3))."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(numInterior_tensor.shape()),
                        errors::InvalidArgument("numInterior expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto filter_flat   = filter_tensor.flat<float>();
            auto baryCoeff_flat  = baryCoeff_tensor.flat<float>();
            auto numInterior_flat = numInterior_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NfIn,Cout}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* input  = &(input_flat(0));
            const float* filter = &(filter_flat(0));
            const float* baryCoeff = &(baryCoeff_flat(0));
            const int* numInterior = &(numInterior_flat(0));

            float* output = &(output_flat(0));
            cudaMemset(output,0,sizeof(float)*NfIn*Cout);
            facet2facetConv3dLauncher(NfIn, Cin, Cout, numInterior, baryCoeff, input, filter, output);
        }
};
REGISTER_KERNEL_BUILDER(Name("Facet2facetConv3d").Device(DEVICE_GPU), Facet2facetConv3dGpuOp);


void facet2facetConv3dGradLauncher(int NfIn, int Cin, int Cout, const int* numInterior,
                                   const float* baryCoeff, const float* input, const float* filter,
                                   const float* gradOutput, float* gradInput, float* gradFilter);
class Facet2facetConv3dGradGpuOp : public OpKernel {
    public:
        explicit Facet2facetConv3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor    = context->input(0);
            const Tensor& filter_tensor   = context->input(1);
            const Tensor& gradOut_tensor  = context->input(2);
            const Tensor& baryCoeff_tensor  = context->input(3);
            const Tensor& numInterior_tensor = context->input(4);

            // get the dims required by computations
            int NiK  = input_tensor.shape().dim_size(0);      // total number of interiors interpolated in NfIn faces
            int Cin  = input_tensor.shape().dim_size(1);      // number of input channels
            int Cout = filter_tensor.shape().dim_size(0);     // number of output channels
            int NfIn = numInterior_tensor.shape().dim_size(0);// number of faces by concatenating batch samples

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==Cin && filter_tensor.shape().dim_size(2)==3,
                        errors::InvalidArgument("The filter size should be (Cout, Cin, 3)."));
            OP_REQUIRES(context, baryCoeff_tensor.dims()==2 && baryCoeff_tensor.shape().dim_size(0)==NiK
                        && baryCoeff_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("The shape of input vertex should be (ptNiK, 3))."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(numInterior_tensor.shape()),
                errors::InvalidArgument("numInterior expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat    = input_tensor.flat<float>();
            auto filter_flat   = filter_tensor.flat<float>();
            auto gradOut_flat  = gradOut_tensor.flat<float>();
            auto baryCoeff_flat  = baryCoeff_tensor.flat<float>();
            auto numInterior_flat = numInterior_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor  = NULL;
            Tensor* grad_filter_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NiK,Cin}, &grad_input_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Cout,Cin,3}, &grad_filter_tensor));
            auto grad_input_flat  = grad_input_tensor->flat<float>();
            auto grad_filter_flat = grad_filter_tensor->flat<float>();

            const float* input  = &(input_flat(0));
            const float* filter = &(filter_flat(0));
            const float* gradOut = &(gradOut_flat(0));
            const float* baryCoeff = &(baryCoeff_flat(0));
            const int* numInterior  = &(numInterior_flat(0));

            float* gradInput  = &(grad_input_flat(0));
            float* gradFilter = &(grad_filter_flat(0));
            cudaMemset(gradInput,0,sizeof(float)*NiK*Cin);
            cudaMemset(gradFilter,0,sizeof(float)*Cout*Cin*3);
            facet2facetConv3dGradLauncher(NfIn, Cin, Cout, numInterior, baryCoeff, input, filter,
                                          gradOut, gradInput, gradFilter);
        }
};
REGISTER_KERNEL_BUILDER(Name("Facet2facetConv3dGrad").Device(DEVICE_GPU), Facet2facetConv3dGradGpuOp);
