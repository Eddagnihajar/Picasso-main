/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("FarthestPointSample")
    .Input("xyz_in: float32")   // concatenated point xyz: concat_Np * 3
    .Input("nv_in: int32")      // number of points in each input batch sample: batch_size
    .Input("nv_out: int32")     // number of points in each output batch sample: batch_size
    .Output("index_out: int32") // concatenated point indices: concat_Mp
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle xyz_in; // concat_Np * 3
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &xyz_in));
        ::tensorflow::shape_inference::ShapeHandle nv_in; // batch_size
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &nv_in));
        ::tensorflow::shape_inference::ShapeHandle nv_out; // batch_size
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &nv_out));

        int mpoint;
        TF_RETURN_IF_ERROR(c->GetAttr("mpoint", &mpoint));
    });


int computeOutputSize(int B, const int* nvOut);
void farthestPointSampleLauncher(int B, int Np, const int* nvIn, const int* nvOut,
                                 const float* xyzIn, float* temp, int* indexOut);
class FarthestPointSampleGpuOp: public OpKernel{
    public:
        explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {}

        void Compute(OpKernelContext * context)override{
            const Tensor& xyz_in_tensor = context->input(0);
            const Tensor& nv_in_tensor  = context->input(1);
            const Tensor& nv_out_tensor = context->input(2);

             // get the dims required by computations
            int Np = xyz_in_tensor.shape().dim_size(0);
            int B  = nv_in_tensor.shape().dim_size(0);

            OP_REQUIRES(context,xyz_in_tensor.dims()==2 && xyz_in_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("FarthestPointSample expects (Np,3) inp shape"));
            OP_REQUIRES(context,nv_in_tensor.dims()==1,
                        errors::InvalidArgument("Rank of nv_in should be 1."));
            OP_REQUIRES(context,nv_out_tensor.dims()==1,
                        errors::InvalidArgument("Rank of nv_out should be 1."));

            auto xyz_in_flat = xyz_in_tensor.flat<float>();
            auto nv_in_flat  = nv_in_tensor.flat<int32>();
            auto nv_out_flat = nv_out_tensor.flat<int32>();

            const float* xyzIn = &(xyz_in_flat(0));
            const int* nvIn    = &(nv_in_flat(0));
            const int* nvOut   = &(nv_out_flat(0));

            int Mp = computeOutputSize(B, nvOut); // extract Mp from nvOut

            Tensor temp_tensor;
            Tensor* index_out_tensor;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{Np},&temp_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{Mp},&index_out_tensor));
            auto temp_flat      = temp_tensor.flat<float>();
            auto index_out_flat = index_out_tensor->flat<int>();

            float* temp   = &(temp_flat(0));
            int* indexOut = &(index_out_flat(0));
            farthestPointSampleLauncher(B, Np, nvIn, nvOut, xyzIn, temp, indexOut);
        }
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);




REGISTER_OP("FarthestPointSample3D")
  .Attr("npoint: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });

void farthestPointSample3DLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
class FarthestPointSample3DGpuOp: public OpKernel{
  public:
    explicit FarthestPointSample3DGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample3D expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,
                  errors::InvalidArgument("FarthestPointSample3D expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestPointSample3DLauncher(b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample3D").Device(DEVICE_GPU),FarthestPointSample3DGpuOp);