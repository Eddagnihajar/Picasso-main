#include <cmath> // sqrtf
#include <math.h>  // floor, ceil, round
#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;


REGISTER_OP("SphericalKernel")
    .Attr("radius: float")         // range search radius
    .Attr("n_azim: int")           // division along azimuth direction
    .Attr("p_elev: int")           // division along elevation direction
    .Attr("q_radi: int")           // division along radius direction
    .Input("database: float32")    // database points: concat_Np * 3 (x,y,z)
    .Input("query: float32")       // query points: concat_Mp * 3
    .Input("nn_index: int32")      // neighbor and kernel bin indices: Nout * 2
    .Input("nn_dist: float32")     // distance to the neighbors: Nout
    .Output("filt_index: int32")   // kernel bin indices: Nout
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(3));
        return Status::OK();
    });
REGISTER_OP("FuzzySphericalKernel")
    .Attr("radius: float")          // range search radius
    .Attr("n_azim: int")            // division along azimuth direction
    .Attr("p_elev: int")            // division along elevation direction
    .Attr("q_radi: int")            // division along radius direction
    .Input("database: float32")     // database points: concat_Np * 3 (x,y,z)
    .Input("query: float32")        // query points: concat_Mp * 3
    .Input("nn_index: int32")       // neighbor and kernel bin indices: (Nout, 2)
    .Input("nn_dist: float32")      // distance to the neighbors: Nout
    .Output("filt_index: int32")    // kernel bin indices: Nout * 3
    .Output("filt_coeff: float32")  // kernel bin coefficients: Nout * 3
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int n_azim, p_elev, q_radi;
        TF_RETURN_IF_ERROR(c->GetAttr("n_azim", &n_azim));
        TF_RETURN_IF_ERROR(c->GetAttr("p_elev", &p_elev));
        TF_RETURN_IF_ERROR(c->GetAttr("q_radi", &q_radi));

        ::tensorflow::shape_inference::ShapeHandle nn_index; // Nout
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &nn_index));

        ::tensorflow::shape_inference::ShapeHandle filt_coeff = c->MakeShape({c->Dim(nn_index, 0), 3});
        c->set_output(0, filt_coeff);
        c->set_output(1, filt_coeff);
        return Status::OK();
    });


void sphericalKernelLauncher(int Nout, int n, int p, int q, float radius, const float* database,
                             const float* query, const int* nnIndex, const float* nnDist, int* filtIndex);
class SphericalKernelGpuOp : public OpKernel {
    public:
        explicit SphericalKernelGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("Range search requires radius>0, got ", radius_));

            OP_REQUIRES_OK(context, context->GetAttr("n_azim", &n_));
            OP_REQUIRES(context, n_>2 && n_%2==0, errors::InvalidArgument("Need n_>2 and n_%2==0, got ", n_));

            OP_REQUIRES_OK(context, context->GetAttr("p_elev", &p_));
            OP_REQUIRES(context, p_>0 && p_%2==0, errors::InvalidArgument("Need p_>0 and p_%2==0, got ", p_));

            OP_REQUIRES_OK(context, context->GetAttr("q_radi", &q_));
            OP_REQUIRES(context, q_>0, errors::InvalidArgument("Need q_>0, got ", q_));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& database_tensor = context->input(0);
            const Tensor& query_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_dist_tensor = context->input(3);

            // get the dims required by computations
            int concat_Np = database_tensor.shape().dim_size(0); // number of database points
            int concat_Mp = query_tensor.shape().dim_size(0);    // number of query points
            int Nout      = nn_index_tensor.shape().dim_size(0); // number of neighbor pairs

            OP_REQUIRES(context, database_tensor.dims()==2 && database_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of database points requires to be (Np, 3)"));
            OP_REQUIRES(context, query_tensor.dims()==2 && query_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of query points requires to be (Mp, 3)"));
            OP_REQUIRES(context, nn_index_tensor.dims()==2  && nn_index_tensor.shape().dim_size(1)==2,
                        errors::InvalidArgument("Shape of database points requires to be (Nout, 2)"));

            // flatten the input tensors
            auto database_flat = database_tensor.flat<float>();
            auto query_flat = query_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_dist_flat = nn_dist_tensor.flat<float>();

            // Create an output tensor
            Tensor* filt_index_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Nout}, &filt_index_tensor));
            auto filt_index_flat = filt_index_tensor->flat<int32>();

            const float* database = &(database_flat(0));
            const float* query = &(query_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const float* nnDist = &(nn_dist_flat(0));

            int* filtIndex = &(filt_index_flat(0));
            cudaMemset(filtIndex, 0, sizeof(int)*Nout);

            sphericalKernelLauncher(Nout, n_, p_, q_, radius_, database, query, nnIndex, nnDist, filtIndex);
        }
    private:
        float radius_;
        int n_, p_, q_;
};
REGISTER_KERNEL_BUILDER(Name("SphericalKernel").Device(DEVICE_GPU), SphericalKernelGpuOp);


void fuzzySphericalKernelLauncher(int Nout, int n, int p, int q, float radius, const float* database, const float* query,
                                  const int* nnIndex, const float* nnDist, int* filtIndex, float* filtCoeff);
class FuzzySphericalKernelGpuOp : public OpKernel {
    public:
        explicit FuzzySphericalKernelGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("Range search requires radius>0, got ", radius_));

            OP_REQUIRES_OK(context, context->GetAttr("n_azim", &n_));
            //OP_REQUIRES(context, n_>2 && n_%2==0, errors::InvalidArgument("Need n_>2 and n_%2==0, got ", n_));
            OP_REQUIRES(context, n_>0, errors::InvalidArgument("Need n_>2 and n_%2==0, got ", n_));

            OP_REQUIRES_OK(context, context->GetAttr("p_elev", &p_));
            //OP_REQUIRES(context, p_>0 && p_%2==0, errors::InvalidArgument("Need p_>0 and p_%2==0, got ", p_));
            OP_REQUIRES(context, p_>0, errors::InvalidArgument("Need p_>0 and p_%2==0, got ", p_));

            OP_REQUIRES_OK(context, context->GetAttr("q_radi", &q_));
            OP_REQUIRES(context, q_>0, errors::InvalidArgument("Need q_>0, got ", q_));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& database_tensor = context->input(0);
            const Tensor& query_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_dist_tensor = context->input(3);

            // get the dims required by computations
            int Np   = database_tensor.shape().dim_size(0); // number of database points
            int Mp   = query_tensor.shape().dim_size(0);    // number of query points
            int Nout = nn_index_tensor.shape().dim_size(0); // neighbor pairs

            OP_REQUIRES(context, database_tensor.dims()==2 && database_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of database points requires to be (Np, 3)"));
            OP_REQUIRES(context, query_tensor.dims()==2 && query_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of query points requires to be (Mp, 3)"));
            OP_REQUIRES(context, nn_index_tensor.dims()==2 && nn_index_tensor.shape().dim_size(1)==2,
                        errors::InvalidArgument("Shape of database points requires to be (Nout, 2)"));

            // flatten the input tensors
            auto database_flat = database_tensor.flat<float>();
            auto query_flat = query_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_dist_flat = nn_dist_tensor.flat<float>();

            // Create an output tensor
            Tensor* filt_index_tensor = NULL;
            Tensor* filt_coeff_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Nout,3}, &filt_index_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Nout,3}, &filt_coeff_tensor));
            auto filt_index_flat = filt_index_tensor->flat<int32>();
            auto filt_coeff_flat = filt_coeff_tensor->flat<float>();

            const float* database = &(database_flat(0));
            const float* query = &(query_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const float* nnDist = &(nn_dist_flat(0));

            int* filtIndex = &(filt_index_flat(0));
            float* filtCoeff = &(filt_coeff_flat(0));
            cudaMemset(filtIndex, 0, sizeof(int)*Nout*3);
            cudaMemset(filtCoeff, 0, sizeof(float)*Nout*3);

            fuzzySphericalKernelLauncher(Nout, n_, p_, q_, radius_, database, query,
                                        nnIndex, nnDist, filtIndex, filtCoeff);
        }
    private:
        float radius_;
        int n_, p_, q_;
};
REGISTER_KERNEL_BUILDER(Name("FuzzySphericalKernel").Device(DEVICE_GPU), FuzzySphericalKernelGpuOp);












