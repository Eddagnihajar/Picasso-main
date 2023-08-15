#include <cmath> // sqrtf
#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <iostream>

using namespace tensorflow;
typedef long int LLint;

REGISTER_OP("BuildSphereNeighbor")
    .Attr("radius: float")        // range search radius
    .Attr("nn_sample: int")       // max number of neighbors sampled in the range
    .Input("database: float32")   // database points: concat_Np * 3
    .Input("query: float32")      // query points: concat_Mp * 3
    .Input("nv_database: int32")  // batch: each element is the vertex number of a database sample
    .Input("nv_query: int32")     // batch: each element is the vertex number of a query sample
    .Output("cnt_info: int64")    // number of neighbors: concat_Mp
    .Output("nn_index: int32")    // neighbor indices: Nout * 2
    .Output("nn_dist: float32")   // distance to the neighbors: Nout
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // get and check input shape
        ::tensorflow::shape_inference::ShapeHandle database;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &database));
        ::tensorflow::shape_inference::ShapeHandle query;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &query));
        ::tensorflow::shape_inference::ShapeHandle nv_database;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &nv_database));
        ::tensorflow::shape_inference::ShapeHandle nv_query;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &nv_query));

        // get attribute parameter
        int nn_sample;
        TF_RETURN_IF_ERROR(c->GetAttr("nn_sample", &nn_sample));

        // set output with known shape
        ::tensorflow::shape_inference::ShapeHandle cnt_info = c->MakeShape({c->Dim(query, 0)});
        c->set_output(0, cnt_info);

        return Status::OK();
    });
REGISTER_OP("BuildCubeNeighbor")
    .Attr("length: float")        // cube size: length * length * length
    .Attr("nn_sample: int")       // max number of neighbors sampled in the range
    .Attr("grid_size: int")       // division along azimuth direction
    .Input("database: float32")   // database points: concat_Np * 3
    .Input("query: float32")      // query points: concat_Mp * 3
    .Input("nv_database: int32")  // batch: each element is the vertex number of a database sample
    .Input("nv_query: int32")     // batch: each element is the vertex number of a query sample
    .Output("cnt_info: int64")    // number of neighbors: concat_Mp
    .Output("nn_index: int32")    // neighbor and kernel bin indices: Nout * 2 * 2
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // get attribute parameter
        int nn_sample;
        TF_RETURN_IF_ERROR(c->GetAttr("nn_sample", &nn_sample));

        // get and check input shape
        ::tensorflow::shape_inference::ShapeHandle database;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &database));
        ::tensorflow::shape_inference::ShapeHandle query;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &query));
        ::tensorflow::shape_inference::ShapeHandle nv_database;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &nv_database));
        ::tensorflow::shape_inference::ShapeHandle nv_query;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &nv_query));

        // set output with known shape
        ::tensorflow::shape_inference::ShapeHandle cnt_info = c->MakeShape({c->Dim(query, 0)});
        c->set_output(0, cnt_info);

        return Status::OK();
    });
REGISTER_OP("BuildNearestNeighbor")
    .Input("database: float32")   // database points: concat_Np * 3
    .Input("query: float32")      // query points: concat_Mp * 3
    .Input("nv_database: int32")  // batch: each element is the vertex number of a database sample
    .Input("nv_query: int32")     // batch: each element is the vertex number of a query sample
    .Output("nn_index: int32")    // neighbor indices: concat_Mp * nn_out
    .Output("nn_dist: float32")   // distance to the neighbors: concat_Mp * nn_out
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int nn_out=3; // define the number of nearest neighbors

        // get and check input shape
        ::tensorflow::shape_inference::ShapeHandle database;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &database));
        ::tensorflow::shape_inference::ShapeHandle query;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &query));
        ::tensorflow::shape_inference::ShapeHandle nv_database;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &nv_database));
        ::tensorflow::shape_inference::ShapeHandle nv_query;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &nv_query));

        // set output shape
        ::tensorflow::shape_inference::ShapeHandle nn_index = c->MakeShape({c->Dim(query, 0), nn_out});
        c->set_output(0, nn_index);
        ::tensorflow::shape_inference::ShapeHandle nn_dist = c->MakeShape({c->Dim(query, 0), nn_out});
        c->set_output(1, nn_dist);

        return Status::OK();
    });


LLint countSphereNeighborLauncher(int B, int Np, int Mp, float radius, int nnSample, const float* database,
                                const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo);
void buildSphereNeighborLauncher(int B, int Np, int Mp, float radius, const float* database,
                                 const float* query, const int* nvDatabase, const int* nvQuery,
                                 const LLint* cntInfo, int* nnIndex, float* nnDist);
class BuildSphereNeighborGpuOp : public OpKernel {
    public:
        explicit BuildSphereNeighborGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("Range search requires radius>0, got ", radius_));

            OP_REQUIRES_OK(context, context->GetAttr("nn_sample", &nn_sample_));
            OP_REQUIRES(context, nn_sample_ > 0, errors::InvalidArgument("BuildSphereNeighbor requires nn_sample>0, got ", nn_sample_));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& database_tensor    = context->input(0);
            const Tensor& query_tensor       = context->input(1);
            const Tensor& nv_database_tensor = context->input(2);
            const Tensor& nv_query_tensor    = context->input(3);

            // get the dims required by computations
            int Np = database_tensor.shape().dim_size(0);    // number of database points
            int Mp = query_tensor.shape().dim_size(0);       // number of query points
            int B  = nv_database_tensor.shape().dim_size(0); // batch size

            OP_REQUIRES(context, database_tensor.dims()==2 && database_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of database points requires to be (Np, 3)"));
            OP_REQUIRES(context, query_tensor.dims()==2 && query_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of query points requires to be (Mp, 3)"));
            OP_REQUIRES(context, nv_database_tensor.dims()==1 && nv_query_tensor.dims()==1 &&
                        nv_database_tensor.shape().dim_size(0)==nv_query_tensor.shape().dim_size(0),
                        errors::InvalidArgument("Shape of nv_database and nv_query should be identical"));

            // flatten the input tensors
            auto database_flat    = database_tensor.flat<float>();
            auto query_flat       = query_tensor.flat<float>();
            auto nv_database_flat = nv_database_tensor.flat<int32>();
            auto nv_query_flat    = nv_query_tensor.flat<int32>();

            // Create an output tensor
            Tensor* cnt_info_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp}, &cnt_info_tensor));
            auto cnt_info_flat = cnt_info_tensor->flat<int64>();

            const float* database   = &(database_flat(0));
            const float* query      = &(query_flat(0));
            const int*   nvDatabase = &(nv_database_flat(0));
            const int*   nvQuery    = &(nv_query_flat(0));

            LLint* cntInfo = &(cnt_info_flat(0));
            cudaMemset(cntInfo, 0, sizeof(LLint)*Mp);
            LLint Nout = countSphereNeighborLauncher(B, Np, Mp, radius_, nn_sample_,
                                                     database, query, nvDatabase, nvQuery, cntInfo);
//            std::cout<<"Nout:"<<Nout<<std::endl;

            Tensor* nn_index_tensor = NULL;
            Tensor* nn_dist_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Nout,2}, &nn_index_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{Nout}, &nn_dist_tensor));
            auto nn_index_flat = nn_index_tensor->flat<int32>();
            auto nn_dist_flat = nn_dist_tensor->flat<float>();
            int*   nnIndex = &(nn_index_flat(0));
            float* nnDist  = &(nn_dist_flat(0));
            cudaMemset(nnIndex, 0, sizeof(int)*Nout*2);
            cudaMemset(nnDist,  0, sizeof(float)*Nout);
            buildSphereNeighborLauncher(B, Np, Mp, radius_, database, query,
                                        nvDatabase, nvQuery, cntInfo, nnIndex, nnDist);
        }
    private:
        float radius_;
        int nn_sample_;
};
REGISTER_KERNEL_BUILDER(Name("BuildSphereNeighbor").Device(DEVICE_GPU), BuildSphereNeighborGpuOp);



LLint countCubeNeighborLauncher(int B, int Np, int Mp, float length, int nnSample, const float* database,
                              const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo);
void buildCubeNeighborLauncher(int B, int Np, int Mp, float length, int gridSize, const float* database,
                               const float* query, const int* nvDatabase, const int* nvQuery,
                               const LLint* cntInfo, int* nnIndex);
class BuildCubeNeighborGpuOp : public OpKernel {
    public:
        explicit BuildCubeNeighborGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("length", &length_));
            OP_REQUIRES(context, length_ > 0, errors::InvalidArgument("Cube size requires length>0, got ", length_));

            OP_REQUIRES_OK(context, context->GetAttr("nn_sample", &nn_sample_));
            OP_REQUIRES(context, nn_sample_ > 0, errors::InvalidArgument("BuildSphereNeighbor requires nn_sample>0, got ", nn_sample_));

            OP_REQUIRES_OK(context, context->GetAttr("grid_size", &grid_size_));
            OP_REQUIRES(context, grid_size_>0, errors::InvalidArgument("Need grid_size_>0, got ", grid_size_));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& database_tensor    = context->input(0);
            const Tensor& query_tensor       = context->input(1);
            const Tensor& nv_database_tensor = context->input(2);
            const Tensor& nv_query_tensor    = context->input(3);

            // get the dims required by computations
            int Np = database_tensor.shape().dim_size(0);    // number of database points
            int Mp = query_tensor.shape().dim_size(0);       // number of query points
            int B  = nv_database_tensor.shape().dim_size(0); // batch size

            OP_REQUIRES(context, database_tensor.dims()==2 && database_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of database points requires to be (Np, 3)"));
            OP_REQUIRES(context, query_tensor.dims()==2 && query_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of query points requires to be (Mp, 3)"));
            OP_REQUIRES(context, nv_database_tensor.dims()==1 && nv_query_tensor.dims()==1 &&
                        nv_database_tensor.shape().dim_size(0)==nv_query_tensor.shape().dim_size(0),
                        errors::InvalidArgument("Shape of nv_database and nv_query should be identical"));

            // flatten the input tensors
            auto database_flat    = database_tensor.flat<float>();
            auto query_flat       = query_tensor.flat<float>();
            auto nv_database_flat = nv_database_tensor.flat<int32>();
            auto nv_query_flat    = nv_query_tensor.flat<int32>();

            // Create an output tensor
            Tensor* cnt_info_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp}, &cnt_info_tensor));
            auto cnt_info_flat = cnt_info_tensor->flat<int64>();

            const float* database   = &(database_flat(0));
            const float* query      = &(query_flat(0));
            const int*   nvDatabase = &(nv_database_flat(0));
            const int*   nvQuery    = &(nv_query_flat(0));

            LLint* cntInfo = &(cnt_info_flat(0));
            cudaMemset(cntInfo, 0, sizeof(LLint)*Mp);
            LLint Nout = countCubeNeighborLauncher(B, Np, Mp, length_, nn_sample_, database, query,
                                                   nvDatabase, nvQuery, cntInfo);

            Tensor* nn_index_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Nout,3}, &nn_index_tensor));
            auto nn_index_flat = nn_index_tensor->flat<int32>();
            int* nnIndex = &(nn_index_flat(0));
            cudaMemset(nnIndex, 0, sizeof(int)*Nout*3);
            buildCubeNeighborLauncher(B, Np, Mp, length_, grid_size_, database, query,
                                      nvDatabase, nvQuery, cntInfo, nnIndex);
        }
    private:
        float length_;
        int nn_sample_, grid_size_;
};
REGISTER_KERNEL_BUILDER(Name("BuildCubeNeighbor").Device(DEVICE_GPU), BuildCubeNeighborGpuOp);


void buildNearestNeighborLauncher(int B, int Np, int Mp, const float* database, const float* query,
                                  const int* nvDatabase, const int* nvQuery, int* nnIndex, float* nnDist);
class BuildNearestNeighborGpuOp : public OpKernel {
    public:
        explicit BuildNearestNeighborGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& database_tensor    = context->input(0);
            const Tensor& query_tensor       = context->input(1);
            const Tensor& nv_database_tensor = context->input(2);
            const Tensor& nv_query_tensor    = context->input(3);

            // get the dims required by computations
            int Np = database_tensor.shape().dim_size(0);    // number of database points
            int Mp = query_tensor.shape().dim_size(0);       // number of query points
            int B  = nv_database_tensor.shape().dim_size(0); // batch size

            OP_REQUIRES(context, database_tensor.dims()==2 && database_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of database points requires to be (Np, 3)"));
            OP_REQUIRES(context, query_tensor.dims()==2 && query_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("Shape of query points requires to be (Mp, 3)"));
            OP_REQUIRES(context, nv_database_tensor.dims()==1 && nv_query_tensor.dims()==1 &&
                        nv_database_tensor.shape().dim_size(0)==nv_query_tensor.shape().dim_size(0),
                        errors::InvalidArgument("Shape of nv_database and nv_query should be identical"));

            // flatten the input tensors
            auto database_flat    = database_tensor.flat<float>();
            auto query_flat       = query_tensor.flat<float>();
            auto nv_database_flat = nv_database_tensor.flat<int32>();
            auto nv_query_flat    = nv_query_tensor.flat<int32>();

            // Create an output tensor
            Tensor* nn_index_tensor = NULL;
            Tensor* nn_dist_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Mp,nn_out_}, &nn_index_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Mp,nn_out_}, &nn_dist_tensor));
            auto nn_index_flat = nn_index_tensor->flat<int32>();
            auto nn_dist_flat = nn_dist_tensor->flat<float>();

            const float* database   = &(database_flat(0));
            const float* query      = &(query_flat(0));
            const int*   nvDatabase = &(nv_database_flat(0));
            const int*   nvQuery    = &(nv_query_flat(0));

            int* nnIndex  = &(nn_index_flat(0));
            float* nnDist = &(nn_dist_flat(0));
            cudaMemset(nnIndex, 0, sizeof(int)*Mp*nn_out_);
            cudaMemset(nnDist,  0, sizeof(float)*Mp*nn_out_);
            buildNearestNeighborLauncher(B, Np, Mp, database, query,
                                         nvDatabase, nvQuery, nnIndex, nnDist);
        }
    private:
        const int nn_out_ = 3; // find 3 nearest neighbors
};
REGISTER_KERNEL_BUILDER(Name("BuildNearestNeighbor").Device(DEVICE_GPU), BuildNearestNeighborGpuOp);






