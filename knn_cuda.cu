// REWRITTEN knn_cuda.cu using float* with 3 floats per point instead of real4.
// This version keeps OpenCL/Voronoi compatibility but the CUDA KNN path uses
// tightly packed XYZ floats.

#include "knn_cuda.h"

#include "bitonic-hubs-grid.cuh"
#include "bitonic-shared.cu"
#include "cuda_util.cu"
#include "spatial.cu"

#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <memory>

// -----------------------------------------------------------------------------
// Utility: allocate an OpenCL buffer and copy from host
// -----------------------------------------------------------------------------
static cl_mem createCLBufferFromHost(std::shared_ptr<OpenCLContext> ctx,
                                     const void* data,
                                     size_t bytes) {
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx->getContext(),
                                CL_MEM_READ_WRITE,
                                bytes,
                                nullptr,
                                &err);

    clEnqueueWriteBuffer(ctx->getQueue(),
                         buf,
                         CL_TRUE,
                         0,
                         bytes,
                         data,
                         0,
                         nullptr,
                         nullptr);
    return buf;
}

// =============================================================================
// CudaKNearests constructor
// =============================================================================
CudaKNearests::CudaKNearests(std::shared_ptr<OpenCLContext> context,
                             cl_mem pointsCL,
                             int numpoints,
                             bool debug)
    : m_context(context),
      allocated_points(numpoints),
      m_debug(debug) {

    size_t clBytes = sizeof(real) * size_t(numpoints) * 4;

    std::vector<real> temp_cl_points(numpoints * 4);

    clEnqueueReadBuffer(m_context->getQueue(),
                        pointsCL,
                        CL_TRUE,
                        0,
                        clBytes,
                        temp_cl_points.data(),
                        0,
                        nullptr,
                        nullptr);

    // Now convert real4 -> float3
    host_points.resize(numpoints * 3);
    for (int i = 0; i < numpoints; i++) {
        host_points[i * 3 + 0] = temp_cl_points[i * 4 + 0];
        host_points[i * 3 + 1] = temp_cl_points[i * 4 + 1];
        host_points[i * 3 + 2] = temp_cl_points[i * 4 + 2];
    }

    // trivial permutation
    host_perm.resize(numpoints);
    for (int i = 0; i < numpoints; i++) host_perm[i] = i;

    gpu_stored_points = createCLBufferFromHost(m_context,
                                               temp_cl_points.data(),
                                               clBytes);

    gpu_permutation = createCLBufferFromHost(m_context,
                                             host_perm.data(),
                                             sizeof(int) * numpoints);

    nearest_knearests = nullptr;
}

CudaKNearests::~CudaKNearests() {
    if (gpu_stored_points) clReleaseMemObject(gpu_stored_points);
    if (gpu_permutation) clReleaseMemObject(gpu_permutation);
    if (nearest_knearests) clReleaseMemObject(nearest_knearests);
}

void CudaKNearests::printStats(std::ostream &os) const {
    os << "CudaKNearests n=" << allocated_points << "\n";
}

template <class R>
void save_debug_data(
    size_t totalPoints,
    R* pointData,
    int q,
    const std::vector<int>& batch_queries)
{
    std::filesystem::create_directories("knn_debug");

    // ---- Save totalPoints ----
    {
        std::ofstream ofs("knn_debug/totalPoints.txt");
        ofs << totalPoints << "\n";
    }

    // ---- Save pointData as readable text (optional) ----
    {
        std::ofstream ofs("knn_debug/pointData.txt");
        for (size_t i = 0; i < totalPoints; ++i) {
            ofs << pointData[i*3 + 0] << " " << pointData[i*3 + 1] << " " << pointData[i*3 + 2] << "\n";
        }
    }

    // ---- Save q ----
    {
        std::ofstream ofs("knn_debug/q.txt");
        ofs << q << "\n";
    }

    // ---- Save batch_queries ----
    {
        std::ofstream ofs("knn_debug/batch_queries.txt");
        for (int i = 0; i < q; ++i) {
            ofs << batch_queries[i] << "\n";
        }
    }
}

// =============================================================================
// CUDA compute KNN wrapper
// =============================================================================
template <class R>
bool computeKNN_CUDA(
    int num,
    int kOffset,
    int K,
    const int* ids,
    const std::string& mesh_name,
    double& sum_knn,
    size_t totalPoints,
    R* pointData,            
    int* KNearestIndices,
    R* KNearestDistances
) {
    std::size_t q = num;

    std::vector<int> batch_queries(q);
    for (int i = 0; i < num; i++)
        batch_queries[i] = ids[kOffset + i];

    std::vector<int> results_knn(q * K);
    std::vector<R> results_distances(q * K);
    int best_p = -1;
    float time_ms = 0.0f;

    // if(q == 11083)
    // {
    //     save_debug_data<R>(
    //     totalPoints,
    //     pointData,
    //     q,
    //     batch_queries
    // );
    // }


    bitonic_hubs_grid::C_and_Q<R>(
        totalPoints,
        pointData,
        q,
        batch_queries.data(),
        K,
        &best_p,
        results_knn.data(),
        results_distances.data(),
        const_cast<std::string&>(mesh_name),
        &time_ms
    );

    sum_knn += time_ms;

    for (int qi = 0; qi < num; qi++) {
        int global_id = ids[kOffset + qi];
        for (int kk = 0; kk < K; kk++) {
            KNearestIndices[global_id * K + kk] = results_knn[qi * K + kk];
            KNearestDistances[global_id * K + kk] = results_distances[qi * K + kk];
        }
    }
    return true;
}

// =============================================================================
// Main entry from Voronoi (OpenCL expects CL buffers)
// =============================================================================
bool CudaKNearests::buildKnearests(int K,
                                   cl_mem idsCL,
                                   int numIds,
                                   int kOffset,
                                   double &totalTime) {

    std::vector<int> ids(numIds);
    clEnqueueReadBuffer(m_context->getQueue(),
                        idsCL,
                        CL_TRUE,
                        0,
                        sizeof(int) * numIds,
                        ids.data(),
                        0,
                        nullptr,
                        nullptr);

    if (!nearest_knearests) {
        nearest_knearests =
            clCreateBuffer(m_context->getContext(),
                           CL_MEM_READ_WRITE,
                           sizeof(int) * allocated_points * K,
                           nullptr,
                           nullptr);
    }

    std::vector<real> dist_host(allocated_points * K);
    std::vector<int> idx_host(allocated_points * K);

    bool ok = computeKNN_CUDA<real>(
        numIds,
        kOffset,
        K,
        ids.data(),
        std::string("mesh"),
        totalTime,
        allocated_points,
        host_points.data(),      // float3 array
        idx_host.data(),
        dist_host.data()
    );

    clEnqueueWriteBuffer(m_context->getQueue(),
                         nearest_knearests,
                         CL_TRUE,
                         0,
                         sizeof(int) * allocated_points * K,
                         idx_host.data(),
                         0,
                         nullptr,
                         nullptr);

    return ok;
}
