#pragma once

#include <memory>
#include <vector>
#include <string>
#include <CL/cl.h>   // needed only because voronoi.cpp expects cl_mem

#include "basic.h"   // for int
#include "cvoro_config.h"

class OpenCLContext;

/// ---------------------------------------------------------------------------
/// CUDA-backed replacement for original KNearests class
/// Matches original API in knearests.h
/// ---------------------------------------------------------------------------
class CudaKNearests {
public:
    CudaKNearests(std::shared_ptr<OpenCLContext> context,
                  cl_mem pointsCL,
                  int numpoints,
                  bool debug = false);

    ~CudaKNearests();

    // called from voronoi.cpp: builds KNN for a batch of ids
    bool buildKnearests(int K,
                        cl_mem idsCL,
                        int numIds,
                        int id_offset,
                        double &totalTime);

    // matches KNearests API
    int getNumPoints() const { return allocated_points; }
    cl_mem getPoints() const { return gpu_stored_points; }
    cl_mem getPermutation() const { return gpu_permutation; }
    cl_mem getNearests() const { return nearest_knearests; }

    void printStats(std::ostream &os) const;

private:
    // OpenCL context (voronoi.cpp requires this to exist)
    std::shared_ptr<OpenCLContext> m_context;

    int allocated_points;

    // CL buffers required by voronoi.cpp
    cl_mem gpu_stored_points;      // copy of input points
    cl_mem gpu_permutation;        // trivial perm (0..n-1)
    cl_mem nearest_knearests;      // result buffer (n x K)

    bool m_debug;

    // host copies for CUDA usage
    std::vector<real> host_points; // SoA: x0 x1 x2 ...
    std::vector<int> host_perm;
};


/// ---------------------------------------------------------------------------
/// CUDA KNN wrapper used inside buildKnearests
/// ---------------------------------------------------------------------------
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
);

