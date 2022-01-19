#include <assert.h>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
// #include <hipblas.h>
#include "hipblas.h"

void getri(
    hipblasHandle_t* handle, int* n, double* A, int* lda, int* ipiv, double* lwork, int* info);

int main(int argc, char** argv)
{

    std::vector<double> const test{
        0.767135868133925, -0.641484652834663, 0.641484652834663, 0.767135868133926};
    std::vector<int>    ipiv(2);
    std::vector<int>    info(10);
    std::vector<double> work(4);
    int                 n   = 2;
    int                 lda = 4;

    double *test_d, *work_d;
    int *   ipiv_d, *info_d;

    // Create device copies of input matrix, ipiv, info, and work buffers
    auto success = hipMalloc((void**)&ipiv_d, ipiv.size() * sizeof(int*));
    assert(success == hipSuccess);
    success = hipMemcpy(ipiv_d, ipiv.data(), ipiv.size() * sizeof(int), hipMemcpyHostToDevice);
    assert(success == 0);

    success = hipMalloc((void**)&info_d, info.size() * sizeof(int*));
    assert(success == hipSuccess);
    success = hipMemcpy(info_d, info.data(), info.size() * sizeof(int), hipMemcpyHostToDevice);
    assert(success == 0);

    success = hipMalloc((void**)&test_d, test.size() * sizeof(double));
    assert(success == hipSuccess);
    success = hipMemcpy(test_d, test.data(), test.size() * sizeof(double), hipMemcpyHostToDevice);
    assert(success == 0);

    success = hipMalloc((void**)&work_d, work.size() * sizeof(double));
    assert(success == hipSuccess);
    success = hipMemcpy(work_d, work.data(), work.size() * sizeof(double), hipMemcpyHostToDevice);
    assert(success == 0);

    // Make a hipblas handle
    hipblasHandle_t handle;
    auto            hipblas_success = hipblasCreate(&handle);
    assert(hipblas_success == HIPBLAS_STATUS_SUCCESS);

    // Call hipBlasDgetriBatched
    std::cout << "Running getri..\n";
    getri(&handle, &n, test_d, &lda, ipiv_d, work_d, info_d);

    success = hipDeviceSynchronize();
    assert(success == 0);
    std::cout << " -- after getri\n";

    hipblas_success = hipblasDestroy(handle);
    assert(hipblas_success == HIPBLAS_STATUS_SUCCESS);

    std::cout << "Done\n";

    return 0;
}

void getri(
    hipblasHandle_t* handle, int* n, double* A, int* lda, int* ipiv, double* lwork, int* info)
{
    double** A_d;
    double** work_d;

    auto stat = hipMalloc((void**)&A_d, sizeof(double*));
    assert(stat == 0);
    stat = hipMemcpy(A_d, &A, sizeof(double*), hipMemcpyHostToDevice);
    assert(stat == 0);

    stat = hipMalloc((void**)&work_d, sizeof(double*));
    assert(stat == 0);
    stat = hipMemcpy(work_d, &lwork, sizeof(double*), hipMemcpyHostToDevice);
    assert(stat == 0);

    auto const success = hipblasDgetriBatched(*handle, *n, A_d, *lda, nullptr, work_d, *n, info, 1);
    std::cout << "status: " << success << "\n";
    assert(success == 0);
}
