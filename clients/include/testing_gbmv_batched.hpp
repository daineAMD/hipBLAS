/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_gbmvBatched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGbmvBatchedFn
        = FORTRAN ? hipblasGbmvBatched<T, true> : hipblasGbmvBatched<T, false>;

    int M    = argus.M;
    int N    = argus.N;
    int KL   = argus.KL;
    int KU   = argus.KU;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    int A_size = lda * N;
    int X_size;
    int Y_size;
    int X_els;
    int Y_els;

    int batch_count = argus.batch_count;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    // transA = HIPBLAS_OP_T;
    if(transA == HIPBLAS_OP_N)
    {
        X_els = N;
        Y_els = M;
    }
    else
    {
        X_els = M;
        Y_els = N;
    }
    X_size = X_els * incx;
    Y_size = Y_els * incy;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || KL < 0 || KU < 0 || lda < KL + KU + 1 || incx == 0 || incy == 0
       || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = (T)argus.alpha;
    T beta  = (T)argus.beta;

    // arrays of pointers-to-host on host
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hx(X_size, 1, batch_count);
    host_batch_vector<T> hy(Y_size, 1, batch_count);
    host_batch_vector<T> hz(Y_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(X_size, 1, batch_count);
    device_batch_vector<T> dy(Y_size, 1, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    // Initial Data on CPU
    hipblas_init(hA, true);
    hipblas_init(hx, false);
    hipblas_init(hy, false);
    hz.copy_from(hy);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasGbmvBatchedFn(handle,
                                      transA,
                                      M,
                                      N,
                                      KL,
                                      KU,
                                      (T*)&alpha,
                                      dA.ptr_on_device(),
                                      lda,
                                      dx.ptr_on_device(),
                                      incx,
                                      (T*)&beta,
                                      dy.ptr_on_device(),
                                      incy,
                                      batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            // here in cuda
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hy.transfer_from(dy));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_gbmv<T>(transA, M, N, KL, KU, alpha, hA[b], lda, hx[b], incx, beta, hz[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, batch_count, incy, hz, hy);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
