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
hipblasStatus_t testing_dgmm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDgmmBatchedFn
        = FORTRAN ? hipblasDgmmBatched<T, true> : hipblasDgmmBatched<T, false>;

    hipblasSideMode_t side = char2hipblas_side(argus.side_option);

    int M           = argus.M;
    int N           = argus.N;
    int lda         = argus.lda;
    int incx        = argus.incx;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    int A_size = size_t(lda) * N;
    int C_size = size_t(ldc) * N;
    int k      = (side == HIPBLAS_SIDE_RIGHT ? N : M);
    int X_size = size_t(incx) * k;
    if(!X_size)
        X_size = 1;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hA_copy(A_size, 1, batch_count);
    host_batch_vector<T> hx(X_size, 1, batch_count);
    host_batch_vector<T> hx_copy(X_size, 1, batch_count);
    host_batch_vector<T> hC(C_size, 1, batch_count);
    host_batch_vector<T> hC_1(C_size, 1, batch_count);
    host_batch_vector<T> hC_gold(C_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(X_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    hipblas_init(hA, true);
    hipblas_init(hx, false);
    hipblas_init(hC, false);

    hA_copy.copy_from(hA);
    hx_copy.copy_from(hx);
    hC_1.copy_from(hC);
    hC_gold.copy_from(hC);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    status = hipblasDgmmBatchedFn(handle,
                                  side,
                                  M,
                                  N,
                                  dA.ptr_on_device(),
                                  lda,
                                  dx.ptr_on_device(),
                                  incx,
                                  dC.ptr_on_device(),
                                  ldc,
                                  batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    CHECK_HIP_ERROR(hC_1.transfer_from(dC));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        for(int b = 0; b < batch_count; b++)
        {
            for(size_t i1 = 0; i1 < M; i1++)
            {
                for(size_t i2 = 0; i2 < N; i2++)
                {
                    if(HIPBLAS_SIDE_RIGHT == side)
                    {
                        hC_gold[b][i1 + i2 * ldc]
                            = hA_copy[b][i1 + i2 * lda] * hx_copy[b][i2 * incx];
                    }
                    else
                    {
                        hC_gold[b][i1 + i2 * ldc]
                            = hA_copy[b][i1 + i2 * lda] * hx_copy[b][i1 * incx];
                    }
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_1);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
