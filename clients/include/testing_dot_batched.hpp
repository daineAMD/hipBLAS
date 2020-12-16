/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, bool CONJ = false>
hipblasStatus_t testing_dot_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDotBatchedFn
        = FORTRAN ? (CONJ ? hipblasDotcBatched<T, true> : hipblasDotBatched<T, true>)
                  : (CONJ ? hipblasDotcBatched<T, false> : hipblasDotBatched<T, false>);

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || incy < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int sizeX = N * incx;
    int sizeY = N * incy;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_vector<T>       h_cpu_result(batch_count);
    host_vector<T>       h_rocblas_result1(batch_count);
    host_vector<T>       h_rocblas_result2(batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_rocblas_result(batch_count);

    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    int device_pointer = 1;
    int host_pointer   = 1;

    // Initial Data on CPU
    hipblas_init(hx, true);
    hipblas_init(hy, false);
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    // hipblasDot accept both dev/host pointer for the scalar
    if(device_pointer)
    {

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        status_2 = (hipblasDotBatchedFn)(handle,
                                         N,
                                         dx.ptr_on_device(),
                                         incx,
                                         dy.ptr_on_device(),
                                         incy,
                                         batch_count,
                                         d_rocblas_result);
    }
    if(host_pointer)
    {

        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_3 = (hipblasDotBatchedFn)(handle,
                                         N,
                                         dx.ptr_on_device(),
                                         incx,
                                         dy.ptr_on_device(),
                                         incy,
                                         batch_count,
                                         h_rocblas_result2);
    }

    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS)
       || (status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
    {
        hipblasDestroy(handle);
        if(status_1 != HIPBLAS_STATUS_SUCCESS)
            return status_1;
        if(status_2 != HIPBLAS_STATUS_SUCCESS)
            return status_2;
        if(status_3 != HIPBLAS_STATUS_SUCCESS)
            return status_3;
        if(status_4 != HIPBLAS_STATUS_SUCCESS)
            return status_4;
    }

    if(device_pointer)
        CHECK_HIP_ERROR(hipMemcpy(
            h_rocblas_result1, d_rocblas_result, sizeof(T) * batch_count, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx[b], incx, hy[b], incy, &(h_cpu_result[b]));
        }

        if(argus.unit_check)
        {
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_rocblas_result1);
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_rocblas_result2);
        }

    } // end of if unit/norm check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_dotc_batched(const Arguments& argus)
{
    return testing_dot_batched<T, true>(argus);
}
