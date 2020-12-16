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

template <typename T>
hipblasStatus_t testing_axpy_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasAxpyBatchedFn
        = FORTRAN ? hipblasAxpyBatched<T, true> : hipblasAxpyBatched<T, false>;

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

    hipblasStatus_t status   = HIPBLAS_STATUS_SUCCESS;
    int             abs_incx = incx < 0 ? -incx : incx;
    int             abs_incy = incy < 0 ? -incy : incy;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || !incx || !incy || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int sizeX = N * abs_incx;
    int sizeY = N * abs_incy;
    T   alpha = argus.alpha;

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hx_cpu(N, incx, batch_count);
    host_batch_vector<T> hy_cpu(N, incy, batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);

    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    hipblas_init(hx, true);
    hipblas_init(hy, false);
    hx_cpu.copy_from(hx);
    hy_cpu.copy_from(hy);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    status = hipblasAxpyBatchedFn(
        handle, N, &alpha, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    CHECK_HIP_ERROR(hx.transfer_from(dx));
    CHECK_HIP_ERROR(hy.transfer_from(dy));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy<T>(N, alpha, hx_cpu[b], incx, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, hx_cpu, hx);
            unit_check_general<T>(1, N, batch_count, abs_incy, hy_cpu, hy);
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
