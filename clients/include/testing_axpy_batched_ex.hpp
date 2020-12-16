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

template <typename Ta, typename Tx = Ta, typename Ty = Tx>
hipblasStatus_t testing_axpy_batched_ex_template(Arguments argus)
{
    bool FORTRAN                = argus.fortran;
    auto hipblasAxpyBatchedExFn = FORTRAN ? hipblasAxpyBatchedExFortran : hipblasAxpyBatchedEx;
    hipblasStatus_t status      = HIPBLAS_STATUS_SUCCESS;

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

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

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    int sizeX = N * abs_incx;
    int sizeY = N * abs_incy;
    Ta  alpha = argus.alpha;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_batch_vector<Ty> hy(N, incy, batch_count);
    host_batch_vector<Tx> hx_cpu(N, incx, batch_count);
    host_batch_vector<Ty> hy_cpu(N, incy, batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    hipblas_init(hx, true);
    hipblas_init(hy, false);
    hx_cpu.copy_from(hx);
    hy_cpu.copy_from(hy);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    status = hipblasAxpyBatchedExFn(handle,
                                    N,
                                    &alpha,
                                    alphaType,
                                    dx.ptr_on_device(),
                                    xType,
                                    incx,
                                    dy.ptr_on_device(),
                                    yType,
                                    incy,
                                    batch_count,
                                    executionType);
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
            cblas_axpy<Tx>(N, alpha, hx_cpu[b], incx, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, abs_incx, hx_cpu, hx);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy);
        }

    } // end of if unit check

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_axpy_batched_ex(Arguments argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_axpy_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_axpy_batched_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_axpy_batched_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_axpy_batched_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_axpy_batched_ex_template<hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
