/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_swap(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasSwapFn = FORTRAN ? hipblasSwap<T, true> : hipblasSwap<T, false>;

    int N          = argus.N;
    int incx       = argus.incx;
    int incy       = argus.incy;
    int unit_check = argus.unit_check;
    int norm_check = argus.norm_check;
    int timing     = argus.timing;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || incy < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int sizeX = N * incx;
    int sizeY = N * incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    // allocate memory on device
    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);
    int              device_pointer = 1;

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);
    hipblas_init<T>(hy, 1, N, incy);
    hx_cpu = hx;
    hy_cpu = hy;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSwapFn(handle, N, dx, incx, dy, incy));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

    if(unit_check || norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_swap<T>(N, hx.data(), incx, hy.data(), incy);

        if(unit_check)
        {
            unit_check_general<T>(1, N, incx, hx_cpu.data(), hx.data());
            unit_check_general<T>(1, N, incy, hy_cpu.data(), hy.data());
        }
        if(norm_check)
        {
            hipblas_error
                = std::max(norm_check_general<T>('F', 1, N, incx, hx_cpu.data(), hx.data()),
                           norm_check_general<T>('F', 1, N, incy, hy_cpu.data(), hy.data()));
        }

    } // end of if unit/norm check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasSwapFn(handle, N, dx, incx, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<T>(std::cout,
                                                         argus,
                                                         gpu_time_used,
                                                         swap_gflop_count<T>(N),
                                                         swap_gbyte_count<T>(N),
                                                         hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
