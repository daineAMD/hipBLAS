/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_tpsv_batched(Arguments argus)
{
    int                N           = argus.N;
    int                incx        = argus.incx;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);
    int                batch_count = argus.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int size_A   = N * N;
    int size_AP  = N * (N + 1) / 2;
    int size_x   = abs_incx * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hAP[batch_count];
    host_vector<T> AAT[batch_count];
    host_vector<T> hb[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hx_or_b_1[batch_count];
    host_vector<T> hx_or_b_2[batch_count];
    host_vector<T> cpu_x_or_b[batch_count];

    device_batch_vector<T> bAP(batch_count, size_AP);
    device_batch_vector<T> bx_or_b(batch_count, size_x);

    device_vector<T*, 0, T> dAP(batch_count);
    device_vector<T*, 0, T> dx_or_b(batch_count);

    int last = batch_count - 1;
    if(!dAP || !dx_or_b || (!bAP[last] && size_AP) || (!bx_or_b[last] && size_x))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]         = host_vector<T>(size_A);
        hAP[b]        = host_vector<T>(size_AP);
        AAT[b]        = host_vector<T>(size_A);
        hb[b]         = host_vector<T>(size_x);
        hx[b]         = host_vector<T>(size_x);
        hx_or_b_1[b]  = host_vector<T>(size_x);
        hx_or_b_2[b]  = host_vector<T>(size_x);
        cpu_x_or_b[b] = host_vector<T>(size_x);

        srand(1);
        hipblas_init<T>(hA[b], N, N, N);

        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(HIPBLAS_OP_N, HIPBLAS_OP_T, N, N, N, 1.0, hA[b], N, hA[b], N, 0.0, AAT[b], N);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < N; i++)
        {
            T t = 0.0;
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * N] = AAT[b][i + j * N];
                t += abs(AAT[b][i + j * N]);
            }
            hA[b][i + i * N] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, N, hA[b], N);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < N; i++)
                {
                    T diag = hA[b][i + i * N];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * N] = hA[b][i + j * N] / diag;
                }
            else
                for(int j = 0; j < N; j++)
                {
                    T diag = hA[b][j + j * N];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * N] = hA[b][i + j * N] / diag;
                }
        }

        hipblas_init<T>(hx[b], 1, N, abs_incx);
        hb[b] = hx[b];

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hb[b], incx);
        cpu_x_or_b[b] = hb[b]; // cpuXorB <- B
        hx_or_b_1[b]  = hb[b];
        hx_or_b_2[b]  = hb[b];

        regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], (T*)hAP[b], N);

        // copy data from CPU to device
        hipMemcpy(bAP[b], hAP[b], sizeof(T) * size_AP, hipMemcpyHostToDevice);
        hipMemcpy(bx_or_b[b], hx_or_b_1[b], sizeof(T) * size_x, hipMemcpyHostToDevice);
    }
    hipMemcpy(dAP, bAP, sizeof(T*) * batch_count, hipMemcpyHostToDevice);
    hipMemcpy(dx_or_b, bx_or_b, sizeof(T*) * batch_count, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status
            = hipblasTpsvBatched<T>(handle, uplo, transA, diag, N, dAP, dx_or_b, incx, batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hx_or_b_1[b], bx_or_b[b], sizeof(T) * size_x, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
            double    tolerance = eps * 40 * N;

            double error = 0.0, max_err_scal = 0.0, max_err = 0.0;
            for(int i = 0; i < N; i++)
            {
                T diff = (hx[b][i * abs_incx] - hx_or_b_1[b][i * abs_incx]);
                if(diff != T(0))
                {
                    max_err += abs(diff);
                }
                max_err_scal += abs(hx_or_b_1[b][i * abs_incx]);
            }
            error = max_err / max_err_scal;
            unit_check_error(error, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}