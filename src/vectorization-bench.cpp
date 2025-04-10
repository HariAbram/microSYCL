#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
//#include <omp.h>

#ifndef TYPE
#define TYPE double
#endif

#ifndef ALIGNMENT
#define ALIGNMENT 32
#endif

#ifndef OPT_BLOCK_SIZE
#define OPT_BLOCK_SIZE 32
#endif

#include "../include/timer.hpp"
#include "../include/utils.hpp"

bool verification (TYPE *m1, TYPE *m2 , TYPE *m3, int size)
{
    bool result = true;

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            TYPE temp = 0.0;
            for (size_t k = 0; k < size; k++)
            {
                temp += m2[i*size+k]*m1[k*size+j];
            }
            if (m3[i*size+j] != temp)
            {
                std::cout<< "verification failed at these indices i: "<<i<<" j: "<<j<<std::endl;
                result = false;
                return result;
            }
        }        
    }
    return result;

}

bool verification_spmv (TYPE *m1, TYPE *v1 , TYPE *v2, int size)
{
    bool result = true;

    for (size_t i = 0; i < size; i++)
    {
        TYPE temp = 0.0;
        for (size_t j = 0; j < size; j++)
        {
            temp += m1[i*size+j]*m1[j];
            
        }   
        if (v2[i] != temp)
        {
            result = false;
            return result;
        }     
    }
    return result;

}

bool verification_stencil_1 (TYPE *m1, TYPE *m2, int size)
{
    bool result = true;

    for (size_t i = 1; i < size-1; i++)
    { 
        TYPE temp = 0.0;
        for (size_t j = 1; j < size-1; j++)
        {
            TYPE temp1 = m1[i*size+j];
            TYPE temp2 = m1[(i-1)*size+j];
            TYPE temp3 = m1[(i+1)*size+j];
            TYPE temp4 = m1[i*size+(j-1)];
            TYPE temp5 = m1[i*size+(j-1)];

            temp = temp1 + temp2 + temp3 + temp4 + temp5;

            if (temp != m2[i*size+j])
            {
                result = false;
                return result;
            }
        }     
    }
    return result;

}

////////////////////////////////////////////////////////// mat-vec


void gemv_range_usm(sycl::queue &Q, int size)
{
    timer time;

    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,0.0);
    std::fill(m1,m1+(size*size),2.0);

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMV");
    }

    time.start_timer();
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int i = it.get_id(0);

            const int N = it.get_range(0);
            TYPE temp = 0.0;
            for (size_t k = 0; k < N; k++)
            {
                temp += m1[i*N+k] * v1[k];
            }
            v2[i] = temp;
        });
    });
    Q.wait();
    time.end_timer();

    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMV");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemv with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(v1,Q);
    sycl::free(v2,Q);
    sycl::free(m1,Q);

}

void gemv_range_buff_acc(sycl::queue &Q, int size)
{
    timer time;

    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,0.0);
    std::fill(m1,(m1+size*size),2.0);

    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);
    sycl::buffer<TYPE,1> m1_buff(m1,size*size);

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMV");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int i = it.get_id(0);

            const int N = it.get_range(0);
            TYPE temp = 0.0;
            for (size_t k = 0; k < N; k++)
            {
                temp += m1_acc[i*N+k] * v1_acc[k];
            }
            v2_acc[i] = temp;
        });
    });
    Q.wait();
    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMV");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemv with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(v1);
    free(v2);
    free(m1);

}

void gemv_ndrange_usm(sycl::queue &Q, int size, int block_size)
{
    timer time;

    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,0.0);
    std::fill(m1,m1+(size*size),2.0);

    Q.wait();

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMV");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            const int i = it.get_global_id(0);
            const int N = it.get_global_range(0);
            TYPE temp = 0.0;
            for (size_t k = 0; k < N; k++)
            {
                temp += m1[i*N+k] * v1[k];
            }
            v2[i] = temp;
        });
    });
    Q.wait();
    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMV");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemv with ndrange ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(v1,Q);
    sycl::free(v2,Q);
    sycl::free(m1,Q);

}

void gemv_ndrange_buff_acc(sycl::queue &Q, int size, int block_size)
{
    timer time;

    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,0.0);
    std::fill(m1,m1+(size*size),2.0);

    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);
    sycl::buffer<TYPE,1> m1_buff(m1,size*size);

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMV");
    }
    time.start_timer();
    
    Q.submit([&](sycl::handler& cgh){
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            const int i = it.get_global_id(0);
            const int N = it.get_global_range(0);
            TYPE temp = 0.0;
            for (size_t k = 0; k < N; k++)
            {
                temp += m1_acc[i*N+k] * v1_acc[k];
            }
            v2_acc[i] = temp;
        });
    });
    Q.wait();
    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMV");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemv with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(v1);
    free(v2);
    free(m1);
}

////////////////////////////////////////////////////////// mat-mul

void gemm_range_usm(sycl::queue &Q, int size)
{
    timer time;

    auto N = static_cast<size_t>(size);
    

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,1.0);
    std::fill(m2,m2+size*size,1.0);
    std::fill(m3,m3+size*size,0.0);

    sycl::range<2> global1 {N,N};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMM");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for< >(sycl::range<2>(global1), [=](sycl::item<2>it){

            auto i = it.get_id(0);
            auto j = it.get_id(1);

            float temp = 0.0;

            for (size_t k = 0; k < N; k++)
            {
                temp += m2[i*N+k]*m1[k*N+j];
            }

            m3[i*N+j] = temp;
        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMM");
    }

#ifdef VERIFY
    if (m3[0]!= size)
    {
        std::cout<< "Verification Failed" << std::endl;
    }
#endif

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemm with range ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(m2,Q);
    sycl::free(m3,Q);
}

void gemm_range_buff_acc(sycl::queue &Q, int size)
{

    auto N = static_cast<size_t>(size);

    timer time;  
    

    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m2 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m3 = (TYPE *)malloc(size*size*sizeof(TYPE));

    std::fill(m1,m1+size*size,1.0);
    std::fill(m2,m2+size*size,1.0);
    std::fill(m3,m3+size*size,0.0);

    sycl::buffer<TYPE,1> m1_buff(m1,size*size);
    sycl::buffer<TYPE,1> m2_buff(m2,size*size);
    sycl::buffer<TYPE,1> m3_buff(m3,size*size);

    sycl::range<2> global1 {N,N};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMM");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read>(cgh);
        auto m2_acc = m2_buff.get_access<sycl::access::mode::read>(cgh);
        auto m3_acc = m3_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for< >(sycl::range<2>(global1), [=](sycl::item<2>it){

            auto i = it.get_id(0);
            auto j = it.get_id(1);

            TYPE temp = 0.0;

            for (size_t k = 0; k < N; k++)
            {
                temp += m2_acc[i*N+k]*m1_acc[k*N+j];
            }

            m3_acc[i*N+j] = temp;
        });
    });
    Q.wait();
    
    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMM");
    }

#ifdef VERIFY
    auto m3_r = m3_buff.get_host_access();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
#endif

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemm with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(m1);
    free(m2);
    free(m3);
}

void gemm_ndrange_usm(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    timer time;
#ifdef ALIGNED
    TYPE * __restrict__ m1 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*sizeof(TYPE),Q); Q.wait();

#else
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
#endif

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);
    std::fill(m3,m3+size*size,0.0);

    //auto N_m = static_cast<size_t>(size*size);

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMM");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);

            TYPE temp = 0.0;

            for (size_t k = 0; k < N; k++)
            {
                temp+= m2[i*N+k]*m1[k*N+j];
            }

            m3[i*N+j] = temp;

        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMM");
    }

#ifdef VERIFY
    if (m3[0]!= size)
    {
        std::cout<< "Verification Failed" << std::endl;
    }
#endif

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemm with nd_range ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(m2,Q);
    sycl::free(m3,Q);
}

void gemm_ndrange_buff_acc(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    timer time;

    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m2 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m3 = (TYPE *)malloc(size*size*sizeof(TYPE));

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);
    std::fill(m3,m3+size*size,0.0);

    sycl::buffer<TYPE,1> m1_buff(m1,size*size);
    sycl::buffer<TYPE,1> m2_buff(m2,size*size);
    sycl::buffer<TYPE,1> m3_buff(m3,size*size);

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMM");
    }
    time.start_timer();

 
    Q.submit([&](sycl::handler& cgh){
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read>(cgh);
        auto m2_acc = m2_buff.get_access<sycl::access::mode::read>(cgh);
        auto m3_acc = m3_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);

            TYPE temp = 0.0;

            for (size_t k = 0; k < N; k++)
            {
                temp += m2_acc[i*N+k]*m1_acc[k*N+j];
            }

            m3_acc[i*N+j] = temp;

        });

    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMM");
    }

#ifdef VERIFY
    auto m3_r = m3_buff.get_host_access();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
#endif 

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemm with nd_range( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(m1);
    free(m2);
    free(m3);

}

//optimized gemm
void gemm_opt_ndrange_usm(sycl::queue &Q, int size, int block_size){
    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    timer time;

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);
    std::fill(m3,m3+size*size,0.0); Q.wait();

    sycl::range<2> global1 {N,N};
    
    sycl::range<2> local1{OPT_BLOCK_SIZE,OPT_BLOCK_SIZE};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("GEMM-OPT");
    }
    time.start_timer();
 
    Q.submit([&](sycl::handler& cgh){

        sycl::accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local>  localm1(sycl::range<2>(OPT_BLOCK_SIZE, OPT_BLOCK_SIZE), cgh);
        sycl::accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local>  localm2(sycl::range<2>(OPT_BLOCK_SIZE, OPT_BLOCK_SIZE), cgh);

        cgh.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            int i = it.get_global_id(0);
            int j = it.get_global_id(1);
            int ii = it.get_local_id(0);
            int jj = it.get_local_id(1);            

            TYPE acc = 0.0;

            // Loop over the tiles required to compute the C element.
            for (int t = 0; t < (N+OPT_BLOCK_SIZE -1) / OPT_BLOCK_SIZE; t++) {
                // Load a BLOCK_SIZE x BLOCK_SIZE tile of matrix m1 into local memory.
                int m1Row = i;
                int m1Col = t * OPT_BLOCK_SIZE + jj;
                localm1[ii][jj] = m1[m1Row * N + m1Col];
    
                // Load a BLOCK_SIZE x BLOCK_SIZE tile of matrix m2 into local memory.
                int m2Row = t * OPT_BLOCK_SIZE + ii;
                int m2Col = j;
                localm2[ii][jj] = m2[m2Row * N + m2Col];
    
                // Synchronize to ensure the tile is loaded.
                it.barrier(sycl::access::fence_space::local_space);
    
                // Multiply the two tiles together.
                for (int k = 0; k < OPT_BLOCK_SIZE; k++) {
                    acc += localm1[ii][k] * localm2[k][jj];
                }
    
                // Synchronize to ensure that computation using the current tile is done.
                it.barrier(sycl::access::fence_space::local_space);
            }
    
            // Write the result to matrix m3.
            m3[i * N + j] = acc;
                
        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("GEMM-OPT");
    }
    
#ifdef VERIFY
    bool verify = verification(m1, m2, m3, size);
    if (verify == false)
    {
        std::cout << "Verification Failed " << std::endl;
    }
#endif

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : gemm with nd_range( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(m2,Q);
    sycl::free(m3,Q);
}

////////////////////////////////////////////////////////// outer-product

void outer_product(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);

    timer time;

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q);
    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q);
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,0.0);
    std::fill(v1,v1+size,1);
    std::fill(v2,v2+size,1);

    Q.wait();

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("OUT-PRODUCT");
    }
    time.start_timer();
 
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);
            auto N = it.get_global_range(0);

            m1[i*N+j] = v1[i]*v2[j];
        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("OUT-PRODUCT");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : outer product with ndrange ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(v1,Q);
    sycl::free(v2,Q);
}

////////////////////////////////////////////////////////// triad

void triad(sycl::queue &Q, int size, int block_size)
{
    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);

    timer time;

    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); 
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); 
    TYPE * __restrict__ v3 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q);Q.wait(); 

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,2.0);
    std::fill(v3,v3+size,0.0);

    Q.wait();

    sycl::range<1> global1{N};
    sycl::range<1> local1{N_b};
    
    #pragma omp parallel
    {
        LIKWID_MARKER_START("TRIAD");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for< >(sycl::nd_range<1>(global1,local1), [=](sycl::nd_item<1>it){

            auto i = it.get_global_id(0);

            v3[i]  = v2[i] + v1[i]* N;
        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("TRIAD");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : triad with ndrange ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(v1,Q);
    sycl::free(v2,Q);
    sycl::free(v3,Q);

}

////////////////////////////////////////////////////////// cross-product

void cross_product(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);

    timer time;

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q);
    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q);
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,0.0);
    std::fill(v1,v1+size,1);
    std::fill(v2,v2+size,1);

    Q.wait();

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};
    
    #pragma omp parallel
    {
        LIKWID_MARKER_START("CROSS-PRODUCT");
    }
    time.start_timer();
 
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);
            auto N = it.get_global_range(0);
            auto tmp= std::pow(-1, i+j);

            if (i == j)
            {
                m1[i*N+j] = 0;
            }
            else
            {
                m1[i*N+j] = v1[i]*v2[j]*tmp;
            }
            
        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("CROSS-PRODUCT");
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : cross product with ndrange ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(v1,Q);
    sycl::free(v2,Q);
}

////////////////////////////////////////////////////////// SPMV

void spmv_csr_ndrange_usm(sycl::queue &Q, int size, int block_size){
    
    timer time;
    int sparsity=10;

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N*N};

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,0.0);
    init_sparse_arrays(m1, size, sparsity);

    Q.wait();

    int* nnz = sycl::malloc_shared<int>(1*sizeof(int),Q); Q.wait();
    *nnz = 0;
    
    Q.submit([&](sycl::handler &cgh){
        auto nnz_red = sycl::reduction(nnz, sycl::plus<int>()); 
        
        cgh.parallel_for(sycl::range<>(global), nnz_red,[=](sycl::item<1> it, auto &nnz){
            auto k = it.get_id(0);

            if (m1[k]!=0)
            {
                nnz+=1;
            }
        });
    });
    Q.wait();

    int * __restrict__ row          =  sycl::malloc_shared<int>(sizeof(int)*nnz[0], Q); Q.wait();
    int * __restrict__ col          =  sycl::malloc_shared<int>(sizeof(int)*nnz[0], Q); Q.wait();
    TYPE * __restrict__ value       =  sycl::malloc_shared<TYPE>(sizeof(TYPE)*nnz[0], Q); Q.wait();
    int * __restrict__ row_offset   = sycl::malloc_shared<int>(sizeof(int)*(size+1),Q); Q.wait();

    int j = 0;
    for (size_t i = 0; i < N*N; i++)
    { 
        if(m1[i] != 0)
        {
            row[j] = (int)((i-(i%N))/N);
            col[j] = i%N;
            value[j] = m1[i];
            j++;
            
        }
    }

    Q.wait();

    row_offset[0] = 0;
    row_offset[size] = *nnz;

    int a = 0;
    
    for (int i =0 ; i < *nnz; i++)
    {
        if(row[i] != row[i+1])
        {
            a++;
            row_offset[a] = i+1;
        }
    }
    Q.wait();
    
    sycl::range<1> global1{static_cast<size_t>(N)};
    sycl::range<1> local1{N_b};

    #pragma omp parallel
    {
        LIKWID_MARKER_START("SPMV");
    }
    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<1>(global1,local1), [=](sycl::nd_item<1>it){

            const int k = it.get_global_id(0);
            TYPE temp = 0.0;

            int start = row_offset[k];
            int end = row_offset[k+1];

            for (int j = start; j < end;j++)
            {
                temp += value[j]*v1[col[j]];
                
            }
            v2[k] = temp;
        });
    });
    Q.wait();

    time.end_timer();
    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("SPMV");
    }

    if (verification_spmv(m1, v1 , v2, size))
    {
        std::cout << "Verification Failed" << std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : SPMV with ndrange ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(v1,Q);
    sycl::free(v2,Q);
    sycl::free(m1,Q);

}

////////////////////////////////////////////////////////// stencil

void stencil_1_ndrange_usm(sycl::queue &Q, int size, int block_size){
    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    timer time;
#ifdef ALIGNED
    TYPE * __restrict__ m1 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*sizeof(TYPE),Q); Q.wait();
#else
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
#endif
    init_sparse_arrays(m1, size, 1); Q.wait();
    std::fill(m2,m2+size*size,0.0); Q.wait();

    sycl::range<2> global1 {N,N};
    
    sycl::range<2> local1{N_b,N_b};

    time.start_timer();

    Q.submit([&](sycl::handler &cgh){
        sycl::accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local>  localm1(sycl::range<2>(N, N), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global1, local1), [=](sycl::nd_item<2> it){
            auto k = it.get_global_id(0);
            auto k1 = it.get_global_id(1);

            if (k == 0 )
            {
                localm1[k][k1] = m1[k*N+k1];
            }
            else if (k1 ==N)
            {
                localm1[k][k1] = m1[k*N+k1];
            }
            else if (k1 ==0)
            {
                localm1[k][k1] = m1[k*N+k1];
            }
            else if (k == N)
            {
                localm1[k][k1] = m1[k*N+k1];
            }
            else
            {
                localm1[k][k1] = m1[k*N+k1]+m1[(k+1)*N+k1]+m1[(k-1)*N+k1]+m1[k*N+(k1+1)]+m1[k*N+(k1-1)];
            }
            
            
        });
    });
    Q.wait();

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : stencil 1 with ndrange ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(m2,Q);
}