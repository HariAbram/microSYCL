#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>
#include <string>


#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"
#include "../include/parallel-bench.hpp"
#include "../include/kernels.hpp"
#include "../include/utils.hpp"

using namespace cl;

////////////////////////////////////////////////////////////////////////////////////////////////
// memory allocations

void host_memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter)
{

    timer time;

    timer time1;
    
    int i;

    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        sycl::free(m_host,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();

    }

    if (print)
    {
        print_results(timings_alloc, iter, size, "Host memory alloc(ms)",1, 1);
    }
   
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);
  
    auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
    auto a_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

    sycl::range<1> global{N*N};

    init_arrays(Q, m_host, a_host, global);

    for (size_t i = 0; i < iter; i++)
    {   
        time1.start_timer();
        kernel_copy(Q, m_host, a_host, global);
        time1.end_timer();

        timings[i] = time1.duration();
    }

    if (print)
    {
        print_results(timings, iter, size, "Host memory (r)",1, 1);
    }
    
    ///////////////////////////////////////////

    Q.wait();

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    init_arrays(Q, m_host, a_host, global);

    for (size_t i = 0; i < iter; i++)
    {
        time1.start_timer();
        kernel_copy(Q,m_host,a_host,global,local);
        time1.end_timer();
        timings_nd[i] = time1.duration();   
    }
    
    sycl::free(m_host,Q);
    sycl::free(a_host,Q);

    if (print)
    {
        print_results(timings_nd, iter, size, "Host memory (ndr)",1, 1);
    }

    free(timings);
    free(timings_alloc);
    free(timings_nd);

}

void shared_memory_alloc(sycl::queue &Q, int size, int block_size ,bool print, int iter)
{

    timer time;

    timer time1;

    int i;

    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
        sycl::free(m_shared,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();
    }

    if (print)
    {
        print_results(timings_alloc, iter, size, "Shared memory alloc(ms)",1, 1);
    }
    

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    sycl::range<1> global{N*N};
    auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
    auto a_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
    
    init_arrays(Q, m_shared, a_shared, global);

    for (size_t i = 0; i < iter; i++)
    {
        time1.start_timer();
        kernel_copy(Q, m_shared, a_shared, global);
        time1.end_timer();

        timings[i] = time1.duration();
    }

    if (print)
    {
        print_results(timings, iter, size, "Shared memory (r)",1, 1);
    }

    ///////////////////////////////////////////////

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    init_arrays(Q, m_shared, a_shared, global);
    
    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q,m_shared,a_shared,global,local);
        time1.end_timer();

        timings_nd[i] = time1.duration();
        
    }

    sycl::free(m_shared,Q);
    sycl::free(a_shared,Q);

    if (print)
    {
        print_results(timings_nd, iter, size, "Shared memory (ndr)",1, 1);
    }

    free(timings);
    free(timings_alloc);
    free(timings_nd);

}

void device_memory_alloc(sycl::queue &Q, int size, int block_size ,bool print, int iter)
{

    timer time;
    timer time1;

    int i;
    
    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
        sycl::free(m_device,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();

    }

    if (print)
    {
        print_results(timings_alloc, iter, size, "Device memory alloc(ms)",1, 1);
    }

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    sycl::range<1> global{N*N};
    auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
    auto a_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();

    init_arrays(Q, m_device, a_device, global);
    
    Q.wait();

    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q, m_device, a_device, global);
        time1.end_timer();

        timings[i] = time1.duration();
        
    }

    if (print)
    {
        print_results(timings, iter, size, "Device memory (r)",1, 1);
    }

    /////////////////////////////////////////////////

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    init_arrays(Q, m_device, a_device, global);

    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q,m_device,a_device,global,local);
        time1.end_timer();

        timings_nd[i] = time1.duration();    
    }

    sycl::free(m_device,Q);
    sycl::free(a_device,Q);

    if (print)
    {
        print_results(timings_nd, iter, size, "Device memory (ndr)",1, 1);
    }

    free(timings);
    free(timings_alloc);
    free(timings_nd);

}


// sycl::range constuct


void range_with_usm(sycl::queue &Q, int size, int dim, bool print, int iter)
{

    timer time;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size*size),0.0);

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_1(Q, sum, global);
            time.end_timer();

            timings[i] = time.duration();
            
        }
        
        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "range_USM", 1, 2);
        }
        sycl::free((TYPE*)sum,Q);

    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_2(Q, sum, global);
            time.end_timer();

            timings[i] = time.duration();
        }

        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "range_USM", 2, 2);
        }

        sycl::free((TYPE*)sum,Q);
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
        sycl::free((TYPE*)sum,Q);
    }
    
    
    
}

// sycl::nd_range constuct


void nd_range_with_usm(sycl::queue &Q, int size, int block_size ,int dim, bool print, int iter)
{

    timer time;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size*size),0.0);

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        int i;

        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<1> local{N_b};

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_1(Q, sum, global, local);
            time.end_timer();

            timings[i] = time.duration();
        }

        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value "<< sum[1]
                      <<std::endl;
        }


        if (print)
        {
            print_results(timings, iter, size, "ndrange_USM", 1, 2);
        }
        sycl::free(sum,Q);   
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        int i;

        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<2> local{N_b,N_b};

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_2(Q, sum, global, local);
            time.end_timer();

            timings[i] = time.duration();
        }

        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[1]
                      << std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "ndrange_USM", 2, 2);
        }
        sycl::free(sum,Q);   
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }
    
     
    
} 


// reduction 

void atomics_usm(sycl::queue &Q, int size, bool print, int iter)
{
    timer time;

    size = size*size;

    auto m_shared = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    std::fill(m_shared,m_shared+size,1.0); Q.wait();
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();
    sum[0] = 0.0;

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        kernel_atomics(Q, global, m_shared, sum);
        time.end_timer();

        timings[i] = time.duration();
    }   

    if (sum[0]!= size*iter)
    {
        std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[0]
                      <<std::endl;
    }

    if (print)
    {
        print_results(timings, iter, size, "atomics USM", 1, 3);
    }
    
    sycl::free(m_shared,Q);
    sycl::free(sum, Q);
    free(timings);
}


void reduction_with_usm(sycl::queue &Q, int size, int block_size, bool print, int iter)
{
    timer time;

    auto m_shared = (TYPE *)sycl::malloc_shared(size*size*sizeof(TYPE), Q);
    std::fill(m_shared,m_shared+size*size,1.0); Q.wait();
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();
    sum[0] = 0.0;

    auto N = static_cast<size_t>(size*size);

    sycl::range<1> global{N};
        
    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        kernel_reduction(Q, sum, m_shared, global);

        time.end_timer();

        timings[i] = time.duration();
    }   

    if (print)
    {
        print_results(timings, iter, size, "Reduction USM", 1, 3);
    }   
    
    sycl::free(m_shared,Q);
    sycl::free(sum,Q);
}


void group_barrier_test_usm(sycl::queue &Q, int size, int block_size, bool print, int iter)
{
    
    timer time;

    size = size*size;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size),0);
    
    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        kernel_group_barrier(Q, sum, global, local);
        time.end_timer();

        timings[i] = time.duration();
    }

    if (sum[0]!= 1024*iter)
    {
        std::cout << "Verification failed "
                  << "Expected value "<< 1024*iter
                  << "Final value"<< sum[0]
                  <<std::endl;
    }

    if (print)
    {
        print_results(timings, iter, size, "Group barrier USM", 1, 4);
    }

    sycl::free(sum,Q);
}


void subgroup_barrier_test_usm(sycl::queue &Q, int size, int block_size, bool print, int iter)
{

    timer time;

    size = size*size;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size),0);

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {

        time.start_timer();
        kernel_subgroup_barrier(Q, sum, global, local);
        time.end_timer();

        timings[i] = time.duration();
    }

    if (sum[0]!= 1024*iter)
    {
        std::cout << "Verification failed "
                  << "Expected value "<< 1024*iter
                  << "Final value"<< sum[0]
                  <<  std::endl;
    }

    if (print)
    {
        print_results(timings, iter, size, "Subgroup barrier USM", 1, 4);
    }
    sycl::free(sum,Q);

}
