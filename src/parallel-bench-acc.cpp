#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
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


////////////////////////////////////////////////////////////////////////////////////////////////
// memory allocations

void memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter)
{
    timer time;

    int i;

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);
    sycl::range<1> global{N*N};

    sycl::buffer<TYPE , 1> m_buff(global);
    sycl::buffer<TYPE , 1> a_buff(global);

    init_arrays(Q, m_buff, a_buff, global);

    for (i = 0; i < iter; i++)
    {
        time.start_timer();
        kernel_copy(Q, m_buff, a_buff, global);
        time.end_timer();

        timings[i] = time.duration();
    }

    if (print)
    {
        print_results(timings, iter, size, "memory B&A", 1, 1);
    }

    free(timings);

}

// sycl::range constuct

void range_with_buff_acc(sycl::queue &Q, int size, int dim, bool print, int iter)
{

    timer time;

    TYPE * sum = (TYPE *)std::malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0.0);
    
    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            kernel_parallel_1(Q, sum_buff, global);

            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "range_BA", dim, 2);
        }        
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_2(Q, sum_buff, global);
            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "range_BA", dim, 2);
        }
    } 
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }

    free((TYPE*)sum);
         
}

// sycl::nd_range constuct

void nd_range_with_buff_acc(sycl::queue &Q, int size, int block_size ,int dim, bool print, int iter)
{

    timer time;

    TYPE * sum = (TYPE *)malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0.0);
   
    auto N = static_cast<size_t>(size);
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }

    sycl::range<1> global{N};    

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        int i;
        sycl::range<1> local{N_b*N_b};

        sycl::buffer<TYPE , 1> sum_buff(sum,size*size);        

        for ( i = 0; i < iter; i++)
        {

            time.start_timer();
            kernel_parallel_1(Q, sum_buff, global, local);
            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<  std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "ndrange_BA", dim, 2);
        }
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        int i;
        sycl::range<2> local{N_b,N_b};

        sycl::buffer<TYPE , 1> sum_buff(sum,size*size);

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_2(Q, sum_buff,global, local);
            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "ndrange_BA", dim, 2);
        }
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }

    free(sum);
    
}

// reduction 

void atomics_buf_acc(sycl::queue &Q, int size, bool print, int iter)
{
    timer time;

    size = size*size;

    auto m = (TYPE *)std::malloc(size*sizeof(TYPE)); 
    std::fill(m,m+size,1.0);
    auto sum = (TYPE *)std::malloc(1*sizeof(TYPE)); 
    sum[0] = 0.0;

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};

    sycl::buffer<TYPE , 1> m_buff(m,size);

    sycl::buffer<TYPE , 1> sum_buff(sum,1);
    
    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        kernel_atomics(Q,  global,  m_buff, sum_buff);
        time.end_timer();

        timings[i] = time.duration();
    }   
    
    auto sum_r = sum_buff.get_host_access();

    if (sum_r[0]!= size*iter)
    {
        std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[0]
                      <<std::endl;
    }

    if (print)
    {
        print_results(timings, iter, size, "atomics BA", 1, 3);
    }
    
    
    free(m);
    free(sum);
}

void reduction_with_buf_acc(sycl::queue &Q, int size, int block_size, bool print, int iter)
{
    timer time;

    auto m_shared = (TYPE *)std::malloc(size*sizeof(TYPE));
    std::fill(m_shared,m_shared+size,1.0);
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();
    sum[0] = 0.0;

    auto N = static_cast<size_t>(size);
    auto N_b = static_cast<size_t>(block_size);

    sycl::range<1> global{N};
    sycl::range<1> local{N_b};

    sycl::buffer<TYPE , 1> m_buff(m_shared,size);

    sycl::buffer<TYPE , 1> sum_buff(sum,1);
        
    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    #pragma omp parallel
    {
        LIKWID_MARKER_START("reduction_buf_acc");
    }

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        kernel_reduction(Q, sum_buff, m_buff, global);
        Q.wait();
        time.end_timer();

        timings[i] = time.duration();
    }   

    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("reduction_buf_acc");
    }

    if (print)
    {
        print_results(timings, iter, size, "Reduction BA", 1, 3);
    }   
    
    free(m_shared);
}

// barrier
void group_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size, bool print, int iter, int dim)
{

    timer time;

    TYPE * sum = (TYPE *)malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0);
    sycl::buffer<TYPE , 1> sum_buff(sum,size*size);

    auto N = static_cast<size_t>(size);
    
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        sycl::range<1> local{N_b*N_b};

        int i;
        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_group_barrier(Q, sum_buff, global, local);
            time.end_timer();

            timings[i] = time.duration();
        }
        auto sum_r = sum_buff.get_host_access();

        if (sum_r[0]!= 1024*iter)
        {
            std::cout << "Verification failed "
                    << "Expected value "<< 1024*iter
                    << "Final value"<< sum_r[0]
                    <<std::endl;
        }
        if (print)
        {
            print_results(timings, iter, size, "Group barrier BA", dim, 4);
        }
        free(sum);
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        sycl::range<2> local{N_b,N_b};

        int i;
        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_group_barrier(Q, sum_buff, global, local);
            time.end_timer();

            timings[i] = time.duration();
        }
        auto sum_r = sum_buff.get_host_access();

        if (sum_r[0]!= 1024*iter)
        {
            std::cout << "Verification failed "
                    << "Expected value "<< 1024*iter
                    << "Final value"<< sum_r[0]
                    <<std::endl;
        }
        if (print)
        {
            print_results(timings, iter, size, "Group barrier BA", dim, 4);
        }
        free(sum);
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }     

}

void subgroup_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size, bool print, int iter, int dim)
{

    timer time;

    TYPE * sum = (TYPE *)malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0);
    sycl::buffer<TYPE , 1> sum_buff(sum,size*size);
    
    auto N = static_cast<size_t>(size);
    
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        sycl::range<1> local{N_b*N_b};
        int i;
        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_subgroup_barrier(Q, sum_buff,global, local);
            time.end_timer();

            timings[i] = time.duration();
        }
        auto sum_r = sum_buff.get_host_access();

        if (sum_r[0]!= 1024*iter)
        {
            std::cout << "Verification failed "
                    << "Expected value "<< 1024*iter
                    << "Final value"<< sum_r[0]
                    << std::endl;
        }
        if (print)
        {
            print_results(timings, iter, size, "Subgroup barrier BA", dim, 4);
        }
        free(sum);
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        sycl::range<2> local{N_b,N_b};
        int i;
        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_subgroup_barrier(Q, sum_buff,global, local);
            time.end_timer();

            timings[i] = time.duration();
        }
        auto sum_r = sum_buff.get_host_access();

        if (sum_r[0]!= 1024*iter)
        {
            std::cout << "Verification failed "
                    << "Expected value "<< 1024*iter
                    << "Final value"<< sum_r[0]
                    << std::endl;
        }
        if (print)
        {
            print_results(timings, iter, size, "Subgroup barrier BA", dim, 4);
        }
        free(sum);
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }
}