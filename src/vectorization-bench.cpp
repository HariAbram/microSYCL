#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <vector>

#ifndef TYPE
#define TYPE double
#endif

#ifndef OPT_BLOCK_SIZE
#define OPT_BLOCK_SIZE 32
#endif

#include "../include/timer.hpp"

using namespace cl;
using shared_allocator = sycl::usm_allocator<TYPE, sycl::usm::alloc::shared>;

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

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat vec with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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
    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat vec with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat vec with ndrange ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat vec with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    if (m3[0]!= size)
    {
        std::cout<< "Verification Failed" << std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat mul with range ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    auto m3_r = m3_buff.get_host_access();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
    

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat mul with range ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);
    std::fill(m3,m3+size*size,0.0);

    //auto N_m = static_cast<size_t>(size*size);

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

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

    if (m3[0]!= size)
    {
        std::cout<< "Verification Failed" << std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat mul with nd_range ( USM ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    auto m3_r = m3_buff.get_host_access();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
    

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat mul with nd_range( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

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

    time.start_timer();

 
    Q.submit([&](sycl::handler& cgh){
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read>(cgh);
        auto m2_acc = m2_buff.get_access<sycl::access::mode::read>(cgh);
        auto m3_acc = m3_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);

            TYPE temp = 0.0;

            for (size_t k = 0; k < N; k+=OPT_BLOCK_SIZE)
            {
                for (size_t kk = 0; kk < OPT_BLOCK_SIZE; kk++)
                {
                    temp += m2_acc[i*N+kk]*m1_acc[kk*N+j];
                }
                
                
            }

            m3_acc[i*N+j] = temp;

        });

    });
    Q.wait();

    time.end_timer();

    auto m3_r = m3_buff.get_host_access();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
    

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : mat mul with nd_range( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(m1);
    free(m2);
    free(m3);
}


////////////////////////////////////////////////////////// outer-product

void outer_product(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);

    timer time;

    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE));

    std::fill(m1,m1+size*size,0.0);
    std::fill(v1,v1+size,1);
    std::fill(v2,v2+size,1);

    sycl::buffer<TYPE,1> m1_buff(m1,size*size);
    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

    time.start_timer();
 
    Q.submit([&](sycl::handler& cgh){
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);
            auto N = it.get_global_range(0);

            m1_acc[i*N+j] = v1_acc[i]*v2_acc[j];
        });
    });
    Q.wait();

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : outer product with ndrange ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(m1);
    free(v1);
    free(v2);
}

////////////////////////////////////////////////////////// triad

void triad(sycl::queue &Q, int size, int block_size)
{
    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);

    timer time;

    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE)); 
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE)); 
    TYPE * __restrict__ v3 = (TYPE *)malloc(size*sizeof(TYPE)); 

    std::fill(v1,v1+size,1.0);
    std::fill(v2,v2+size,2.0);
    std::fill(v3,v3+size,0.0);

    sycl::range<1> global1{N};
    sycl::range<1> local1{N_b};

    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);
    sycl::buffer<TYPE,1> v3_buff(v3,size);

    time.start_timer();

    Q.submit([&](sycl::handler& cgh){
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read>(cgh);
        auto v3_acc = v3_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for< >(sycl::nd_range<1>(global1,local1), [=](sycl::nd_item<1>it){

            auto i = it.get_global_id(0);

            v3_acc[i]  = v2_acc[i] + v1_acc[i]* N;
        });
    });
    Q.wait();

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : triad with ndrange ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(v1);
    free(v2);
    free(v3);

}

////////////////////////////////////////////////////////// cross-product

void cross_product(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);

    timer time;

    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE));

    std::fill(m1,m1+size*size,0.0);
    std::fill(v1,v1+size,1);
    std::fill(v2,v2+size,1);

    sycl::buffer<TYPE,1> m1_buff(m1,size*size);
    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

    time.start_timer();
 
    Q.submit([&](sycl::handler& cgh){
        auto m1_acc = m1_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

            auto i = it.get_global_id(0);
            auto j = it.get_global_id(1);
            auto N = it.get_global_range(0);
            auto tmp= std::pow(-1, i+j);

            if (i == j)
            {
                m1_acc[i*N+j] = 0;
            }
            else
            {
                m1_acc[i*N+j] = v1_acc[i]*v2_acc[j]*tmp;
            }
            
        });
    });
    Q.wait();

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : cross product with ndrange ( buff and acc ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(m1);
    free(v1);
    free(v2);
}