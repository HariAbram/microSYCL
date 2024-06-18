#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <omp.h>

#ifndef TYPE
#define TYPE double
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


void vec_add_range_usm(sycl::queue &Q, int size)
{


    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v3 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

    
    std::fill(v1,v1+size,1);
    std::fill(v2,v2+size,1);
    std::fill(v3,v3+size,1);

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};
   

    Q.parallel_for<class vec_add_r_usm>(sycl::range<1>(global), [=](sycl::item<1>it){

        auto i = it.get_id(0);

        v3[i] = v2[i] + v1[i];

    });
    Q.wait();

    free(v1,Q);
    free(v2,Q);
    free(v3,Q);

}


void vec_add_range_buff_acc(sycl::queue &Q, int size)
{

    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v3 = (TYPE *)malloc(size*sizeof(TYPE));

    std::fill(v1,v1+size,1);
    std::fill(v1,v1+size,1);

    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);
    sycl::buffer<TYPE,1> v3_buff(v3,size);

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};
    
    Q.submit([&](sycl::handler& cgh){
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read>(cgh);
        auto v3_acc = v3_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class vec_add_r_ba>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto i = it.get_id(0);

            v3_acc[i] = v2_acc[i] + v1_acc[i];

        });

    });
    Q.wait();

    free(v1);
    free(v2);
    free(v3);

}

void vec_add_ndrange_usm(sycl::queue &Q, int size, int block_size)
{

    TYPE * __restrict__ v1 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v2 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ v3 = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

    std::fill(v1,v1+size,1);
    std::fill(v1,v1+size,1);
    
    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    Q.parallel_for<class vec_add_ndr_usm>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

        auto i = it.get_global_id(0);

        v3[i] = v2[i] + v1[i];

    });
    Q.wait();

    free(v1,Q);
    free(v2,Q);
    free(v3,Q);

}

void vec_add_ndrange_buff_acc(sycl::queue &Q, int size, int block_size)
{

    TYPE * __restrict__ v1 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v2 = (TYPE *)malloc(size*sizeof(TYPE));
    TYPE * __restrict__ v3 = (TYPE *)malloc(size*sizeof(TYPE));

    std::fill(v1,v1+size,1);
    std::fill(v1,v1+size,1);

    sycl::buffer<TYPE,1> v1_buff(v1,size);
    sycl::buffer<TYPE,1> v2_buff(v2,size);
    sycl::buffer<TYPE,1> v3_buff(v3,size);

    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};
   
    Q.submit([&](sycl::handler& cgh){
        auto v1_acc = v1_buff.get_access<sycl::access::mode::read>(cgh);
        auto v2_acc = v2_buff.get_access<sycl::access::mode::read>(cgh);
        auto v3_acc = v3_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class vec_add_ndr_ba>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            auto i = it.get_global_id(0);

            v3_acc[i] = v2_acc[i] + v1_acc[i];

        });

    });
    Q.wait();

    free(v1);
    free(v2);
    free(v3);


}




void mat_mul_range_usm(sycl::queue &Q, int size)
{
    timer time;

    auto N = static_cast<size_t>(size);
    

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);

    //auto N_m = static_cast<size_t>(size*size);

    sycl::range<2> global1 {N,N};

    time.start_timer();


    Q.parallel_for< >(sycl::range<2>(global1), [=](sycl::item<2>it){

        auto i = it.get_id(0);
        auto j = it.get_id(1);

        float temp = 0.0;

        for (size_t k = 0; k < N; k++)
        {
            temp += m2[i*N+k]*m1[k*N+j];
        }

        m3[i*N+j] = temp;
        

    });



    Q.wait();

    time.end_timer();

    if (m3[0]!= size)
    {
        std::cout<< "Verification Failed" << std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken for mat mul for range with USM "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;


    free(m1,Q);
    free(m2,Q);
    free(m3,Q);



}



void mat_mul_range_buff_acc(sycl::queue &Q, int size)
{

    auto N = static_cast<size_t>(size);

    timer time;  
    

    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m2 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m3 = (TYPE *)malloc(size*size*sizeof(TYPE));

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);

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

    auto m3_r = m3_buff.get_access<sycl::access::mode::read>();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
    

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken for mat mul for range with buff and acc "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;


    free(m1);
    free(m2);
    free(m3);



}



void mat_mul_ndrange_usm(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    timer time;

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);

    //auto N_m = static_cast<size_t>(size*size);

    sycl::range<2> global1 {N,N};
    sycl::range<2> local1{N_b,N_b};

    time.start_timer();


    Q.parallel_for< >(sycl::nd_range<2>(global1,local1), [=](sycl::nd_item<2>it){

        auto i = it.get_global_id(0);
        auto j = it.get_global_id(1);

        TYPE temp = 0.0;

        for (size_t k = 0; k < N; k++)
        {
            temp+= m2[i*N+k]*m1[k*N+j];
        }

        m3[i*N+j] = temp;

    });



    
    Q.wait();

    time.end_timer();

    if (m3[0]!= size)
    {
        std::cout<< "Verification Failed" << std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken for mat mul for nd_range with USM  "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;


    free(m1,Q);
    free(m2,Q);
    free(m3,Q);



}



void mat_mul_ndrange_buff_acc(sycl::queue &Q, int size, int block_size)
{

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    timer time;

    TYPE * __restrict__ m1 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m2 = (TYPE *)malloc(size*size*sizeof(TYPE));
    TYPE * __restrict__ m3 = (TYPE *)malloc(size*size*sizeof(TYPE));

    std::fill(m1,m1+size*size,1);
    std::fill(m2,m2+size*size,1);

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

    auto m3_r = m3_buff.get_access<sycl::access::mode::read>();

    if (m3_r[0] != size)
    {
        std::cout << "Verification Failed" << std::endl;
    }
    

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken for mat mul for nd_range with buff and acc  "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    free(m1);
    free(m2);
    free(m3);



}